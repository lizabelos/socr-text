import argparse
import sys
import os

import torch

from models.resRnn import resRnn
from utils.logger import print_normal, print_warning, print_error, TerminalColors
from dataset.iam_handwriting_line_database import IAMHandwritingLineDatabase
from utils.trainer import CPUParallel, MovingAverage
from rating.word_error_rate import levenshtein
from coder.language.language_model import LanguageModel
from coder.language.word_beam_search import wordBeamSearch


def main():
    parser = argparse.ArgumentParser(description="SOCR Text Recognizer")
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--model', type=str, default="resRnn", help="Model name")
    parser.add_argument('--name', type=str, default="resRnn")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--clipgradient', type=float, default=None)
    parser.add_argument('--epochlimit', type=int, default=None)
    parser.add_argument('--overlr', action='store_const', const=True, default=False)
    parser.add_argument('--disablecuda', action='store_const', const=True, default=False)
    parser.add_argument('--iamtrain', type=str)
    parser.add_argument('--iamtest', type=str, default=None)
    parser.add_argument('--generated', action='store_const', const=True, default=False)
    args = parser.parse_args()

    assert args.iamtrain is not None

    with open("characters.txt", "r") as content_file:
        characters = content_file.read() + " "
        lst = characters
        labels = {"": 0}
        for i in range(0, len(lst)):
            labels[lst[i]] = i + 1

    with open("word_characters.txt", "r") as content_file:
        word_characters = content_file.read()

    with open("dictionnary.txt", "r") as content_file:
        dictionnary = content_file.read()

    lm = LanguageModel(dictionnary, characters, word_characters)

    model = resRnn(labels)
    loss = model.create_loss()

    if not args.disablecuda:
        model = model.cuda()
        loss = loss.cuda()
    else:
        print_warning("Using the CPU")
        model = model.cpu()
        loss = loss.cpu()

    image_height = model.get_input_image_height()

    print_normal("Using Adam with a Learning Rate of " + str(args.lr))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    adaptative_optimizer = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)

    os.makedirs('checkpoints', exist_ok=True)

    if not args.disablecuda:
        print_normal("Using GPU Data Parallel")
        model = torch.nn.DataParallel(model)
    else:
        model = CPUParallel(model)

    checkpoint_name = "checkpoints/" + args.name + ".pth.tar"

    epoch = 0

    if os.path.exists(checkpoint_name):
        print_normal("Restoring the weights...")
        checkpoint = torch.load(checkpoint_name)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        adaptative_optimizer.load_state_dict(checkpoint['adaptative_optimizer'])
    else:
        print_warning("Can't find '" + checkpoint_name + "'")

    if args.overlr is not None:
        print_normal("Overwriting the lr to " + str(args.lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    train_databases = [IAMHandwritingLineDatabase(args.iamtrain, height=image_height, loss=loss)]

    if args.generated:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "submodules/scribbler"))
        from scribbler.generator import LineGenerator
        train_databases.append(LineGenerator(height=image_height, loss=loss))

    train_database = torch.utils.data.ConcatDataset(train_databases)


    test_database = None
    if args.iamtest is not None:
        test_database = IAMHandwritingLineDatabase(args.iamtest, height=image_height, loss=loss)

    moving_average = MovingAverage(max(train_database.__len__() // args.bs, 1024))

    try:
        while True:
            if args.epochlimit is not None and epoch > args.epochlimit:
                print_normal("Epoch " + str(args.epochlimit) + "reached !")
                break

            model.train()

            loader = torch.utils.data.DataLoader(train_database, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=collate)
            for i, data in enumerate(loader, 0):

                inputs, labels = data

                optimizer.zero_grad()

                variable = torch.autograd.Variable(inputs).float()

                if not args.disablecuda:
                    variable = variable.cuda()
                else:
                    variable = variable.cpu()

                outputs = model(variable)
                loss_value = loss.forward(outputs, labels)
                loss_value.backward()

                loss_value_cpu = loss_value.data.cpu().numpy()

                if args.clipgradient is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipgradient)

                optimizer.step()

                loss_value_np = float(loss_value.data.cpu().numpy())
                moving_average.addn(loss_value_np)

                if (i * args.bs) % 8 == 0:
                    sys.stdout.write(TerminalColors.BOLD + '[%d, %5d] ' % (epoch + 1, (i * args.bs) + 1) + TerminalColors.ENDC)
                    sys.stdout.write('lr: %.8f; loss: %.4f ; curr: %.4f ;\r' % (optimizer.state_dict()['param_groups'][0]['lr'], moving_average.moving_average(), loss_value_cpu))

            epoch = epoch + 1
            adaptative_optimizer.step()

            sys.stdout.write("\n")

            if args.iamtest is not None:
                test(model, lm, loss, test_database)

    except KeyboardInterrupt:
        pass

    print_normal("Done training ! Saving...")
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'adaptative_optimizer': adaptative_optimizer.state_dict(),
    }, checkpoint_name)


def test(model, lm, loss, test_database, limit=32):
    """
    Test the network

    :param limit: Limit of images of the test
    :return: The average cer
    """
    model.eval()

    is_cuda = next(model.parameters()).is_cuda

    loader = torch.utils.data.DataLoader(test_database, batch_size=1, shuffle=False, num_workers=1)

    test_len = len(test_database)
    if limit is not None:
        test_len = min(limit, test_len)

    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
    cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0

    sen_err = 0
    count = 0

    for i, data in enumerate(loader, 0):
        image, label = data
        label = label[1][0]

        if image.shape[2] < 8:
            continue

        if is_cuda:
            result = model(torch.autograd.Variable(image.float().cuda()))
        else:
            result = model(torch.autograd.Variable(image.float().cpu()))

        # text = loss.ytrue_to_lines(result.cpu().detach().numpy())
        text = wordBeamSearch(result[0].data.cpu().numpy(), 32, lm, False)

        # update CER statistics
        _, (s, i, d) = levenshtein(label, text)
        cer_s += s
        cer_i += i
        cer_d += d
        cer_n += len(label)
        # update WER statistics

        _, (s, i, d) = levenshtein(label.split(), text.split())
        wer_s += s
        wer_i += i
        wer_d += d
        wer_n += len(label.split())
        # update SER statistics
        if s + i + d > 0:
            sen_err += 1

        count = count + 1

        sys.stdout.write("Testing..." + str(count * 100 // test_len) + "%\r")

        if count == test_len:
            break

    cer = (100.0 * (cer_s + cer_i + cer_d)) / cer_n
    wer = (100.0 * (wer_s + wer_i + wer_d)) / wer_n
    ser = (100.0 * sen_err) / count

    print_normal("CER : %.3f; WER : %.3f; SER : %.3f \n" % (cer, wer, ser))


def collate(batch):
    data = [item[0] for item in batch]
    max_width = max([d.size()[2] for d in data])

    data = [torch.nn.functional.pad(d, (0, max_width - d.size()[2], 0, 0)) for d in data]
    data = torch.stack(data)

    target = [item[1] for item in batch]

    return [data, target]


if __name__ == '__main__':
    main()
