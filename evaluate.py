import argparse
import sys
import os

import torch

from models.resRnn import resRnn
from utils.logger import print_normal, print_warning, print_error, TerminalColors
from utils.trainer import CPUParallel, MovingAverage
from rating.word_error_rate import levenshtein
from utils.dataset import FileDataset


def main():
    parser = argparse.ArgumentParser(description="SOCR Text Recognizer")
    parser.add_argument('paths', metavar='N', type=str, nargs='+')
    parser.add_argument('--model', type=str, default="resRnn", help="Model name")
    parser.add_argument('--name', type=str, default="resRnn")
    parser.add_argument('--disablecuda', action='store_const', const=True, default=False)
    args = parser.parse_args()

    with open("characters.txt", "r") as content_file:
        lst = content_file.read() + " "

    labels = {"": 0}
    for i in range(0, len(lst)):
        labels[lst[i]] = i + 1

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

    if not args.disablecuda:
        print_normal("Using GPU Data Parallel")
        model = torch.nn.DataParallel(model)
    else:
        model = CPUParallel(model)

    checkpoint_name = "checkpoints/" + args.name + ".pth.tar"

    if os.path.exists(checkpoint_name):
        print_normal("Restoring the weights...")
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise FileNotFoundError()

    data_set = FileDataset(image_height)
    for path in args.paths:
        data_set.recursive_list(path)
    data_set.sort()

    print(str(data_set.__len__()) + " files")

    loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)
    count = 0

    for i, data in enumerate(loader, 0):
        image, path = data

        percent = i * 100 // data_set.__len__()
        print(str(percent) + "%... Processing " + path[0])

        if not args.disablecuda:
            result = model(torch.autograd.Variable(image.float().cuda()))
        else:
            result = model(torch.autograd.Variable(image.float().cpu()))

        text = loss.ytrue_to_lines(result.cpu().detach().numpy())
        print(text)

        count = count + 1




if __name__ == '__main__':
    main()
