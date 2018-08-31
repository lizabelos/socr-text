# SOCR TEXT

## Requirements

 - Python3 with Cython
 - Pytorch 0.4 : ```conda install pytorch=0.4 -c pytorch```
 - OpenCV 3 : ```conda install opencv```
 - scikit-image :  ```conda install -c conda-forge scikit-image```
 - All in requirements.txt : ```pip install -r requirements.txt```

## Compilation

SOCR Text was created with Cython. To compile it, run : 
```
python3 setup.py build_ext --inplace
```

## Training

To train the network, run :
```
python3 train.py --iamtrain [train_path]
```

If you want to enable test during the training, use the commande line argument ```--iamtest```.

Use the ```--help``` argument for more arguments, like the batch size or the learning rate.

```
usage: train.py [-h] [--bs BS] [--model MODEL] [--name NAME] [--lr LR]
                [--clipgradient CLIPGRADIENT] [--epochlimit EPOCHLIMIT]
                [--overlr] [--disablecuda] [--iamtrain IAMTRAIN]
                [--iamtest IAMTEST] [--generated]

SOCR Text Recognizer

optional arguments:
  -h, --help            show this help message and exit
  --bs BS               Batch size
  --model MODEL         Model name
  --name NAME           Name for this training
  --lr LR               Learning rate
  --clipgradient CLIPGRADIENT
                        Gradient clipping
  --epochlimit EPOCHLIMIT
                        Limit the training to a number of epoch
  --overlr              Override the learning rate
  --disablecuda         Disable cuda
  --iamtrain IAMTRAIN   IAM Training Set
  --iamtest IAMTEST     IAM Testing Set
  --generated           Enable generated data
```

## Evaluate

To evaluate the network, where path is a directory or a image file, run : 
```
python3 evaluate.py path
```
The result will be printed in the terminal.

## Dataset

This is the link to IAM Handwriting Database :

[IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

The ```lines``` folder, and the ```xml``` folder need to be placed into the same folder, for example name ```train```. So you will have a folder ```train/lines``` and ```train/xml```. This is the path to the train folder which need to be specified to the --iamtrain command line.

If you want to enable test during the training, you have to split yourself the dataset into a train part and a test part.

## How to use a custom dataset

1. Create a file ```my_custom_dataset.py``` in the dataset directory.
2. This file must contains a class ```MyCustomDataset``` inheriting from ```torch.utils.data.dataset.Dataset```. You must implements the ```__getitem__``` and ```__len__``` function of this class.
3. The ```__getitem__``` function must return ```image, (preprocessed_text, text, width)``` where : 
    1. ```image``` is the torch text image, resized to the input height of the model.
    2. ```preprocessed_text``` is the text preprocessed by ```loss.preprocess_label```
    3. ```text``` is the text of the image
    4. ```width``` is the width of the resized image


```python
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.image import image_pillow_to_numpy

class MyCustomDataset(Dataset):

	def __init__(self, path, height, loss):
		self.height = height
		self.loss = loss
		
		# self.items = ...
		
	def __len__(self):
		return len(self.items)
		
	def __getitem__(self, index):
		item = self.items[index]
		
		# image_path = ...
		# image_text = ...
		
		image = Image.open(image_path).convert('RGB')
		width, height = image.size
		image = image.resize((width * self.height // height, self.height), Image.ANTIALIAS)
		image = image_pillow_to_numpy(image)
		
		return torch.from_numpy(image), (self.loss.preprocess_label(text, image.shape[2]), text, image.shape[2])
```

## How to use a custom model

1. Create a file ```my_custom_model.py``` in the models directory.
2. This file must contains a class ```MyCustomModel``` inheriting from ```torch.nn.Module```
3. The ```forward``` function must return the output of the neural network, the ```get_input_image_height``` must return the input image height, and the ```create_loss``` must return an instance of the loss.

```python
import torch

from loss.ctc import CTC


class MyCustomModel(torch.nn.Module):

    def __init__(self, labels):
        super().__init__()

        self.labels = labels
        self.output_numbers =  max(labels.values()) + 1

        self.rnn = torch.nn.LSTM(self.convolutions_output_size[1] * self.convolutions_output_size[2], self.output_numbers, num_layers=2)

        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x):
        batch_size = x.data.size()[0]
        channel_num = x.data.size()[1]
        height = x.data.size()[2]
        width = x.data.size()[3]

        x = x.view(batch_size, channel_num * height, width)
        x = torch.transpose(x, 1, 2)
        
        x, _ = self.rnn(x)
        
        if not self.training:
            x = self.softmax(x)

        return x

    def get_input_image_height(self):
        return 32

    def create_loss(self):
        return CTC(self.labels, lambda x: x)
```


## How to use a custom loss

1. Create a file ```my_custom_loss.py``` in the loss directory
2. This file must contains a class ```MyCustomLoss``` inheriting from ```torch.nn.Module```
3. Multiple function must be implemented : 
    1. The ```forward``` function must return the loss
    2. ```preprocess_label``` is called in the dataset
    3. ```proces_label``` is called during the training
    4. ```ytrue_to_lines``` decode the output of the neural network
    
See ```loss/ctc.pyx```
		
## Generated document

Use the ```--generated``` argument to use Scribbler generated document.
Scribbler need to be cloned in the submodules folder.
 
[Scribbler](https://github.com/belosthomas/scribbler)