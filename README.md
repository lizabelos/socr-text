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
		
## Generated document

Use the ```--generated``` argument to use Scribbler generated document.
Scribbler need to be cloned in the submodules folder.
 
[Scribbler](https://github.com/dtidmarsh/scribbler)