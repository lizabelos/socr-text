import math

import numpy as np
import glymur
import matplotlib
from PIL import Image

matplotlib.use('tkagg')
from matplotlib import pyplot
from skimage import filters
import cv2

def image_pillow_to_numpy(image):
    image = np.array(image, dtype='float') / 255.0
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    return image


def image_numpy_to_pillow(image):
    image = np.swapaxes(image, 1, 2)
    image = np.swapaxes(image, 0, 2)
    image = image * 255.0
    return Image.fromarray(image.astype('uint8'), 'RGB')

def image_numpy_to_pillow_bw(image):
    image = image * 255.0
    return Image.fromarray(image.astype('uint8'), 'L')


def image_pytorch_to_pillow(image):
    return image_numpy_to_pillow(image.cpu().detach().numpy())


def load_jp2_numpy(image_path):
    image = glymur.Jp2k(image_path)
    image = np.array(image[:], dtype='float') / 255.0
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    return image


def show_numpy_image(image, invert_axes=True, pause=3):
    if invert_axes:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 0, 2)
    print("Showing image using tkagg..." + str(image.shape))
    pyplot.imshow(image)
    pyplot.pause(pause)


def show_pytorch_image(image, invert_axes=True, pause=3):
    return show_numpy_image(image.cpu().detach().numpy(), invert_axes, pause)




