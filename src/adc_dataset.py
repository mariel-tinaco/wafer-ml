import torch.utils.data
import torchvision.transforms.functional as tvf
import os
import numpy as np

import math

from PIL import Image

####HELPER FUNCTIONS####

# each folder in the data directory is one of the classes.
# we need to convert them to class IDs.
def get_category_map():    
    categories = [
        'CMPMicroscratch',
        'CrystalDislocation',
        'DAP',
        'EnclosedDefect',
        'EtchBlock',
        'Fiber',
        'Flake',
        'MissingTrenchFill',
        'NonVisual',
        'Particle',
        'PolyNodules',
        'Residue'
    ]

    # for convenience, we'll build maps that translate between class ID and class name
    cat_to_id_map = {}
    id_to_cat_map = {}

    for idx, cat in enumerate(categories):
        cat_to_id_map[cat] = idx
        id_to_cat_map[idx] = cat

    return cat_to_id_map, id_to_cat_map

# The function below loads and image and applies some basic preprocessing.
# some of our data might be non-square images of different sizes, but we want
# every training sample to be a uniform size and shape.  These functions pad
# an image with zeros along its shorter dimension to make it a square.
def load_image(img_path):

    img = Image.open(img_path)

    #'L' indicates a single-channel, monochrome image.  We want to stick with
    # three-channel RGB for consistency, so make sure to convert.
    if img.mode == "L":
        img = img.convert("RGB")

    # we ensure that all of our images are a standard size, 224x224.
    # This is convenient because it's the standard resolution for the ImageNet
    # dataset, but we could also choose our own.
    img = pad_to_square(img)
    img = tvf.resize(img, 224)

    return img


def pad_to_size(img, size):
    width, height = img.size

    w_pad = size - width

    w_pad_l = math.ceil(w_pad / 2)
    w_pad_r = math.floor(w_pad / 2)

    h_pad = size - height

    h_pad_t = math.ceil(h_pad / 2)
    h_pad_b = math.floor(h_pad / 2)

    padding = (w_pad_l, h_pad_t, w_pad_r, h_pad_b)

    padded_img = tvf.pad(img, padding)

    return padded_img


def pad_to_square(img):

    width, height = img.size

    size = max(width, height)

    return pad_to_size(img, size)


####END HELPER FUNCTIONS####


class ADC_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, training=True):
        self.cat_to_id_map, _ = get_category_map()

        # we may want to perform different operations when dealing with training data and testing data
        # for example, if we add data augmentation, it should only be applied when training, and
        # we should test on clean samples.  In this example code, we don't do any augmentation,
        # but the flag is included for convenient extension.
        self.is_training = training

        # this will hold a list of all of of data samples.  Each sample is a tuple
        # which contains a path to the image, and the corresponding class ID.
        self.all_items = []

        # go through all of our categories, reading in data, and adding them to our list
        for cat, idx in self.cat_to_id_map.items():
            items = os.listdir(data_dir + "/" + cat)
            for item in items:
                datum = (data_dir + "/" + cat + "/" + item, idx)
                self.all_items.append(datum)

    # When we build our classifier, we'll need to know how many outputs to have,
    # so make a convenient way to access that information.
    def num_classes(self):
        return len(self.cat_to_id_map)

    # this is a required part of torch's Dataset interface.  It just says how many items we have.
    def __len__(self):
        return len(self.all_items)

    # this is the meat of the class.  When constructing training batches, torch will
    # ask the dataset for samples by index (where index is from 0 to __len__)
    # we return two items, the image and it's class ID.
    def __getitem__(self, index):

        # first, fetch the corresponding information from our list of items.
        img_path, label_idx = self.all_items[index]

        # now we have to read in the file.  After all, our network needs to operate
        # on pixel values, not file paths!
        img = load_image(img_path)

        # we could also perform any other manipulations here, such as data augmentation.
        # make sure to check the self.is_training flag, and only do augmentation on training data!

        # finally, we convert our image data to a torch tensor.
        torch_img = tvf.to_tensor(img)

        return torch_img, label_idx
