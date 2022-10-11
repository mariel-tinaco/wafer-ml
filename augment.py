import pandas as pd
import os
# import shutil
from PIL import Image, ImageOps
import torch
from torchvision import transforms

from copy import copy

data_dir = "ADC_Dataset"

new_dir = f"{data_dir}_Augmented"

torch.manual_seed(0)
aug = 0
train_cnt = 0
test_cnt = 0

sd = (0.2, 5)

def aug_flip(img, img_name, state, dst_dir, img_size = (128,128), train = False):
    prefix = "f"

    for i in range(2):
        aug_img_name = "{}_{}{}".format(img_name, prefix, i)

        if i == 1:
            img = ImageOps.mirror(img)

        #rotate resulting images
        aug_rotate(img, aug_img_name, state, dst_dir=dst_dir, img_size=img_size, train = train)

def aug_rotate(img, img_name, state, dst_dir, img_size = (128,128), train = False):
    prefix = "r"
    
    for i in range(4):
        aug_img_name = "{}_{}{}".format(img_name, prefix, i)
        img = img.rotate(90)

        #blur, rotate and/or add jitter to resulting images
        augment(img, aug_img_name, state, dst_dir=dst_dir, img_size=img_size, train = train)



def augment(img, img_name, state, dst_dir, img_size = (128,128), train = False):
    global train_cnt
    global test_cnt
    file_ext = 'jpg'
    if train:    
        img_dst = os.path.join(dst_dir, 'train', state, img_name)
        train_cnt += 1
    else:
        img_dst = os.path.join(dst_dir, 'test', state, img_name)
        test_cnt += 1

    res_img = img.resize(img_size)
    res_img.save("{}.{}".format(img_dst, file_ext))

    if train and aug>0:
        # crop center & blur only
        aug_img_file = "{}_{}{}.{}".format(img_dst, 'b', 0, file_ext)
        aug_img = augment_blur(img)
        aug_img = aug_img.resize(img_size)
        aug_img.save(aug_img_file)
        train_cnt += 1

        # random jitter, affine, brightness & blur
        for i in range(aug):
            aug_img_file = "{}_{}{}.{}".format(img_dst, 'j', i, file_ext)
            aug_img = augment_affine_jitter_blur(img)
            aug_img = aug_img.resize(img_size)
            aug_img.save(aug_img_file)
            train_cnt += 1


def augment_affine_jitter_blur(orig_img):
    """
    Augment with multiple transformations
    """
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.05, 0.05)),
        #transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
        transforms.CenterCrop((220, 220)),
        transforms.ColorJitter(brightness=(0.5)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=sd),
        ])
    return train_transform(orig_img)


def augment_blur(orig_img):
    """
    Augment with center crop and bluring
    """
    train_transform = transforms.Compose([
        #transforms.CenterCrop((240, 240)), #220,220
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=sd) #sigma 5
        ])
    return train_transform(orig_img)


def create_set(aug = 0, img_size = (128,128)):
    src_path = os.path.join(data_dir, "train")
    dst_path = os.path.join(new_dir, "train")
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir, exist_ok=True)
        os.makedirs(dst_path, exist_ok=True)

    for state in os.listdir(src_path):
        train_img_dir = os.path.join(src_path, state)
        dst_img_dir = os.path.join(dst_path, state)
        os.makedirs(dst_img_dir, exist_ok=True)

        for file in os.listdir(train_img_dir):
            img_src = os.path.join(train_img_dir,file)
            img_name = file.rsplit('.', 1)[0]

            img = Image.open(img_src)

            aug_flip(img, img_name, state, dst_dir=new_dir, img_size=img_size, train = True)
            # aug_rotate(img, img_name, state, dst_dir=new_dir, img_size=img_size, train = True) #aug_flip is bypassed


    print(f'Augmented dataset: {test_cnt} test, {train_cnt} train samples')



create_set(img_size=(480,480))
