'''

Create a Pre-Processed Dataset

'''
import glob
import cv2
import os
import numpy as np
import imutils
import matplotlib.pyplot as plt

# fix random seed for reproduciblity
seed = 1169
np.random.seed(seed)

class_names = ['CMPMicroscratch',
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
                'Residue']
class_names_label = {class_name:i for i,class_name in enumerate(class_names)}
num_classes = len(class_names)


# set the directory where the data lives
froot = os.getcwd()
fpath = ('./ADC_Dataset/train/')
root_dir =  froot + fpath

# Get the Filenames and Filepaths of the Figures
folders = os.listdir(root_dir)

paths = {}
names = {}

# directory where data will be saved

ftarget = ('/ADC_Dataset/preproccesed/')

# Go trough all images and apply
# augmentation
# denoising
# denoising + augmentation


# Load Data Paths and Filenames
paths = {}
names = {}
for f in folders:
    paths[f] = glob.glob(root_dir+f+'/'+'*.jpg', recursive=True)
    names[f]  = [(os.path.split(i)[1]).split('.')[0] for i in paths[f]]

# Preprocess Data
for key in paths.keys():
    target_dir = froot+ftarget + key + '/'
    if os.path.exists(target_dir) == False:
        print('Creating Folder:   ',target_dir)
        os.makedirs(target_dir)
    os.chdir(target_dir)

    for index,fpath in enumerate(paths[key]):
        print(key,': ',np.round(index/len(paths[key])*100),2)
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        cv2.imwrite(names[key][index]+'r.jpg', img)
        
        # Rotate Images
        r90 = imutils.rotate(img, angle=90)
        cv2.imwrite(names[key][index]+'r90.jpg', r90)
        r180 = imutils.rotate(img, angle=180)
        cv2.imwrite(names[key][index]+'r180.jpg', r180)
        r270 = imutils.rotate(img, angle=270)
        cv2.imwrite(names[key][index]+'r270.jpg', r270)

        # Denoise
        den = cv2.fastNlMeansDenoising(np.asarray(img),None,7,11,81)
        cv2.imwrite(names[key][index]+'d.jpg', den)

        # Rotate Denosied Images
        d90 = imutils.rotate(den, angle=90)
        cv2.imwrite(names[key][index]+'d90.jpg', d90)
        d180 = imutils.rotate(den, angle=180)
        cv2.imwrite(names[key][index]+'d180.jpg', d180)
        d270 = imutils.rotate(den, angle=270)
        cv2.imwrite(names[key][index]+'d270.jpg', d270)

        

