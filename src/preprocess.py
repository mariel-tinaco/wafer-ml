'''

Create a Pre-Processed Dataset

'''
import glob
import cv2
import os
import numpy as np
import imutils
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance,ImageFilter

def rotates(img,tail):
    # Rotate Images
    r90 = imutils.rotate(img, angle=90)
    r180 = imutils.rotate(img, angle=180)
    r270 = imutils.rotate(img, angle=270)
    
    cv2.imwrite(names[key][index]+tail+'_90.jpg', r90)
    cv2.imwrite(names[key][index]+tail+'_180.jpg', r180)
    cv2.imwrite(names[key][index]+tail+'_270.jpg', r270)


def flips(img,tail):
    # Flip Images
    rx = cv2.flip(img,0)
    ry = cv2.flip(img,1)

    cv2.imwrite(names[key][index]+tail+'_rx.jpg', rx)
    cv2.imwrite(names[key][index]+tail+'_ry.jpg', ry)


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

ftarget = ('/ADC_Dataset/noised/')

# Apply all processes and augment
# augmentation
# denoising
# Edge Enhance


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
        # Load Raw Image
        append_str = '_r'
        print(key,': ',np.round(index/len(paths[key])*100,2),' %  ')
        img = cv2.imread(fpath)
        RAW_IMG = img
        cv2.imwrite(names[key][index]+append_str+'.jpg', img)

        # Augment then Save
        rotates(img,append_str)
        flips(img,append_str)

        '''# Denoise
        append_str = '_d'
        den = cv2.fastNlMeansDenoising(np.asarray(img),None,7,11,81)
        cv2.imwrite(names[key][index]+append_str+'.jpg', den)

        # Augment then Save
        rotates(den,append_str)
        flips(den,append_str)
        '''


        '''# Add Gaussian Noise
        append_str = '_n'
        row,col,ch= img.shape
        mean = 0.5
        var = 0.5
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        img = img+gauss
        cv2.imwrite(names[key][index]+append_str+'.jpg', img)
    
        # Augment then Save
        rotates(img,append_str)
        flips(img,append_str)
        '''

        # Edge Enhanced
        append_str = '_e'
        img = Image.fromarray(RAW_IMG)
        img = img.filter(ImageFilter.EDGE_ENHANCE)
        img = np.asarray(img, dtype='uint8')
        cv2.imwrite(names[key][index]+append_str+'.jpg', img)

        # Augment then Save
        rotates(img,append_str)
        flips(img,append_str)

        # Blurred
        append_str = '_b'
        img = Image.fromarray(RAW_IMG)
        img = img.filter(ImageFilter.BLUR)
        img = np.asarray(img, dtype='uint8')
        cv2.imwrite(names[key][index]+append_str+'.jpg', img)

        # Augment then Save
        rotates(img,append_str)
        flips(img,append_str)

        

    
        


        

