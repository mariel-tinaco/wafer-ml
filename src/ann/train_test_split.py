
import glob
from sre_parse import CATEGORIES
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

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


for f in folders:
    paths[f] = glob.glob(root_dir+f+'/'+'*.jpg', recursive=True)
    names[f]  = [(os.path.split(i)[1]).split('.')[0] for i in paths[f]]


x_data = []
y_data = []
# Split Data for Test and Train
for key in names.keys():
    for fname in paths[key]:
        x_data.append(fname)
        y_data.append(class_names_label[key])



# create train and test sets
x_train_path, x_test_path, y_train_path, y_test_path = train_test_split(x_data, y_data,
                                                    stratify=y_data, 
                                                    test_size=0.25,
                                                    random_state=seed)

# Save train and Test dataset to NPZ
'''
EfficientNetB0	224
EfficientNetB1	240
EfficientNetB2	260
EfficientNetB3	300
EfficientNetB4	380
EfficientNetB5	456
EfficientNetB6	528
EfficientNetB7	600
'''
IMG_SIZE = (240,240) 

x_train = []
y_train = []
for i,paths in enumerate(x_train_path):
    # Reformat the Image (Color, Size)
    img = cv2.imread(paths)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img,IMG_SIZE)

    # Append to Images and Label Array
    x_train.append(img)
    y_train.append(y_train_path[i])


x_test = []
y_test = []
for i,paths in enumerate(x_test_path):
    # Reformat the Image (Color, Size)
    img = cv2.imread(paths)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img,IMG_SIZE)

    # Append to Images and Label Array
    x_test.append(img)
    y_test.append(y_train_path[i])


# save the data with numpy so we can use it later
datafile = os.getcwd()+'/DATASET_NPZ/train_test_'+str(IMG_SIZE[0])+'.npz'
np.savez(datafile, 
        x_train = x_train, y_train = y_train,
        x_test=x_test, y_test=y_test,)

print('NPZ saved')




