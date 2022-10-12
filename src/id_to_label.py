
import glob
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
fpath = ('/ADC_Dataset/train/')
root_dir =  froot + fpath

# Get the Filenames and Filepaths of the Figures
folders = os.listdir(root_dir)

paths = {}
names = {}


for f in folders:
    paths[f] = glob.glob(root_dir+f+'/'+'*.jpg', recursive=True)
    names[f]  = [(os.path.split(i)[1]).split('.')[0] for i in paths[f]]


id_data = []
label_data = []

# Assign Label to every ID
for key in names.keys():
    for fname in names[key]:
        id_data.append(fname)
        label_data.append(class_names_label[key])

# save the data with numpy so we can use it later
datafile = os.getcwd()+'/DATASET_NPZ/id_to_label_map.npz'
np.savez(datafile, 
        id = id_data, label = label_data)
print('NPZ saved')




