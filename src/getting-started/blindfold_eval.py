import cmath
import os
from adc_dataset import get_category_map, load_image
import torchvision
import torchvision.transforms.functional as tvf
import torch
import csv
from torchmetrics.functional.classification import multiclass_confusion_matrix
import numpy as np
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    device = "cpu"

    
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


    # Get the category map to retitreve the category name from predictions
    _, id_to_cat_map = get_category_map()

    # load the trained model
    model = torchvision.models.resnet18(num_classes=len(id_to_cat_map))
    model.load_state_dict(torch.load("./resnet18_17.pth"))
    model.to(device)
    model.eval()

    # load the NPZ file for reference
    data_path = os.getcwd()+'/DATASET_NPZ/id_to_label_map.npz'
    map = np.load(data_path)

    # Create an empty array to store the predictions
    img_id = []
    true_label = []
    pred_label = []
    
    for key in names.keys():
        for index,fname in enumerate(names[key]):
            img = tvf.to_tensor(load_image(paths[key][index]))
            img = torch.unsqueeze(img, dim=0)
            pred = torch.argmax(model(img), dim=1)

            # Get Predicted Label
            img_id.append(fname)
            pred_label.append(id_to_cat_map[pred.detach().numpy()[0]])

            # Get True Label
            for index,id in enumerate(map['id']):
                if fname == id:
                    true_label.append(id_to_cat_map[map['label'][index]])

        n_classes = len(np.unique(map['label']))
        target = torch.tensor(true_label)
        preds = torch.tensor(pred_label)
        cm = multiclass_confusion_matrix(preds, target, num_classes=n_classes)

        cm_df = pd.DataFrame(cm,
                     index = class_names, 
                     columns = class_names)

        #Plotting the confusion matrix
        plt.figure(figsize=(10,10))
        sns.heatmap(cm_df, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.savefig('Confusion_Matrix.png')

        with open('blindfoldtest.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(['Id', 'Predicted','True'])
            for index in range(0,len(img_id)):
                write.writerow([str(img_id[index]), 
                                str(pred_label[index]),
                                str(true_label[index])])
        

