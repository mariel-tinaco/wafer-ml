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
import itertools

def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Confusion matrix, with normalization')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # create a plot for the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)    
    
    # if we want a title displayed
    if title:        
        plt.title(title)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = ['CMPMicroscratch',
                'CrystalDislocation', 
                'DnakakaAP', 
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
    model_name = 'n_resnet152_21'
    model = torchvision.models.resnet152(num_classes=len(id_to_cat_map))
    model.load_state_dict(torch.load('./'+model_name+'.pth'))
    model.to(device)  
    model.eval()

    # load the NPZ file for reference
    data_path = os.getcwd()+'/DATASET_NPZ/id_to_label_map.npz'
    map = np.load(data_path)

    # Create an empty array to store the predictions
    img_id = []
    true_label = []
    pred_label = []
    counter = 0
    bf_acc = 0
    for key in names.keys():
        for index,fname in enumerate(names[key]):
            counter += 1
            img = tvf.to_tensor(load_image(paths[key][index]))
            img = torch.unsqueeze(img, dim=0)
            pred = torch.argmax(model(img), dim=1)

            # Get Predicted Label
            img_id.append(fname)
            #pred_label.append(id_to_cat_map[pred.detach().numpy()[0]])
            pred_label.append(pred.detach().numpy()[0])

            # Get True Label
            for index,id in enumerate(map['id']):
                if fname == id:
                    print(counter, fname)
                    true_label.append(map['label'][index])
                    if pred.detach().numpy()[0] == map['label'][index]:
                        bf_acc += 1

    bf_acc = np.round(bf_acc/len(pred_label)*100,3)
    n_classes = len(np.unique(map['label']))
    target = torch.tensor(np.asarray(true_label))
    preds = torch.tensor(np.asarray(pred_label))
    cm = multiclass_confusion_matrix(preds, target, num_classes=n_classes)

    plt.figure(figsize=[15,15])
    plot_confusion_matrix(cm, 
                        classes=class_names,
                        title='Wafer Defect Detection')
    plt.savefig('blind_confusion_plot_'+model_name+'.png')
    
    # Saving Data
    with open('blindfoldtest_'+model_name+'.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(['Accuracy', bf_acc])
        write.writerow(['Id', 'Predicted','True',])
        for index in range(0,len(img_id)):
            write.writerow([str(img_id[index]), 
                            str(pred_label[index]),
                            str(true_label[index])])

    

