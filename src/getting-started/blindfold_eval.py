import os
from adc_dataset import get_category_map, load_image
import torchvision
import torchvision.transforms.functional as tvf
import torch
import csv
from torchmetrics.functional.classification import multiclass_confusion_matrix
import numpy as np
import glob

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
        for fname in names[key]:
            img = tvf.to_tensor(load_image(paths[key]))
            img = torch.unsqueeze(img, dim=0)
            pred = torch.argmax(model(img), dim=1)

            # Get Predicted Label
            img_id.append(fname)
            pred_label.append(id_to_cat_map[pred.detach().numpy()[0]])

            # Get True Label
            for index,id in enumerate(map['id']):
                if img_id == id:
                    true_label.append(id_to_cat_map[map['label'][index]])

        n_classes = len(np.unique(map['label']))
        target = torch.tensor([2, 1, 0, 0])
        preds = torch.tensor([2, 1, 0, 1])
        multiclass_confusion_matrix(preds, target, num_classes=3)

        with open('blindfoldtest.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(['Id', 'Predicted','True'])
            for index in range(0,len(img_id)):
                write.writerow([str(img_id[index]), 
                                str(pred_label[index]),
                                str(true_label[index])])

