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
    # this defines where we'll perform our training.  By default, we use the cpu.
    # if you have a GPU and cuda set up properly, you can instead set this to 'cuda'
    device = "cpu"

    # load the test images
    data_dir = "./ADC_Dataset/test/"
    items = os.listdir(data_dir)
    items.sort()

    # Get the category map to retitreve the category name from predictions
    _, id_to_cat_map = get_category_map()

    # load the trained model
    model = torchvision.models.resnet18(num_classes=len(id_to_cat_map))
    model.load_state_dict(torch.load("./resnet18_17.pth"))
    model.to(device)
    model.eval()

    # load the NPZ file for reference
    data_path = os.getcwd()+'/DATASET_NPZ/preproc_train_test_224.npz'
    map = np.load(data_path)

    # Create an empty array to store the predictions
    img_id = []
    true_label = []
    pred_label = []

    for img_name in items:
        img = tvf.to_tensor(load_image(data_dir + img_name))
        img = torch.unsqueeze(img, dim=0)
        pred = torch.argmax(model(img), dim=1)

        # Get Predicted Label
        img_id.append(int(img_name.split(".")[0]))
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

