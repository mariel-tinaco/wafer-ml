import os
from adc_dataset import get_category_map, load_image
import torchvision
import torchvision.transforms.functional as tvf
import torch
from torch import nn
import csv
from models import test_model

if __name__ == "__main__":
    # this defines where we'll perform our training.  By default, we use the cpu.
    # if you have a GPU and cuda set up properly, you can instead set this to 'cuda'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the test images
    data_dir = "ADC_Dataset/test/"
    items = os.listdir(data_dir)
    items.sort()

    # Get the category map to retitreve the category name from predictions
    _, id_to_cat_map = get_category_map()

    # load the trained model
#######################################################################################
    model = torchvision.models.resnet18(num_classes=len(id_to_cat_map))
    model.fc = nn.Linear(512, len(id_to_cat_map))
    save_file = 'predictions-resnet18.csv'
#######################################################################################
    
#######################################################################################    
#     model = torchvision.models.mobilenet_v2(num_classes=len(id_to_cat_map))
#     model.classifier = nn.Linear(1280, len(id_to_cat_map))  
#     save_file = 'predictions-mobilenetv2.csv'
#######################################################################################

#######################################################################################
    #model = test_model.ResNet18(num_classes=len(id_to_cat_map))
#     save_file = 'predictions.csv'

#######################################################################################
    model.load_state_dict(torch.load("logs/resnet18-20221010-023016/checkpoint_19.pth"))
    model.to(device)
    # put our model in eval mode before validationing!
    model.eval()

    # Create an empty array to store the predictions
    predictions = []

    for img_name in items:
        img = tvf.to_tensor(load_image(data_dir + img_name))
        img = torch.unsqueeze(img, dim=0)
        img = img.to(device)
        pred = torch.argmax(model(img), dim=1)
        #print(pred[0].cpu().numpy())

        # Append a tuple of (img_id, prediction)
        img_id = int(img_name.split(".")[0])
        label = id_to_cat_map[pred.cpu().numpy()[0]]
        predictions.append(
            (img_id, label)
        )

    predictions.sort()
    # for a in predictions:
    #     print(a)

    with open(save_file, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(["Id", "Category"])
        write.writerows(predictions)

    