# from adc_dataset import ADC_Dataset
import torchvision
from torch import nn
import torch.utils.data
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# from models import test_model
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os

torch.manual_seed(0)
date_time = datetime.today().strftime('%Y%m%d-%H%M%S')
#use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    # this defines where we'll perform our training.  By default, we use the cpu.
    # if you have a GPU and cuda set up properly, you can instead set this to 'cuda'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    save_folder = 'logs'
    
#     data_set = 'ADC_Dataset'
#     data_set = 'ADC_Dataset_Augmented'
    data_set = "data/ADC_Dataset_Split"
    
    test_dir = data_set + '/test'

    # build our training data and test data sets
    # dataset = ADC_Dataset(f"{data_set}/train", training=True)
    test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    ])
    dataset = torchvision.datasets.ImageFolder(test_dir, transform=test_transforms)
    targets = dataset.targets
    n_images = len(dataset)
    classes = dataset.classes
    num_classes = len(dataset.classes)

    batch_size = 32

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
    n_val = len(dataset)

    
    print("Loaded ", n_images, " images")
    print("Testing: ", n_val, " images")
#######################################################################################

#     model_name = 'mobilenetv2'
#     model = torchvision.models.mobilenet.mobilenet_v2(num_classes=num_classes)
#     model.classifier = nn.Linear(1280, num_classes)  

#######################################################################################   

#######################################################################################

#     model_name = 'resnet18'
#     model = torchvision.models.resnet18(num_classes=num_classes)
    
#     model_name = 'resnet34'
#     model = torchvision.models.resnet34(num_classes=num_classes)
    
#     model_name = 'resnet50'
#     model = torchvision.models.resnet50(num_classes=num_classes)
    
    model_name = 'resnet152'
    model = torchvision.models.resnet152(num_classes=num_classes)
    last_layer = 2048

    vers_n = "20221012-234106"
    log_dir = f"logs/{model_name}/{vers_n}"
    model.load_state_dict(torch.load(f"{log_dir}/best.pth"))
#     print(model.parameters)
#     print(ct)

#######################################################################################

#     model = test_model.ResNet34(num_classes=num_classes)
 
#######################################################################################

    model.to(device)
    
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(f'{log_dir}/Test.txt', 'w') as f:
        f.write(f'Model Architecture: {model_name}\n')
        f.write(f'Version : {vers_n}\n')
        f.write('##### Test #####\n')
        f.close()


    testiter = iter(test_dataloader)
    model.eval()

    total_test_correct = 0
    wrong_counter = 0
    running_loss = 0
    cm = np.zeros((num_classes,num_classes), dtype=int)
    
    for test_images, test_labels in testiter:

        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        outputs = model(test_images)
        test_predictions = torch.argmax(outputs, dim=1)
        
        # v_loss = loss_fn(outputs, test_labels)
        # running_loss += outputs.shape[0]*v_loss.item()

        correct = torch.sum((test_predictions == test_labels).float())
        total_test_correct += correct
        for i in range(len(test_labels.cpu().numpy())):
            cm[test_labels.cpu().numpy()[i]][test_predictions.cpu().numpy()[i]] +=1
        
    # test_loss = running_loss / n_val
    test_accuracy = total_test_correct*100 / n_val
    # writer_val.add_scalar("Loss", test_loss, epoch)
    # writer_val.add_scalar("Accuracy", test_accuracy, epoch)
    print("Test Accuracy: {:.2f}%".format(test_accuracy))

    with open(f'{log_dir}/Test.txt', 'a+') as f:
        f.write("Test Accuracy: {:.2f}%\n".format(test_accuracy))
        f.write(f"Confusion Matrix:\n{cm}")
        f.close()

