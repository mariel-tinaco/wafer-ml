# from adc_dataset import ADC_Dataset
import torchvision
from torch import nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import math
from datetime import datetime
from models import test_model
#from sampler import StratifiedBatchSampler
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os

torch.manual_seed(0)
date_time = datetime.today().strftime('%Y%m%d-%H%M%S')
use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    # this defines where we'll perform our training.  By default, we use the cpu.
    # if you have a GPU and cuda set up properly, you can instead set this to 'cuda'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    save_folder = 'logs'
    
#     data_set = 'ADC_Dataset'
#     data_set = 'ADC_Dataset_Augmented'
#     data_set = "data/ADC_Dataset_Split"
    data_set = "data/ADC_Dataset_Split_Augmented"
    
    train_dir = data_set + '/train'

    # build our training data and validation data sets
    # dataset = ADC_Dataset(f"{data_set}/train", training=True)
    train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    ])
    dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    targets = dataset.targets
    n_images = len(dataset)
    classes = dataset.classes
    num_classes = len(dataset.classes)

    batch_size = 32
    num_epochs = 20
    lr = 0.0002
    wd = 0

    best_epoch = 0
    best_epoch_acc = 0
    best_epoch_loss = 0
    ct = 0 
    
    train_idx, valid_idx= train_test_split(np.arange(len(targets)), test_size=0.1, shuffle=True, stratify=targets)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler)
    validation_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4, sampler=valid_sampler)

    n_train = len(train_sampler)
    n_val = len(valid_sampler)

    
    print("Loaded ", n_images, " images")
    print("Training split: ", n_train)
    print("Validation split: ", n_val)
#######################################################################################

#     model_name = 'mobilenetv2'
#     model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
#     model.classifier = nn.Linear(1280, num_classes)  
#     for child in model.children():
#         for grandchild in child.children():
#             ct+=1
#             if ct < 10:
#                 for param in grandchild.parameters():
#                     param.requires_grad = False
#         break


#######################################################################################   

#######################################################################################

#     model_name = 'resnet18'
#     model = torchvision.models.resnet18(pretrained=True)
#     last_layer = 512
    
#     model_name = 'resnet34'
#     model = torchvision.models.resnet34(pretrained=True)
#     last_layer = 512
    
#     model_name = 'resnet50'
#     model = torchvision.models.resnet50(pretrained=True)
#     last_layer = 2048
    
    model_name = 'resnet152'
    model = torchvision.models.resnet152(pretrained=True)
    last_layer = 2048
    
    model.fc = nn.Linear(last_layer, num_classes)  

    for child in model.children():
        ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False   
#     print(model.parameters)
#     print(ct)

#######################################################################################

#     model = test_model.ResNet34(num_classes=num_classes)
 
#######################################################################################

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.CrossEntropyLoss()

    log_dir = "./{}/{}/{}".format(save_folder, model_name, date_time)
    writer_train = SummaryWriter(f"{log_dir}/training")
    writer_val = SummaryWriter(f"{log_dir}/validation")
    steps_per_epoch = math.ceil(n_train/batch_size)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(f'{log_dir}/Training.txt', 'w') as f:
        f.write(f'Model Architecture: {model_name}\n')
        f.write(f'Batch size: {batch_size}\n')
        f.write(f'Number of epochs: {num_epochs}\n')
        f.write('Optimizer: Adam\n')
        f.write(f'Initial Learning Rate: {lr}\n')
        f.write(f'Weight Decay: {wd}\n')
        f.write(f'Dataset: {data_set}\n\n')
        f.write('#####Training Starts#####\n')
        f.close()

    for epoch in range(num_epochs):
        running_loss = 0
        correct_preds = 0
    
        trainiter = iter(train_dataloader)

        model.train()

        for batch_images, batch_labels in trainiter:
            optimizer.zero_grad()
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_images)

            batch_loss = loss_fn(outputs, batch_labels)
            predictions = torch.argmax(outputs, dim=1)
            correct_preds += torch.sum((predictions == batch_labels).float())

            batch_loss.backward()
            optimizer.step()

            running_loss += outputs.shape[0]*batch_loss.item()

        accuracy = correct_preds*100/n_train
        epoch_loss = running_loss/n_train

        writer_train.add_scalar("Loss", epoch_loss, epoch)
        writer_train.add_scalar("Accuracy", accuracy, epoch)

        print("Epoch: {} \nTraining Loss: {:.4f} Training Accuracy: {:.2f}%".format(epoch, epoch_loss, accuracy))
        with open(f'{log_dir}/Training.txt', 'a+') as f:
            f.write("Epoch: {} \nTraining Loss: {:.4f} Training Accuracy: {:.2f}%\n".format(epoch, epoch_loss, accuracy))
            f.close()

#         print("RUNNING VAL")
        validationiter = iter(validation_dataloader)
        model.eval()

        total_validation_correct = 0
        wrong_counter = 0
        running_loss = 0
        cm = np.zeros((num_classes,num_classes), dtype=int)
        for validation_images, validation_labels in validationiter:

            validation_images = validation_images.to(device)
            validation_labels = validation_labels.to(device)

            outputs = model(validation_images)
            validation_predictions = torch.argmax(outputs, dim=1)
            
            v_loss = loss_fn(outputs, validation_labels)
            running_loss += outputs.shape[0]*v_loss.item()

            correct = torch.sum((validation_predictions == validation_labels).float())
            total_validation_correct += correct
            for i in range(len(validation_labels.cpu().numpy())):
                cm[validation_labels.cpu().numpy()[i]][validation_predictions.cpu().numpy()[i]] +=1
            
        validation_loss = running_loss / n_val
        validation_accuracy = total_validation_correct*100 / n_val
        writer_val.add_scalar("Loss", validation_loss, epoch)
        writer_val.add_scalar("Accuracy", validation_accuracy, epoch)
        print("Valdation Loss: {:.2f}  Validation Accuracy: {:.2f}%".format(validation_loss, validation_accuracy))
        if validation_accuracy >= best_epoch_acc:
            best_epoch = epoch
            best_epoch_acc = validation_accuracy
            best_epoch_loss = validation_loss
            torch.save(model.state_dict(), "{}/checkpoint_{:02d}.pth".format(log_dir, epoch))
            torch.save(model.state_dict(), f"{log_dir}/best.pth")
        print("BEST EPOCH: {}  Valdation Loss: {:.2f}  Validation Accuracy: {:.2f}%".format(best_epoch, best_epoch_loss, best_epoch_acc))
        print(cm)

        
        with open(f'{log_dir}/Training.txt', 'a+') as f:
            f.write("Valdation Loss: {:.2f}  Validation Accuracy: {:.2f}%\n".format(validation_loss, validation_accuracy))
            f.write("BEST EPOCH: {}  Valdation Loss: {:.2f}  Validation Accuracy: {:.2f}%\n".format(best_epoch, best_epoch_loss, best_epoch_acc))
            f.write(f"Confusion Matrix:\n{str(cm)}\n\n")
            f.close()

        

    torch.save(model.state_dict(), f"{log_dir}/checkpoint__final.pth")
