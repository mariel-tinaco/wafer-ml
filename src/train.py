from adc_dataset import ADC_Dataset
import torchvision
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime


if __name__ == "__main__":
    # if you have a GPU and cuda set up properly, you can instead set this to 'cuda'
    device = 'cuda'

    # build our training data and validation data sets
    dataset = ADC_Dataset("./ADC_Dataset/noised", training=True)
    n_images = len(dataset)
    n_val = int(n_images*.25)
    n_train = n_images - n_val
    num_classes = dataset.num_classes()

    print("Loaded ", n_images, " images")
    print("Training split: ", n_train)
    print("Validation split: ", n_val)
    
    # Set Model Parameters
    N_EPOCHS = 60
    N_BATCH = 16

    model = torchvision.models.resnet152(num_classes=num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    loss_fn = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter("./logs/n_resnet152")

    # Create Training and Valdiation Dataset 
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    # Create an Iterable Dataset for Training and Validation
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=N_BATCH, num_workers=16, shuffle=True
    )
    validation_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=N_BATCH, num_workers=16, shuffle=True
    )
    
    # Start Training
    time_start = datetime.now()
    
    print("Time Start: ", time_start)
    for epoch in range(N_EPOCHS):
        step_counter = 0
        print("EPOCH", epoch) 
        trainiter = iter(train_dataloader)

        model.train()
        # loop over batches from our training data
        for batch_images, batch_labels in trainiter:
            optimizer.zero_grad()

            # Load Data Batch to Device
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            
            # Compute for Training Accuracy
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.mean((predictions == batch_labels).float())

            # Compute for Training Loss
            batch_loss = loss_fn(outputs, batch_labels)

            # Calculate gradients from backward propagation
            batch_loss.backward()

            # Apply the new gradients
            optimizer.step()
            progress = np.round(step_counter/(n_images/N_BATCH)*100,2)
            if step_counter%10 == 0:
                print(
                    progress,'%         ',
                    '   Epoch:',epoch,
                    '   Acc: ',accuracy.cpu().detach().numpy(),
                    '   Loss: ',batch_loss.cpu().detach().numpy(),
                )
            step_counter += 1

        # Check Performance on Validation Data
        print("RUNNING VAL")
        validationiter = iter(validation_dataloader)
        model.eval()

        val_acc = 0
        for validation_images, validation_labels in validationiter:
            
            # Load Data Batch to Device
            validation_images = validation_images.to(device)
            validation_labels = validation_labels.to(device)
            
            outputs = model(validation_images)

            validation_predictions = torch.argmax(outputs, dim=1)

            # Compute for Validation Accuracy
            correct = torch.sum((validation_predictions == validation_labels).float())
            val_acc += correct/len(val_set)

            # Compute for Validation Loss
            val_loss = loss_fn(outputs, validation_labels)

            if step_counter%50 == 0:
                print(
                    step_counter,
                    epoch,
                    val_acc.cpu().detach().numpy(),
                    val_loss.cpu().detach().numpy(),
                )


        # Write Scalars to Tensorboard
        writer.add_scalars('Accuracy', {"Train":accuracy,
                                "Validation": val_acc}, epoch)

        writer.add_scalars('Loss', {"Train":batch_loss,
                                "Validation": val_loss}, epoch)
        print(
            'Train Accuracy',
            accuracy.cpu().detach().numpy(),
            'Train Loss',
            batch_loss.cpu().detach().numpy(),
        )
        print(
            "Valdation Accuracy",
            val_acc.cpu().detach().numpy(),
            "Valdation Loss",
            val_loss.cpu().detach().numpy(),
        )
        
        # Save Model Checkpoints
        torch.save(model.state_dict(), "n_resnet152_{:02d}.pth".format(epoch))
    # Save Final Model 
    torch.save(model.state_dict(), "n_resnet152_final.pth")
    time_end = datetime.now()
    print("Time End: ", time_end)
    print("Elapsed Time: ", time_start-time_end)
