from adc_dataset import ADC_Dataset
import torchvision
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # this defines where we'll perform our training.  By default, we use the cpu.
    # if you have a GPU and cuda set up properly, you can instead set this to 'cuda'
    device = 'cuda'

    # build our training data and validation data sets
    dataset = ADC_Dataset("./ADC_Dataset/preproccesed", training=True)
    n_images = len(dataset)
    n_val = int(n_images*.25)
    n_train = n_images - n_val
    num_classes = dataset.num_classes()

    print("Loaded ", n_images, " images")
    print("Training split: ", n_train)
    print("Validation split: ", n_val)

    # Apply a simple partition to get training and validation sets.
    # This is just an example. There are much better ways to do this!
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=32, num_workers=4, shuffle=True
    )
    validation_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=32, num_workers=4, shuffle=True
    )

    # initialize our model, loss function, and optimizer
    # torchvision provides a variety of pre-built architectures, let's try one of those.
    model = torchvision.models.resnet34(num_classes=num_classes)
    model.to(device)

    # we use the Adam optimizer, which is a little fancier than standard SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # cross-entropy is the standard loss formulation for classification tasks.
    loss_fn = torch.nn.CrossEntropyLoss()

    # create a logger so that we can track our progress on tensorboard
    writer = SummaryWriter("./logs/resnet34")

    step_counter = 0

    for epoch in range(50):
        print("EPOCH", epoch)
        # initialize a fresh iterator for each epoch.  Each epoch is one pass over
        # the entire training dataset.
        trainiter = iter(train_dataloader)
        # make sure we're in training mode before we do any training!
        model.train()
        # loop over batches from our training data
        for batch_images, batch_labels in trainiter:

            # clear out old gradients, prepare to take a new training step
            optimizer.zero_grad()

            # move our training data over to the proper device.  If we're using the cpu,
            # this is a no-op, but is necessary if we're using cuda.
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # run the training images through our model.
            # outputs will have shape [batch_size,num_classes]
            # and each entry will be a value saying how likely that class is for that input
            # note that these values are unnormalized, they are not probabilities.
            outputs = model(batch_images)

            # compute the loss for these outputs given the true labels
            batch_loss = loss_fn(outputs, batch_labels)

            # log our loss to tensorboard
            writer.add_scalar("Training Loss", batch_loss, step_counter)

            # tracking loss is helpful, but it is also nice to know how many we're getting correct.
            # the predicted class for an input is the one with the largest score, so we take
            # the argmax over the class dimension.
            # predictions will be of shape [batch_size], and each entry will be the class ID
            # that is predicted for that batch item
            predictions = torch.argmax(outputs, dim=1)

            # to compute accuracy, we check whether each prediction matches the ground truth.
            # this trick works because torch treats True as 1 and False as 0 when converting to floats.
            accuracy = torch.mean((predictions == batch_labels).float())

            # log our accuracy for this batch.  This value will fluctuate, as it only counts
            # the current batch.  It is not a long-term average.
            writer.add_scalar("Training Accuracy", accuracy, step_counter)

            # this is where the actual learning happens.
            # backward() will compute gradients for our model from the loss.
            batch_loss.backward()
            # now the optimizer takes a step in the direction indicated by the gradients we just computed.
            optimizer.step()

            print(
                step_counter,
                epoch,
                batch_loss.cpu().detach().numpy(),
                accuracy.cpu().detach().numpy(),
            )
            step_counter += 1

        # at the end of each epoch, we want to validation our model's performance on the validation set
        # here we will track the total accuracy across the entire set, instead of on a per-batch basis

        # while tracking training accuracy is helpful to visualize our progress, it is susceptible
        # to overfitting and memorization.  The true validation is how the network peforms on data it hasn't
        # been trained on, so it's important to check our validation performance regularly.
        print("RUNNING VAL")
        validationiter = iter(validation_dataloader)
        # put our model in eval mode before validationing!
        model.eval()

        total_validation_correct = 0
        wrong_counter = 0
        for validation_images, validation_labels in validationiter:

            validation_images = validation_images.to(device)
            validation_labels = validation_labels.to(device)

            outputs = model(validation_images)
            validation_predictions = torch.argmax(outputs, dim=1)

            correct = torch.sum((validation_predictions == validation_labels).float())
            total_validation_correct += correct

        validation_accuracy = total_validation_correct / len(val_set)
        writer.add_scalar("Validation Accuracy", validation_accuracy, step_counter)
        print(
            "Valdation Accuracy",
            validation_accuracy.cpu().detach().numpy(),
            'Train Accuracy',
            accuracy.cpu().detach().numpy(),
        )
        
        
        # finally, at the end of each epoch, save a checkpoint
        torch.save(model.state_dict(), "resnet34_{:02d}.pth".format(epoch))

    # save our final model with a canonical name rather than relying on epoch numbers
    torch.save(model.state_dict(), "resnet34_final.pth")
