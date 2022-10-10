'''

Image Recognition Using Machine Learning

'''
import os
import time
import pickle
import numpy as np
import glob
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Keras
import keras 
from keras.applications import EfficientNetB0,ResNet50
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,Input
from keras.optimizers import Adam
import keras.metrics as Kmetrics
from keras.callbacks import ModelCheckpoint

  
# fix random seed for reproduciblity
seed = 1169
np.random.seed(seed)

# tell the application whether we are running on a server or not (so as to
# influence which backend matplotlib uses for saving plots)
headless = False

# load the npz file
data_path = os.getcwd()+'/DATASET_NPZ/preproc_train_test_224.npz'
train_data   = np.load(data_path)

# extract the training and validation data sets from this data
x_train = train_data['x_train']
y_train = train_data['y_train']
x_test= train_data['x_test']
y_test = train_data['y_test']

'''# Reshape the images.
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
'''

print('Input Shape')
print(np.shape(x_train))
print(np.shape(x_test))

input_shape = np.asarray(np.shape(x_train[0]))
n_classes = len(np.unique(y_train))

'''
# CNN
# 3x3 Kernels
model = Sequential()
model.add(Conv2D(filters = 256,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(filters = 128,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(filters = 64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 128,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(filters = 256,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 128,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(filters = 64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

'''
#EfficientNetB0
# Create the Model 
inputs = Input(shape = input_shape)

x = inputs
outputs = EfficientNetB0(include_top=True, weights = None, classes=n_classes)(x)
model = keras.Model(inputs, outputs)


print("Summary of Model")
print(model.summary)


# set the optimiser
opt = Adam()
# compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

# set up a model checkpoint callback (including making the directory where to 
# save our weights)
directory = './model/initial_runs_{0}/'.format(time.strftime("%Y%m%d_%H%M"))
os.makedirs(directory)
filename  = 'conv_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpointer = ModelCheckpoint(filepath=directory+filename, 
                               verbose=1, 
                               save_best_only=True)

# set the model parameters
n_epochs = 10 
batch_size = 16 #n_images

# fit the model
history = model.fit(x=x_train, y = y_train,
                    epochs=n_epochs,
                    batch_size= batch_size,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=[checkpointer])

# get the best validation accuracy
best_accuracy = max(history.history['val_acc'])
print('best validation accuracy = {0:f}'.format(best_accuracy))

# pickle the history so we can use it later
with open(directory + 'training_history', 'wb') as file:
    pickle.dump(history.history, file)

# set matplotlib to use a backend that doesn't need a display if we are 
# running remotely
if headless:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# plot the results
plot_dir = os.getcwd() + '/Plots/'
if os.path.exists(plot_dir) == False:
        print('Creating Folder:   ',plot_dir)
        os.makedirs(plot_dir)
# accuracy
f1 = plt.figure()
ax1 = f1.add_subplot(111)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training and Validation Accuracy')
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.text(0.4, 0.05, 
         ('validation accuracy = {0:.3f}'.format(best_accuracy)), 
         ha='left', va='center', 
         transform=ax1.transAxes)
plt.savefig(plot_dir+'/Training_accuracy_{0}.png'
            .format(time.strftime("%Y%m%d_%H%M")))

# loss
f2 = plt.figure()
ax2 = f2.add_subplot(111)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.text(0.4, 0.05, 
         ('validation loss = {0:.3f}'
          .format(min(history.history['val_loss']))), 
         ha='right', va='top', 
         transform=ax2.transAxes)
plt.savefig(plot_dir+'/Training_loss_{0}.png'
            .format(time.strftime("%Y%m%d_%H%M")))

# we're all done!
print('all done!')