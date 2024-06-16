!pip install keras-tuner --upgrade -q

"""### Import Libraries"""

import keras_tuner
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers, losses
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

im = plt.imread("../input/asl-dataset/asl_dataset/f/hand1_f_bot_seg_3_cropped.jpeg",
                format='jpeg')
print("Image shape is :", im.shape)
# or using `io.BytesIO`
# im = plt.imread(io.BytesIO(urllib2.urlopen(url).read()), format='jpeg')
plt.imshow(im, cmap='Greys_r')
plt.title('An image example')
plt.show()

"""### Load the data with keras.utils"""

BATCH_SIZE = 124
train_data = keras.utils.image_dataset_from_directory(
                        directory="../input/asl-dataset/asl_dataset",
                        labels= 'inferred',#[i for i in range(0,27)],
                        label_mode='categorical',
                        color_mode='rgb',
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        image_size=(180, 180),
)

"""### Create model"""
class MyHyperModel(keras_tuner.HyperModel) :
    def build(self, hp, classes=37) :
        model = keras.Sequential()
        model.add(layers.Input( (180,180,3)))
        model.add(layers.Resizing(128, 128, interpolation='bilinear'))
        # Whether to include normalization layer
        if hp.Boolean("normalize"):
            model.add(layers.Normalization())

        drop_rate = hp.Float("drop_rate", min_value=0.05, max_value=0.25, step=0.10, default=0.15)
        # Number of Conv Layers is up to tuning
        for i in range( hp.Int("num_conv", min_value=3, max_value=4, step=1), default=3) :
            # Tune hyperparams of each conv layer separately by using f"...{i}"
            model.add(layers.Conv2D(filters=hp.Choice(f"filters_{i}",[16,32,64], default=32),  #hp.Int(f"filters_{i}", min_value=16, max_value=64, step=16),
                                    kernel_size= hp.Int(f"kernel_", min_value=3, max_value=5, step=2, default=3),
                                    strides=1, padding='same',
                                    activation=hp.Choice("conv_act", ["relu"] , defalt='relu'))) #,"leaky_relu"] )))
            #model.add(layers.MaxPooling2D())
            # Batch Norm and Dropout layers as hyperparameters to be searched
            if hp.Boolean("batch_norm"):
                model.add(layers.BatchNormalization())
            if hp.Boolean("dropout"):
                model.add(layers.Dropout(drop_rate))

        model.add(layers.Flatten())
        for i in range(hp.Int("num_dense", min_value=1, max_value=3, step=1)) :
            model.add(layers.Dense(units=hp.Choice("neurons", [150, 200], default=150),
                                       activation=hp.Choice("mlp_activ", ['sigmoid', 'relu'], default='relu')))
            if hp.Boolean("batch_norm"):
                    model.add(layers.BatchNormalization())
            if hp.Boolean("dropout"):
                    model.add(layers.Dropout(drop_rate))

        # Last layer
        model.add(layers.Dense(classes, activation='softmax'))

        # Picking an opimizer and a loss function
        model.compile(optimizer=hp.Choice('optim',['adam','adamax']),
                      loss=hp.Choice("loss",["sparse_categorical_crossentropy"]), #,"kl_divergence"]),
                      metrics = ['accuracy'])

        # A way to optimize the learning rate while also trying different optimizers
        learning_rate = hp.Choice('lr', [ 0.03, 0.01, 0.003], default=0.01)
        K.set_value(model.optimizer.learning_rate, learning_rate)

        return model


    def fit(self, hp, model,x, *args, **kwargs) :

        return model.fit( x,
                         *args,
                         shuffle=hp.Boolean("shuffle"),
                         **kwargs)

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint

def get_callbacks(weights_file, patience, lr_factor):
  ''' Callbacks are used for saving the best weights and early stopping.'''
  return [
      # Only save the weights that correspond to the maximum validation accuracy.
      ModelCheckpoint(filepath= weights_file,
                      monitor="val_accuracy",
                      mode="max",
                      save_best_only=True,
                      save_weights_only=True),
      # If val_loss doesn't improve for a number of epochs set with 'patience' var
      # training will stop to avoid overfitting.
      EarlyStopping(monitor="val_loss",
                    mode="min",
                    patience = patience,
                    verbose=1),
      # Learning rate is reduced by 'lr_factor' if val_loss stagnates
      # for a number of epochs set with 'patience/2' var.
      ReduceLROnPlateau(monitor="val_loss", mode="min",
                        factor=lr_factor, min_lr=1e-6, patience=patience//2, verbose=1)]

"""### Sanity Check
#### Check everything works correctly by training a "dummy" model for 1 epoch with random data.
"""

classes = 37
hp = keras_tuner.HyperParameters()
hypermodel = MyHyperModel()
model = hypermodel.build(hp, classes)
hypermodel.fit(hp, model, np.random.rand(BATCH_SIZE, 400, 400,3), np.random.rand(BATCH_SIZE, 1))

"""## Initiate the tuner"""

tuner = keras_tuner.BayesianOptimization(
                        hypermodel=MyHyperModel(),
                        objective = "val_accuracy",
                        max_trials = 2,
#                         factor=30,
#                         max_epochs=50,
                        overwrite=True,
                        directory='BO_search_dir',
                        project_name='sign_language_cnn')
#tuner.search_space_summary(extended=False)

"""### Load the training data before searching the hyperparameter space."""

# epochs defines how many epochs each candidate model will be trained for
tuner.search(x=train_data, epochs=2, validation_data=val_data, batch_size=BATCH_SIZE)

# # Get the top 2 models.
# models = tuner.get_best_models(num_models=2)
# best_model = models[0]
# # Build the model.
# # Needed for `Sequential` without specified `input_shape`.
# best_model.build(input_shape=(None, 400, 400))
# best_model.summary()

# Tuner summary results
tuner.results_summary(1)

# Get the top 2 hyperparameters.
best_hps = tuner.get_best_hyperparameters(1)
# Build the model with the best hp.
h_model = MyHyperModel()
model = h_model.build(best_hps[0])
# Fit with the entire dataset.
combined_dataset = train_data.concatenate(val_data)
history = model.fit(x=train_data, validation_data=val_data, epochs=15,
                    callbacks=get_callbacks('Net_weights.h5',
                                            patience=10,
                                            lr_factor=0.3))

model.load_weights('Net_weights.h5')
model.evaluate(test_data)

model.save('Best_model')

predictions = model.predict(test_data)

def plot_training_curves(history, val=True):
    #Defining the metrics we will plot.
    train_acc=history.history['accuracy']
    train_loss = history.history['loss']

    if val :
        val_acc=history.history['val_accuracy']
        val_loss = history.history['val_loss']

    #Range for the X axis.
    epochs = range(len(train_loss))

    fig,axis=plt.subplots(1,2,figsize=(20,8))#1 row, 2 col , width=20,height=8 inches.

    #Plotting Loss figures.
    plt.rcParams.update({'font.size': 22}) #configuring font size.
    plt.subplot(1,2,1) #plot 1st curve.
    plt.plot(epochs,train_loss,c="red",label="Training Loss") #plotting
    if val: plt.plot(epochs,val_loss,c="blue",label="Validation Loss")
    plt.xlabel("Epochs") #title for x axis
    plt.ylabel("Loss")   #title for y axis
    plt.legend()

    #Plotting Accuracy figures.
    plt.subplot(1,2,2) #plot 2nd curve.
    plt.plot(epochs,train_acc,c="red",label="Training Acc") #plotting
    if val : plt.plot(epochs,val_acc,c="blue",label="Validation Acc")
    plt.xlabel("Epochs")   #title for x axis
    plt.ylabel("Accuracy") #title for y axis
    plt.legend()

plot_training_curves(history, val=True)

predictions