from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import codecs
import json
import os
import sys
import glob as glob
import numpy as np
import datetime
import cv2
import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
import tensorflow
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

#%matplotlib inline

from tensorflow.keras.utils import img_to_array, load_img
import tensorflow as tf


# default settings
img_width, img_height = 224, 224

train_dir = '/Users/ayberk/Retina-Damage-Detection-OCT-Images/Dataset/train'
validate_dir = '/Users/ayberk/Retina-Damage-Detection-OCT-Images/Dataset/test'
nb_epochs = 20
batch_size = 16
nb_classes = len(glob.glob(train_dir + '/*'))

nb_train_samples = sum(len(files) for _, _, files in os.walk(train_dir))
print(nb_train_samples)

nb_validate_samples = sum(len(files) for _, _, files in os.walk(validate_dir))
print(nb_validate_samples)

# data pre-processing for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True
)

# data pre-processing for validation
validate_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True
)

# generate and store training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size
)

# generate and store validation data
validate_generator = validate_datagen.flow_from_directory(
    validate_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size
)
# Calculate class weight
total = len(validate_dir)
count_cnv = 2874
count_dme = 2760
count_drusen = 2766
count_normal = 2598
cnv_weight = (1/count_cnv) * (total/4)
dme_weight = (1/count_dme) * (total/4)
drusen_weight = (1/count_drusen) * (total/4)
norm_weight = (1/count_normal) * (total/4)
class_weight = {0: cnv_weight, 1: dme_weight, 2: drusen_weight, 3: norm_weight}
print("Class Weights: 0CVN - 1DME - 2DRUSEN - 3NORMAL"  + str(class_weight))

# set up transfer learning on pre-trained ImageNet VGG16 model
# remove fully connected layer and replace with softmax for classifying nb_classes classes
vgg16_model = VGG16(weights='imagenet', include_top=False)
x = vgg16_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(nb_classes, activation='softmax')(x)
model = Model(inputs=vgg16_model.input, outputs=predictions)

# freeze all layers of the pre-trained model
for layer in vgg16_model.layers:
    layer.trainable = False

# compile the new model using an RMSprop optimizer
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# fit the model, log the results and the training time
now = datetime.datetime.now
t = now()
transfer_learning_history = model.fit(
    train_generator,
    epochs=nb_epochs,
    steps_per_epoch=nb_train_samples // batch_size,
    validation_data=validate_generator,
    validation_steps=nb_validate_samples // batch_size,
    class_weight=class_weight
)

model.save("octvgg16.h5")
with open("octvgg16.json", "w") as f:
    json.dump(transfer_learning_history.history, f)

print('Training time: %s' % (now() - t))
  
# evaluate the performance the new model and report the results
score = model.evaluate(
    validate_generator, steps=nb_validate_samples/batch_size)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


xfer_acc = transfer_learning_history.history['accuracy']
val_acc = transfer_learning_history.history['val_accuracy']
xfer_loss = transfer_learning_history.history['loss']
val_loss = transfer_learning_history.history['val_loss']
epochs = range(len(xfer_acc))

x = np.array(epochs)
y = np.array(xfer_acc)
x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = interp1d(x, y, kind='linear')(x_smooth)
plt.plot(x_smooth, y_smooth, 'r-', label='Training')

x1 = np.array(epochs)
y1 = np.array(val_acc)
x1_smooth = np.linspace(x1.min(), x1.max(), 300)
y1_smooth = interp1d(x1, y1, kind='linear')(x1_smooth)

plt.plot(x1_smooth, y1_smooth, 'g-', label='Validation')
plt.title('Transfer Learning - Training and Validation Accuracy')
plt.legend(loc='lower left', fontsize=9)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1.02)

plt.figure()
x = np.array(epochs)
y = np.array(xfer_loss)
x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = interp1d(x, y, kind='linear')(x_smooth)
plt.plot(x_smooth, y_smooth, 'r-', label='Training')

x1 = np.array(epochs)
y1 = np.array(val_loss)
x1_smooth = np.linspace(x1.min(), x1.max(), 300)
y1_smooth = interp1d(x1, y1, kind='linear')(x1_smooth)

plt.plot(x1_smooth, y1_smooth, 'g-', label='Validation')
plt.title('Transfer Learning - Training and Validation Loss')
plt.legend(loc='upper right', fontsize=9)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, max(y1))
plt.show()


predictions = model.predict(validate_generator)
# Convert prediction probabilities into integers
predictions = np.argmax(predictions, axis=1)

class_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

class_names = class_labels

test = tf.keras.preprocessing.image_dataset_from_directory(
    validate_dir,
    labels='inferred',
    image_size=(224, 224),
    batch_size=16,
    color_mode='rgb',
)
test_labels = [labels for _, labels in test.unbatch()]

cm = confusion_matrix(test_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()


print(validate_generator.total_batches_seen)
