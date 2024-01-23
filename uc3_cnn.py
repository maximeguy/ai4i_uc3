####################################################################################################################
#########################                           Imports                                 ########################
####################################################################################################################
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
import keras.utils
import tensorflow as tf
from uc3 import IMGS_REPORT, IMGS_STUDIO
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import os
import glob
####################################################################################################################
#########################                           Images                                  ########################
####################################################################################################################
print(len(IMGS_STUDIO))

####################################################################################################################
#########################                           DataSets Yassin                         ########################
####################################################################################################################
# Directory PATH to the studio and reportage images
studio_dir_path = "img/studio"
reportage_dir_path = "img/report"

# Extracts the path to every single image in the path
studio_image_files = glob.glob(os.path.join(studio_dir_path, '*.jpg'))
reportage_image_files = glob.glob(os.path.join(reportage_dir_path, '*.jpg'))

#Declare an array that contains all the images, studio first, followed by reportage
X = np.zeros(((len(studio_image_files)+len(reportage_image_files)), 240, 360, 3), dtype=np.float32)
cnt = 0

# imports the images
for img in studio_image_files:
    im = tf.keras.preprocessing.image.load_img(img)
    x_img = tf.keras.preprocessing.image.img_to_array(im)
    
    # conditions
    if x_img.shape != (240, 360, 3):
        x_img = np.resize(x_img, (240, 360, 3))
    X[cnt] = x_img/255.0
    cnt +=1

for img in reportage_image_files:
    im = tf.keras.preprocessing.image.load_img(img)
    x_img = tf.keras.preprocessing.image.img_to_array(im)
    
    # conditions
    if x_img.shape != (240, 360, 3):
        x_img = np.resize(x_img, (240, 360, 3))
    X[cnt] = x_img/255.0
    cnt +=1
    
# Declare an array that contains the classes
y = np.array(697*[0]+701*[1])

train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
####################################################################################################################
#########################                           Parameters                              ########################
####################################################################################################################
base_epochs = 20 
model_epochs = 20
output_classes = 2
####################################################################################################################
#########################                               Model                               ########################
####################################################################################################################
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet',input_shape=(240, 360, 3), include_top=False)

base_model.trainable = False
inputs = keras.Input(shape=(240, 360, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
history = model.fit(x=train_data,y=train_label,epochs=base_epochs,validation_data=(test_data,test_label))

y_pred = model.predict(test_data,verbose=2)
np.save('y_pred.npy',y_pred)

plt.subplot(1, 2, 1)
    
plt.plot(history.epoch, history.history["loss"], 'g', label='Training loss')
plt.plot(history.epoch, history.history["val_loss"], 'r', label='validation loss')
plt.title('Training & validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.subplot(1, 2, 2)
plt.plot(history.epoch, history.history["accuracy"], 'g', label='Training accuracy')
plt.plot(history.epoch, history.history["val_accuracy"], 'r', label='validation accuracy')
plt.title('Training & validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()