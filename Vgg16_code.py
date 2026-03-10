# General imports
import numpy as np # linear algebra
import os
from glob import glob
import matplotlib.pyplot as plt


# Training and testing folders
train_path = './Data/'
valid_path = './Data/'
# Get train and test files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')
# Get number of classes
folders = glob(train_path + '/*')





# Specific imports
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
# Resize all the images to this
IMAGE_SIZE = [100, 100]
# Training config
epochs = 10
batch_size = 128
#vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights=None, include_top=False)
# Don't train existing weights
for layer in vgg.layers:
  layer.trainable = False

x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)
# Create Model
model = Model(inputs=vgg.input, outputs=prediction)

# View structure of the model
model.summary()

# Configure model
model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)



# Define your ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
    preprocessing_function=preprocess_input
)

# Create generators for training and validation data
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = train_datagen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Fit the model using Model.fit
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=len(valid_generator)
)





# Plot the train and validation accuracies
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()



print(" accuracy = {}".format(history.history["accuracy"][-1]))

model.save("vgg16.h5")



from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

tar = 4
path = './dataset/'

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), tar)
    return files, targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset(path)

test_files = train_files
test_targets = train_targets


# We only take the characters from a starting position to remove the path
burn_classes = [item[10:-1] for item in sorted(glob("./dataset/*/"))]

# print statistics about the dataset
print('There are %d total categories.' % len(burn_classes))
print(burn_classes)
print('There are %s total images.\n' % len(np.hstack([train_files, test_files])))
print('There are %d training images.' % len(train_files))
print('There are %d test images.'% len(test_files))

for file in train_files:
    assert('.DS_Store' not in file)

from tensorflow.keras.preprocessing import image                  
from tqdm import tqdm

# Note: modified these two functions, so that we can later also read the inception tensors which 
# have a different format 
def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, height, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_path, width, height) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

import keras
import timeit

# graph the history of model.fit
def show_history_graph(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
     

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32') / 255
test_tensors = paths_to_tensor(test_files).astype('float32') / 255

from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint 

import matplotlib.pyplot as plt
img_width, img_height = 224, 224
batch_size = 4
epoch = 10
img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 3
lr = 0.0004
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import callbacks
import time
#input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(tar, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

hist = model.fit(train_tensors, train_targets, validation_split=0.3, epochs=epoch, batch_size=10)

# Print the accuracy
print("Final Accuracy:", hist.history['accuracy'][-1])

show_history_graph(hist)

model.save('trained_model_CNN.h5')

# Extract the final accuracy values of both models
vgg_accuracy = history.history["accuracy"][-1]
cnn_accuracy = hist.history["accuracy"][-1]

# Plot the comparison bar chart
models = ['VGG16', 'CNN']
accuracies = [vgg_accuracy, cnn_accuracy]

plt.bar(models, accuracies, color=['blue', 'orange'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim(0, 1)  # Set y-axis limit to ensure proper scale
plt.show()

