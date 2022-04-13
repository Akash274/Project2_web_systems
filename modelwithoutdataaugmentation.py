from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


#import tensorflow.python.keras as tk
import keras
from keras.preprocessing.image import ImageDataGenerator

base_dir = os.path.dirname('data/')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')
print(train_dir)

train_malignant_dir = os.path.join(train_dir, 'malignant')
train_benign_dir = os.path.join(train_dir, 'benign')
validation_malignant_dir = os.path.join(validation_dir, 'malignant')
validation_benign_dir = os.path.join(validation_dir, 'benign')

# train_malignant_dir = os.listdir(('data/train/malignant'))
# train_benign_dir = os.listdir('data/train/benign')
# validation_benign_dir = os.listdir('data/test/benign')
# validation_malignant_dir = os.listdir('data/test/malignant')

num_malignant_tr = len(os.listdir(train_malignant_dir))
num_benign_tr = len(os.listdir(train_benign_dir))

num_malignant_val = len(os.listdir(validation_malignant_dir))
num_benign_val = len(os.listdir(validation_benign_dir))

total_train = num_malignant_tr + num_benign_tr
total_val = num_malignant_val + num_benign_val

print('total training malignant images:', num_malignant_tr)
print('total training benign images:', num_benign_tr)

print('total validation malignant images:', num_malignant_val)
print('total validation benign images:', num_benign_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

BATCH_SIZE = 100
IMG_SHAPE  = 224

train_image_generator      = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE),
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

#plotImages(sample_training_images[:5])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=( 224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
EPOCHS = 50
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE))),
    callbacks=[early_stop]
)

def predict_image(image, classifier):
    predict = image
        #.load_img(imagepath, target_size = (224, 224))
    predict_modified = image.img_to_array(predict)
    predict_modified = predict_modified / 255
    predict_modified = np.expand_dims(predict_modified, axis = 0)
    result = classifier.predict(predict_modified)
    if result[0][0] >= 0.5:
        prediction = 'malignant'
        probability = result[0][0]
        print ("probability = " + str(probability))
    else:
        prediction = 'beningn'
        probability = 1 - result[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + prediction)
    return prediction

model.save('wap2m1')


import pickle

file_name = 'project1_mlmodel3.pkl'
save_model = pickle.dump(model, open(file_name,'wb'))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

model.save("model1_Project_2_4 .h5")

epochs_range = range(EPOCHS)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.savefig('./foo.png')
# plt.show()

