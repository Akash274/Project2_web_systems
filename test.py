import tensorflow as tf
import pickle
import os
import cv2
pklfile= open('./project1_model.pkl', 'rb')
model = pickle.load(pklfile)
img_file = cv2.imread('./benign.0.jpg')
img = tf.expand_dims(img_file, axis=0)

print(model.predict(img))