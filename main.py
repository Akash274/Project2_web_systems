'''
Packages needed before running this file
Mac_OSX:
Flask: pip3 install flask
WTF and WTForms: pip3 install flask_wtf wtforms

Windows:
Flask: pip install flask
WTF and WTForms: pip install flask_wtf wtforms
'''
from distutils.log import debug
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import pickle
import tensorflow as tf
from tensorflow.python import keras
import cv2
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras .preprocessing import image


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
app.config['FILE_UPLOAD_FOLDER'] = 'static/files'   #File is uploaded to the location given here
#pklfile = open('./model1_Project_2.h5', 'rb')
#model = pickle.load(pklfile)
model = keras.models.load_model('./model1_Project_2_4 .h5')

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route("/",  methods=['GET', "POST"])
@app.route("/home", methods=['GET', "POST"])
def home():
    form = UploadFileForm()


    if form.validate_on_submit():
        file = form.file.data
        #img_file = form.file


        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['FILE_UPLOAD_FOLDER'],
                               secure_filename(file.filename)))
        img_file = cv2.imread("./static/files/{}".format(file.filename))
        img = tf.expand_dims(img_file, axis=0)
        prediction = model.predict(img)
        print(prediction)
        # if prediction[0][1] < prediction[0][0]:
        #     result = "Benign"
        # else:
        #     result = "Malignant"
        #
        # predict = img
        # # .load_img(imagepath, target_size = (224, 224))
        # predict_modified = image.img_to_array(predict)
        # predict_modified = predict_modified / 255
        # predict_modified = tf.expand_dims(predict_modified, axis=0)
        # predict_modified = tf.reshape(predict_modified, [1, 150528])
        result = model.predict(img)
        if result[0][0] >= 0.5:
            prediction = 'malignant'
            probability = result[0][0]
            print("probability = " + str(probability))
        else:
            prediction = 'beningn'
            probability = 1 - result[0][0]
            print("probability = " + str(probability))
            print("Prediction = " + prediction)
        return "The file has been uploaded\n Prediction: {}".format(prediction )


    return render_template('index.html', form=form)


if __name__ == "__main__":
    app.run(debug=True)