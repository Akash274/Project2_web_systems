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
import base64
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
app.config['FILE_UPLOAD_FOLDER'] = 'static/files'   #File is uploaded to the location given here

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route("/",  methods=['GET', "POST"])
@app.route("/home", methods=['GET', "POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['FILE_UPLOAD_FOLDER'], secure_filename(file.filename)))
        predict = predict_image_classification_sample(
            project="480163682977",
            endpoint_id="9069378836314980352",
            location="us-central1",
            filename="./static/files/{}".format(file.filename)
)
        for prediction in predict:
            print(" prediction:", dict(prediction))
        if str(prediction.get('displayNames')) == "['benign']":
            result = 'Benign'
        else:
            result = 'Malignant'
        return render_template('index.html', form=form, message =r"The mole is {}".format(result))
    return render_template('index.html', form=form)

def predict_image_classification_sample(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5, max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))
    return predictions


if __name__ == "__main__":
    app.run(debug=True)