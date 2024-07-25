from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def load_pickle_model(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model {model_path}: {e}")
        return None

def predict(values):
    model = None
    model_path = None
    if len(values) == 8:
        model_path = 'models/diabetes.pkl'
    elif len(values) == 26:
        model_path = 'models/breast_cancer.pkl'
    elif len(values) == 13:
        model_path = 'models/heart.pkl'
    elif len(values) == 18:
        model_path = 'models/kidney.pkl'
    elif len(values) == 10:
        model_path = 'models/liver.pkl'
    else:
        logging.error("Invalid number of input values.")
        return None

    model = load_pickle_model(model_path)
    if model is None:
        logging.error("Model could not be loaded.")
        return None

    try:
        values = np.asarray(values).reshape(1, -1)
        logging.debug(f"Input values reshaped for prediction: {values}")
        prediction = model.predict(values)[0]
        logging.info(f"Prediction successful: {prediction}")
        return prediction
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    if request.method == 'POST':
        try:
            to_predict_dict = request.form.to_dict()
            logging.debug(f"Received form data: {to_predict_dict}")
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            logging.debug(f"Converted input values for prediction: {to_predict_list}")
            pred = predict(to_predict_list)
            if pred is not None:
                return render_template('predict.html', pred=pred)
            else:
                return render_template('home.html', message="Model prediction failed.")
        except Exception as e:
            logging.error(f"Error in /predict route: {e}")
            return render_template('home.html', message=f"Error: {e}")
    return render_template('home.html', message="Please enter valid data.")

@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 3))
                img = img.astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
                logging.info(f"Malaria prediction successful: {pred}")
                return render_template('malaria_predict.html', pred=pred)
            else:
                return render_template('malaria.html', message="Please upload an image.")
        except Exception as e:
            logging.error(f"Error in /malariapredict route: {e}")
            return render_template('malaria.html', message=f"Error: {e}")
    return render_template('malaria.html')

@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 1))
                img = img / 255.0
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
                logging.info(f"Pneumonia prediction successful: {pred}")
                return render_template('pneumonia_predict.html', pred=pred)
            else:
                return render_template('pneumonia.html', message="Please upload an image.")
        except Exception as e:
            logging.error(f"Error in /pneumoniapredict route: {e}")
            return render_template('pneumonia.html', message=f"Error: {e}")
    return render_template('pneumonia.html')

if __name__ == '__main__':
    app.run(debug=True)
