from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the saved model
model_path = 'Tensorflow_Model'  # Update this path
model = load_model(model_path)


def prepare_image(img_path):
    """
    Preprocess the image to the format required by DenseNet121
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = densenet_preprocess_input(img_array)
    return img_array


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Prepare and predict
        prepared_img = prepare_image(file_path)
        prediction = model.predict(prepared_img)
        predicted_class = True if prediction[0][0] > 0.5 else False
        likelihood_score = prediction[0][0] * 100  # Convert to percentage

        response = {
            'status': 200,
            'prediction': predicted_class,
            'likelihoodScore': f"{likelihood_score:.2f}%"
        }
        return jsonify(response)
    else:
        return jsonify({
            'status': 400,
            'message': "Prediction went wrong. Please try again!"
        })


if __name__ == '__main__':
    app.run(debug=True)
