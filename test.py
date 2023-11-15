from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
import numpy as np

def prepare_image(img_path):
    """
    Preprocess the image to the format required by DenseNet121
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = densenet_preprocess_input(img_array)
    return img_array

def predict_image(model, img_path):
    """
    Predict the class of an image using the provided model
    """
    print(f"Processing image: {img_path}")
    prepared_img = prepare_image(img_path)
    prediction = model.predict(prepared_img)
    predicted_class = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
    likelihood_score = prediction[0][0]
    return predicted_class, likelihood_score

# Load the saved model
model_path = r'C:\Users\HAFEEZ KHAN\Desktop\Tensorflow_Model'  # Update this path
model = load_model(model_path)

# Test image path - update this with the path to your test image
test_image_path = r'C:\Users\HAFEEZ KHAN\Desktop\testimage.jpeg'  # Update this path

# Perform prediction
result, score = predict_image(model, test_image_path)
print(f"The image is predicted as: {result}")
print(f"Likelihood Score (Probability of Pneumonia): {score * 100:.2f}%")

