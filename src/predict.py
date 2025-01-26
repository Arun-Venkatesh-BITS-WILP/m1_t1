import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# CIFAR-10 class labels
class_names = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", 
    "frog", "horse", "ship", "truck"
]

def predict(img_path):
    # Load the trained model (saved as .h5 file)
    model = tf.keras.models.load_model("models/cnn_model.h5")

    # Preprocess the image
    img = image.load_img(img_path, target_size=(32, 32))  # Resize image to 32x32 as required by CIFAR-10
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size
    img_array = tf.image.rgb_to_grayscale(img_array)  # Convert to grayscale using TensorFlow
    img_array = img_array / 255.0  # Normalize the image to [0, 1] range

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)  # Get the index of the class with the highest probability

    # Get the class name corresponding to the predicted index
    predicted_class_name = class_names[predicted_class_index[0]]

    return predicted_class_name


if __name__ == "__main__":
    img_path = r"D:\Users\KumarVe\Downloads\airplane.jpg"  # Replace with the path to your downloaded image
    print(f"Prediction: {predict(img_path)}")
