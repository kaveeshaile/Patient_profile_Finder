import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, Sequential  # Specific imports
import tensorflow.keras.models as keras_models  # Import entire module
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
import numpy as np
from tensorflow.keras import Input



#model loading

model1 = tf.keras.models.load_model('eye_detector.h5')
model2 = tf.keras.models.load_model('finger_detector.h5')

image_path = "finger.BMP"

def preprocess_image(image_path, target_size=(128, 128)):
   # Load the image with resizing to the target size
  img = load_img(image_path, target_size=target_size)
  # Convert the PIL image to a NumPy array
  img_array = img_to_array(img)
  # Normalize pixel values to the range [0, 1]
  img_array = img_array / 255.0
  # Add a batch dimension (None allows flexibility for batch size)
  img_array = np.expand_dims(img_array, axis=0)
  return img_array

preprocessed_image = preprocess_image(image_path)




while True:
  model_choice = input("Enter 1 for Eye Identification or 2 for Fingerprint  Identification: ")
  if model_choice in ('1', '2'):
    break
  else:
    print("Invalid choice. Please enter 1 or 2.")

#model 1 calling
if model_choice == '1':
        predictions = model1.predict(preprocessed_image)
  # Sample class labels (replace with your actual list of names)
        class_labels = [
        "Ace", "Ari", "Ash", "Bea", "Ben", "Bly", "Cam", "Cat", "Dax", "Echo",
        "Em", "Fox", "Gia", "Gus", "Jay", "Jem", "Kai", "Kit", "Liv", "Lux",
        "Mac", "May", "Max", "Nat", "Neo", "Nix", "Nax", "Pax", "Tax",
        "Pip", "Rae", "Ren", "Sam", "Scout", "Shay", "Skye", "Tate", "Ty", "Uma",
        "Val", "Wren", "Xander", "Yaz", "Zee"
        ]  # Assuming these correspond to prediction classes
        # Find the index of the highest probability in the prediction
        predicted_index = np.argmax(predictions, axis=1)  # Assumes probabilities are in the first dimension
        # Map the index to the corresponding name using class labels
        predicted_name = class_labels[predicted_index[0]]  # Assuming you only have one prediction
        print("Eye Detected Patient Name:", predicted_name)




if model_choice == '2':
        predictions1 = model2.predict(preprocessed_image)
       
        # Sample class labels (replace with your actual list of names)
        class_labels = [
        "Ace", "Ari", "Ash", "Bea", "Ben", "Bly", "Cam", "Cat", "Dax", "Echo",
        "Em", "Fox", "Gia", "Gus", "Jay", "Jem", "Kai", "Kit", "Liv", "Lux",
        "Mac", "May", "Max", "Nat", "Neo", "Nix", "Nax", "Pax", "Tax",
        "Pip", "Rae", "Ren", "Sam", "Scout", "Shay", "Skye", "Tate", "Ty", "Uma",
        "Val", "Wren", "Xander", "Yaz", "Zee"
        ]  # Assuming these correspond to prediction classes
        # Find the index of the highest probability in the prediction
        predicted_index = np.argmax(predictions1, axis=1)  # Assumes probabilities are in the first dimension
        # Map the index to the corresponding name using class labels
        predicted_names = class_labels[predicted_index[0]]  # Assuming you only have one prediction
        print("Fingerprint Detected Patient Name:", predicted_names)


