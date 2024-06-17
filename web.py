import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, render_template, jsonify

# Model loading
model1 = tf.keras.models.load_model('eye_detector.h5')
model2 = tf.keras.models.load_model('finger_detector.h5')

# Function to preprocess image
def preprocess_image(image, target_size=(128, 128)):
  img = load_img(image, target_size=target_size)
  img_array = img_to_array(img)
  img_array = img_array / 255.0
  img_array = np.expand_dims(img_array, axis=0)
  return img_array

# Initialize Flask app
app = Flask(__name__)

# Route for homepage with file upload form
@app.route("/", methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    # Get uploaded file
    uploaded_file = request.files['image']
    # Check if file is uploaded
    if uploaded_file.filename != '':
      try:
        # Read the uploaded image bytes
        image_bytes = uploaded_file.read()
        
        # Check for empty file
        if not image_bytes:
          return render_template('index.html', error="Empty file uploaded.")
        
        # Preprocess the image
        with open("temp_image.jpg", "wb") as f:
          f.write(image_bytes)
        try:
          preprocessed_image = preprocess_image("temp_image.jpg")
        except OSError as e:
          # Handle truncated file error
          if "cannot identify image file" in str(e):
            return render_template('index.html', error="Corrupted or truncated image uploaded.")
          else:
            raise e  # Re-raise other OSError exceptions
        
        # Remove the temporary file
        import os
        os.remove("temp_image.jpg")
        
        # Get user selection from form (eye or fingerprint)
        model_choice = request.form['model_choice']
        
        # Process image based on selection
        if model_choice == '1':
          predictions = model1.predict(preprocessed_image)
        elif model_choice == '2':
          predictions = model2.predict(preprocessed_image)
        else:
          return render_template('index.html', error="Invalid selection. Choose 1 or 2.")
        
        # ... rest of your prediction logic ...
        class_labels = [
          "Miler Kirteen", "Ari", "Ash", "Bea", "Ben", "Bly", "Cam", "Cat", "Dax", "Echo",
          "Em", "Fox", "GiaAlexsandra", "Gus", "Jay", "Jem", "Kai", "Kit", "Liv", "Lux",
          "Mac", "May", "Max", "Nat", "Neo", "Nix", "Nax", "Pax", "Tax",
          "Pip", "Rae", "Ren", "Sam", "Scout", "Shay", "Skye", "Tate", "Ty", "Uma",
          "Val", "Wren", "Xander", "Yaz", "Zee","Ace Alexsandra"
        ]
        
        # Find the predicted class
        predicted_index = np.argmax(predictions, axis=1)
        predicted_name = class_labels[predicted_index[0]]
        return render_template('index.html', prediction=predicted_name)
      except Exception as e:
        # Handle any other unexpected errors
        return render_template('index.html', error=f"An error occurred: {str(e)}")
    else:
      # Render the homepage with the form
      return render_template('index.html')
  else:
    # Render the homepage with the form
    return render_template('index.html')
  
# Route to display image based on predicted name
@app.route("/view/<patient_name>")
def view(patient_name):
  return render_template('view.html', patient_name=patient_name)

@app.route("/", methods=["POST"])
def back_to_index():
  return render_template("index.html")



if __name__ == '__main__':
  app.run(debug=True)
