import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from google.colab import files

# Load your saved model
model = load_model('diabetic_retinopathy_model.h5')

# Define the class names exactly as they were in your training generator
class_names = ['Diabetics Not Affected', 'Diabetics Affected']

# --- Image Upload and Preprocessing ---
uploaded = files.upload()

# Get the first (and only) file uploaded
for fn in uploaded.keys():
  
  # Load the image from the uploaded file
  img = image.load_img(fn, target_size=(100, 100)) # Resize to 100x100
  
  # Convert the image to a numpy array
  x = image.img_to_array(img)
  
  # Rescale the image just like you did for training (1/255)
  x = x / 255.0
  
  # Add a 4th dimension (batch size) because the model expects it
  # Shape changes from (100, 100, 3) to (1, 100, 100, 3)
  x = np.expand_dims(x, axis=0)

  # --- Make the Prediction ---
  prediction = model.predict(x)
  
  # `prediction` will be an array like [[0.1, 0.9]]
  # np.argmax finds the index of the highest probability
  predicted_class_index = np.argmax(prediction)
  
  # Get the human-readable class name
  predicted_class_name = class_names[predicted_class_index]
  
  # Get the confidence score
  confidence_score = np.max(prediction) * 100
  
  # Print the result
  print(f"Prediction: {predicted_class_name}")
  print(f"Confidence: {confidence_score:.2f}%")

  # Optional: Display the image you just tested
  import matplotlib.pyplot as plt
  plt.imshow(img)
  plt.title(f"Prediction: {predicted_class_name}")
  plt.axis('off')
  plt.show()
    # Optional: Display the image you just tested
  # Optional: Display the image you just tested
  # Optional: Display the image you just tested
  # Optional: Display the image you just tested
