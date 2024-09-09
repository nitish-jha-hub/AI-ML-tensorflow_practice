from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')


# Step 1: Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Step 2: Load and preprocess the image
img_path = 'download.jpg'  # Change this to the path of your image
img = image.load_img(img_path, target_size=(224, 224))  # Ensure the image is resized
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = preprocess_input(img_array)  # Preprocess the image (normalize it)

# Step 3: Make predictions
predictions = model.predict(img_array)  # This should now work correctly

# Step 4: Decode and display predictions
decoded_predictions = decode_predictions(predictions, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i+1}. {label}: {score:.2f}")
