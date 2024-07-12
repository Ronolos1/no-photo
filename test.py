import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random
from pymongo import MongoClient

from train_product_model import train_generator

# Load the model
model = tf.keras.models.load_model('product_model_resnet50.h5')

# MongoDB connection setup
connection_string = 'mongodb+srv://officialreca0:eswxP6699ruM1jWT@cluster0.yhis61q.mongodb.net/?retryWrites=true&w=majority'
client = MongoClient(connection_string)

# Define image path for testing
image_path = r'C:\Users\SMART\Desktop\images (1).jfif'

# Check if the test image exists
if not os.path.exists(image_path):
    raise ValueError(f"The image {image_path} does not exist.")

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))  # Using 224x224 for ResNet50
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
predictions = model.predict(img_array)

# Get top prediction and its probability
top_index = np.argmax(predictions[0])
top_probability = predictions[0][top_index]

# Get class label
class_labels = list(train_generator.class_indices.keys())
predicted_class = class_labels[top_index]

# Print top prediction
print(f"Top prediction: {predicted_class} with probability: {top_probability}")

# Load data from MongoDB
def load_data_from_mongo(db_name, collection_name):
    db = client[db_name]
    collection = db[collection_name]
    data = list(collection.find())
    return data

# Get some products from the predicted group
def get_recommendations(group_name, db_name, collection_name='products', percentage=0.8):
    data = load_data_from_mongo(db_name, collection_name)
    group_data = [item for item in data if str(item['_id']) in product_groups[group_name]]
    sample_size = int(len(group_data) * percentage)
    recommendations = random.sample(group_data, sample_size)
    return recommendations

# Product groups
product_groups = {
    "group1": ["668930c15e724b282ae05021", "668930d25e724b282ae0502c", "66892d115e724b282ae04fe9"],
    "group2": ["668e6801247a3aad812934f6", "6689aea7bb3871963fbb6d9d", "6689aec7bb3871963fbb6da3", "6689aef0bb3871963fbb6da9", "6689af14bb3871963fbb6db4"],
    "group3": ["6689ac4cbb3871963fbb6d93", "6689ac11bb3871963fbb6d7d", "6689ac31bb3871963fbb6d88", "668931f15e724b282ae05037", "668932125e724b282ae05063"]
}

# Get and print recommendations
recommendations = get_recommendations(predicted_class, 'test')
print(f"Recommendations for group '{predicted_class}':")
for rec in recommendations:
    print(rec)
