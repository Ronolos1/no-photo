import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import logging

# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)

# Define paths
train_dir = './product_images/recon'

# Check if the training directory exists
if not os.path.exists(train_dir):
    raise ValueError(f"The directory {train_dir} does not exist.")

# Image data generator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Training and validation generators
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Using 224x224 for ResNet50
    batch_size=32,
    class_mode='categorical',  # Using 'categorical' for multi-class classification
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Using 224x224 for ResNet50
    batch_size=32,
    class_mode='categorical',  # Using 'categorical' for multi-class classification
    subset='validation'
)

# Build the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # Using 'softmax' for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',  # Using 'categorical_crossentropy' for multi-class classification
              metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // validation_generator.batch_size),
    epochs=10
)

# Save the model
model.save('product_model_resnet50.h5')
