# !pip install tensorflow numpy pillow

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Define local paths
base_dir = r'C:\Users\HAFEEZ KHAN\Desktop\archive'  # Update this path
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Image size for DenseNet121
img_size = (224, 224)

# Training data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=densenet_preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation and test data generator
val_test_datagen = ImageDataGenerator(preprocessing_function=densenet_preprocess_input)

# Load the DenseNet121 base model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=img_size + (3,))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Initialize data generators
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // batch_size),
    verbose=1,
    epochs=5
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.n // batch_size)
# Save the model in TensorFlow's SavedModel format
model_save_path = r'C:\Users\HAFEEZ KHAN\Desktop\archive'
model.save(model_save_path)

# Save the model locally
model_save_path = r'C:\Users\HAFEEZ KHAN\Desktop\archive'
model.save(model_save_path)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
