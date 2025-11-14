import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import json

# -------------------------------
# Paths
# -------------------------------
DATASET_DIR = "D:/plant/final_dataset/new_dataset"
MODEL_SAVE_PATH = "trained_plant_disease_model.keras"
CLASS_INDICES_PATH = "class_indices.json"

# -------------------------------
# Image Data Generator with Augmentation
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% data for validation
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Save class indices
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(train_generator.class_indices, f)

# -------------------------------
# Model Architecture
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------------
# Callbacks
# -------------------------------
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1)

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=35,  # increased to 35
    callbacks=[checkpoint, early_stop]
)

# -------------------------------
# Save final model (optional)
# -------------------------------
model.save(MODEL_SAVE_PATH)
print("✅ Training complete and model saved!")
