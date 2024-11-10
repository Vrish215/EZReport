import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def load_images_from_folder(folder, label, img_size=(224, 224)):
    images, labels = [], []
    for filename in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return images, labels
X, y = [], []
yes_images, yes_labels = load_images_from_folder("brain_mri_images/yes", label="Y")
no_images, no_labels = load_images_from_folder("brain_mri_images/no", label="N")
X.extend(yes_images + no_images)
y.extend(yes_labels + no_labels)
X = np.array(X)
y = np.array(y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.33, random_state=42)

#VGG16 model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def custom_layers(base_model, num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return predictions

vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg_base.layers:
    layer.trainable = False
num_classes = 2
custom_head = custom_layers(vgg_base, num_classes)
model = Model(inputs=vgg_base.input, outputs=custom_head)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Model Training
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)

#Plotting
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
