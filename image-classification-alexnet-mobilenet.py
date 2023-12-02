import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

def create_alexnet():
    model = Sequential()

    # Layer 1
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(227, 227, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # Layer 2
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # Layer 3
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))

    # Layer 4
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))

    # Layer 5
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # Flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='softmax'))  # Assuming 1000 classes for ImageNet

    return model

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to the correct shape
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

if __name__ == "__main__":
    # Create AlexNet-like model
    model = create_alexnet()

    # Load MobileNetV2 model with pre-trained ImageNet weights
    mobilenet_model = MobileNetV2(weights='imagenet')

    # Path to the image you want to classify
    image_path = 'image/daisies-1393484.jpg'

    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)

    # Classify the image using MobileNetV2
    predictions = mobilenet_model.predict(img)

    # Decode and print the top-3 predicted classes
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")
