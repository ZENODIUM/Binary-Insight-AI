import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import tempfile
import visualkeras
import io
import shutil

# Temporary directories for uploaded images
class1_dir = tempfile.mkdtemp()
class2_dir = tempfile.mkdtemp()

# Directories for sample images
sample_class1_dir = 'sample_images/mini_no'
sample_class2_dir = 'sample_images/mini_yes'

st.title(":blue[Binary Insight AI] (Your Binary Neural Network Playground)")
st.write("---")
st.subheader("Info")
st.markdown(''':blue[Model Architecture Display:]
Visualizes and updates the neural network architecture in real-time using VisualKeras.

:blue[Custom Neural Network Configuration:]
Configures the number of layers, filters, kernel sizes, and neurons for tailored model design.
            
:blue[Pretrained Model Option:]
Offers VGG16 and ResNet50 for leveraging advanced pre-trained models.
            
:blue[Early Stopping:]
Provides an option to halt training early to avoid overfitting based on validation performance.
            
:blue[Image Upload and Training:]
Supports training with manually uploaded images for custom data.
            
:blue[Sample Dataset Upload and Training:]
Allows quick training on a provided sample dataset with a single button click.
            
:blue[Real-time Metrics Tracking:]
Tracks and displays training and validation metrics with real-time plots and tables.
            
:blue[Progress Tracking:]
Shows training progress through a visual progress bar.
            
:blue[Model Saving and Downloading:]
Enables saving and downloading of the trained model for later use.
            
:blue[Sample Image Display:]
Displays sample images from each class side by side for visual inspection.''')
st.write("---")

# Upload images for Class 1
st.subheader("Upload Images for Class 1")
uploaded_class1_images = st.file_uploader("Choose Class 1 Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

# Save uploaded images for Class 1
if uploaded_class1_images:
    for i, img in enumerate(uploaded_class1_images):
        img = Image.open(img).convert("RGB")
        img_path = os.path.join(class1_dir, f"class1_image_{i}.jpg")
        img.save(img_path)

# Upload images for Class 2
st.subheader("Upload Images for Class 2")
uploaded_class2_images = st.file_uploader("Choose Class 2 Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

# Save uploaded images for Class 2
if uploaded_class2_images:
    for i, img in enumerate(uploaded_class2_images):
        img = Image.open(img).convert("RGB")
        img_path = os.path.join(class2_dir, f"class2_image_{i}.jpg")
        img.save(img_path)
st.write("---")
# Display sample images from each class
st.subheader("Sample Images from Each Class")
col1, col2 = st.columns(2)
with col1:
    if uploaded_class1_images:
        st.image(uploaded_class1_images[0], caption="Class 1 Sample Image")
    else:
        sample_img_path = os.path.join(sample_class1_dir, os.listdir(sample_class1_dir)[0])
        st.image(sample_img_path, caption="Sample Class 1 Image")
with col2:
    if uploaded_class2_images:
        st.image(uploaded_class2_images[0], caption="Class 2 Sample Image")
    else:
        sample_img_path = os.path.join(sample_class2_dir, os.listdir(sample_class2_dir)[0])
        st.image(sample_img_path, caption="Sample Class 2 Image")

# Parameters for the neural network
st.sidebar.title(":blue[Binary Insight AI]")
st.sidebar.image('robot_image.png')
st.sidebar.title("Model Parameters")
model_type = st.sidebar.selectbox("Select Model Type", ["Custom CNN", "VGG16", "ResNet"])
if model_type == "Custom CNN":
    num_conv_layers = st.sidebar.slider("Number of Convolutional Layers", 1, 5, 2)
    conv_filters = [st.sidebar.slider(f"Filters in Conv Layer {i+1}", 16, 128, 32*(i+1)) for i in range(num_conv_layers)]
    kernel_size = st.sidebar.slider("Kernel Size", 2, 5, 3)
    pool_size = st.sidebar.slider("Pool Size", 2, 3, 2)
    num_dense_layers = st.sidebar.slider("Number of Dense Layers", 1, 3, 1)
    dense_neurons = [st.sidebar.slider(f"Neurons in Dense Layer {i+1}", 32, 512, 128*(i+1)) for i in range(num_dense_layers)]
else:
    num_dense_layers = st.sidebar.slider("Number of Dense Layers", 1, 3, 1)
    dense_neurons = [st.sidebar.slider(f"Neurons in Dense Layer {i+1}", 32, 512, 128*(i+1)) for i in range(num_dense_layers)]

learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.001, step=0.0001)
epochs = st.sidebar.slider("Number of Epochs", 1, 100, 10)
batch_size = st.sidebar.slider("Batch Size", 8, 64, 32, step=8)

optimizer = st.sidebar.selectbox(
    "Select Optimizer",
    ["Adam", "SGD", "RMSprop"]
)

early_stopping = st.sidebar.checkbox("Use Early Stopping")

def get_optimizer(name, learning_rate):
    if name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate)
    elif name == "SGD":
        return tf.keras.optimizers.SGD(learning_rate)
    elif name == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate)
    else:
        return tf.keras.optimizers.Adam(learning_rate)

# Building the model
def build_model():
    if model_type == "Custom CNN":
        model = Sequential()
        model.add(Conv2D(conv_filters[0], (kernel_size, kernel_size), activation='relu', input_shape=(128, 128, 3)))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        for filters in conv_filters[1:]:
            model.add(Conv2D(filters, (kernel_size, kernel_size), activation='relu'))
            model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(Flatten())

        for neurons in dense_neurons:
            model.add(Dense(neurons, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        model = Sequential([
            base_model,
            Flatten(),
        ])
        for neurons in dense_neurons:
            model.add(Dense(neurons, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "ResNet":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        model = Sequential([
            base_model,
            Flatten(),
        ])
        for neurons in dense_neurons:
            model.add(Dense(neurons, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    opt = get_optimizer(optimizer, learning_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Ensure the model is built by making a dummy forward pass
dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
model(dummy_input)

# Display neural network architecture
architecture_placeholder = st.empty()

def update_architecture_image():
    # Save the model architecture to an image file
    img_path = "model_architecture.png"
    visualkeras.layered_view(model, to_file=img_path)
    
    # Display the image in Streamlit
    with open(img_path, "rb") as img_file:
        img_bytes = img_file.read()
        architecture_placeholder.image(img_bytes, use_column_width=True)

update_architecture_image()

# Capture the model summary in a buffer
buffer = io.StringIO()
model.summary(print_fn=lambda x: buffer.write(x + "\n"))
summary_text = buffer.getvalue()
buffer.close()
st.text(summary_text)

# Load images and preprocess
def load_and_preprocess_images(directory, label, preprocess_func=None):
    data = []
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img)
        if preprocess_func:
            img_array = preprocess_func(img_array)
        data.append((img_array, label))
    return data

def get_data():
    if uploaded_class1_images and uploaded_class2_images:
        data_class1 = load_and_preprocess_images(class1_dir, 0)
        data_class2 = load_and_preprocess_images(class2_dir, 1)
    else:
        data_class1 = load_and_preprocess_images(sample_class1_dir, 0)
        data_class2 = load_and_preprocess_images(sample_class2_dir, 1)
    return data_class1 + data_class2

# Convert data to TensorFlow Dataset
def create_dataset(data):
    images, labels = zip(*data)
    images = np.array(images)
    labels = np.array(labels)
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)

data = get_data()
if data:
    dataset = create_dataset(data)
else:
    st.error("No images available for training.")

# Train the model
if st.button("Train Model"):
    if data:
        st.write("Training the model...")

        # Callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True))

        history = model.fit(dataset, epochs=epochs, callbacks=callbacks)
        st.write("Training complete.")

        # Plot training history
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history['loss'], label='Loss')
        ax[0].plot(history.history['accuracy'], label='Accuracy')
        ax[0].set_title("Training Loss and Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Value")
        ax[0].legend()

        st.pyplot(fig)
    else:
        st.error("No images available for training.")

# Clean up temporary directories
shutil.rmtree(class1_dir)
shutil.rmtree(class2_dir)
