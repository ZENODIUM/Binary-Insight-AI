import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

st.title(":blue[Binary Insight AI] Neural Network Binary Classification")
st.markdown('''
:blue[Functionality:]

:blue-background[Image Upload]: Allows users to upload images for two classes (Class 1 and Class 2).

:blue-background[Sample Images:] Displays sample images from each class if no images are uploaded.

:blue-background[Model Configuration:] Customizable CNN parameters through sidebar sliders and selectors, including:
Number of convolutional layers
Filters in each convolutional layer
Kernel size and pool size
Number of dense layers and neurons
Learning rate and optimizer choice
Option for early stopping

:blue-background[Model Training:]
Train the model using uploaded images or a pre-defined sample dataset.
Real-time training metrics and architecture visualizations.
Progress bar and metric plots for accuracy and loss during training.

:blue-background[Model Saving & Download:] Option to save and download the trained model as a .h5 file.

This setup offers a user-friendly interface to build, train, and evaluate a binary image classification model.
''')




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
st.sidebar.title(":blue[Binary Insight] AI")
st.sidebar.image("robot_image.png")
st.sidebar.title("Model Parameters")
num_conv_layers = st.sidebar.slider("Number of Convolutional Layers", 1, 5, 2)
conv_filters = [st.sidebar.slider(f"Filters in Conv Layer {i+1}", 16, 128, 32*(i+1)) for i in range(num_conv_layers)]
kernel_size = st.sidebar.slider("Kernel Size", 2, 5, 3)
pool_size = st.sidebar.slider("Pool Size", 2, 3, 2)
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
model.summary(print_fn=lambda x: st.text(x))

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

# Prepare dataset
def prepare_dataset():
    class1_data = load_and_preprocess_images(class1_dir, 0)
    class2_data = load_and_preprocess_images(class2_dir, 1)

    data = class1_data + class2_data
    np.random.shuffle(data)
    X, y = zip(*data)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Initialize DataFrame to store metrics
metrics_df = pd.DataFrame(columns=['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss'])

# Placeholders for updating the metrics
table_placeholder = st.empty()
chart_placeholder = st.empty()

# Callback to update metrics and architecture image
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics_df.loc[epoch] = [epoch, logs.get('accuracy'), logs.get('loss'), logs.get('val_accuracy'), logs.get('val_loss')]
        table_placeholder.write(metrics_df)
        
        # Plot the metrics
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        ax[0, 0].plot(metrics_df['epoch'], metrics_df['accuracy'], label='Training Accuracy')
        ax[0, 0].set_title('Training Accuracy')
        ax[0, 0].set_xlabel('Epoch')
        ax[0, 0].set_ylabel('Accuracy')
        
        ax[0, 1].plot(metrics_df['epoch'], metrics_df['loss'], label='Training Loss')
        ax[0, 1].set_title('Training Loss')
        ax[0, 1].set_xlabel('Epoch')
        ax[0, 1].set_ylabel('Loss')
        
        ax[1, 0].plot(metrics_df['epoch'], metrics_df['val_accuracy'], label='Validation Accuracy')
        ax[1, 0].set_title('Validation Accuracy')
        ax[1, 0].set_xlabel('Epoch')
        ax[1, 0].set_ylabel('Accuracy')
        
        ax[1, 1].plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss')
        ax[1, 1].set_title('Validation Loss')
        ax[1, 1].set_xlabel('Epoch')
        ax[1, 1].set_ylabel('Loss')

        plt.tight_layout()
        chart_placeholder.pyplot(fig)
        
        # Update the architecture image
        update_architecture_image()

# Progress bar for training
progress_placeholder = st.empty()

# Train the model with manually uploaded images
if st.button("Train Model with Manually Uploaded Images"):
    if not uploaded_class1_images or not uploaded_class2_images:
        st.warning("Please upload images for both classes.")
    else:
        with st.spinner('Training...'):
            X, y = prepare_dataset()
            datagen = ImageDataGenerator(validation_split=0.2)
            train_generator = datagen.flow(X, y, batch_size=batch_size, subset='training')
            val_generator = datagen.flow(X, y, batch_size=batch_size, subset='validation')

            callbacks = [MetricsCallback()]
            if early_stopping:
                callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True))

            class ProgressBarCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    percent_complete = int((epoch + 1) / epochs * 100)
                    progress_placeholder.progress(percent_complete, text=f'Epoch {epoch + 1}/{epochs} complete')

            history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, 
                                callbacks=callbacks + [ProgressBarCallback()])
            
            st.success("Model trained successfully!")
            
            # Display accuracy
            train_accuracy = history.history['accuracy'][-1]
            val_accuracy = history.history['val_accuracy'][-1]
            st.write(f"Training Accuracy: {train_accuracy:.2f}")
            st.write(f"Validation Accuracy: {val_accuracy:.2f}")

# Train the model with sample dataset
if st.button("Upload Sample Brain MRI Image Dataset"):
    class1_data = load_and_preprocess_images(sample_class1_dir, 0)
    class2_data = load_and_preprocess_images(sample_class2_dir, 1)

    data = class1_data + class2_data
    np.random.shuffle(data)
    X, y = zip(*data)
    X = np.array(X)
    y = np.array(y)

    with st.spinner('Training...'):
        datagen = ImageDataGenerator(validation_split=0.2)
        train_generator = datagen.flow(X, y, batch_size=batch_size, subset='training')
        val_generator = datagen.flow(X, y, batch_size=batch_size, subset='validation')

        callbacks = [MetricsCallback()]
        if early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True))

        class ProgressBarCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                percent_complete = int((epoch + 1) / epochs * 100)
                progress_placeholder.progress(percent_complete, text=f'Epoch {epoch + 1}/{epochs} complete')

        history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, 
                            callbacks=callbacks + [ProgressBarCallback()])
        
        st.success("Model trained successfully!")
        
        # Display accuracy
        train_accuracy = history.history['accuracy'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        st.write(f"Training Accuracy: {train_accuracy:.2f}")
        st.write(f"Validation Accuracy: {val_accuracy:.2f}")

# Save and download model
if st.button("Save Model"):
    model.save('model.h5')
    st.success("Model saved successfully!")

# Download link for the saved model
if os.path.exists('model.h5'):
    with open('model.h5', 'rb') as f:
        st.download_button("Download Model", f, file_name="model.h5")
