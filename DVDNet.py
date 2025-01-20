import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from skimage import img_as_float, io
from skimage.util import random_noise
from skimage.restoration import denoise_wavelet

# Data Pre-processing
def add_noise(image, noise_type='gaussian', var=0.01):
    """Add noise to the image."""
    noisy_image = random_noise(image, mode=noise_type, var=var)
    return noisy_image

def preprocess_video_frame(frame, target_size=(128, 128)):
    """Resize and normalize the video frame."""
    frame = img_as_float(io.imread(frame))
    if frame.shape[0:2] != target_size:
        frame = tf.image.resize(frame, target_size).numpy()
    return frame

# Define the DVDNet Model
def build_dvdnet_model(input_shape=(128, 128, 3)):
    """Builds the DVDNet model for video denoising."""
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Denoising block
    for _ in range(5):
        skip = x
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation=None, padding='same')(x)
        x = layers.Add()([skip, x])  # Residual connection
        x = layers.ReLU()(x)

    # Output layer
    outputs = layers.Conv2D(3, (3, 3), activation='linear', padding='same')(x)

    model = models.Model(inputs, outputs)
    return model

# Train the DVDNet Model
def train_dvdnet(model, train_data, train_labels, epochs=10, batch_size=4):
    """Train the DVDNet model."""
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=1)

# Test the DVDNet Model
def test_dvdnet(model, noisy_frame):
    """Apply the trained DVDNet model to a noisy video frame."""
    input_tensor = np.expand_dims(noisy_frame, axis=0)  # Add batch dimension
    denoised_frame = model.predict(input_tensor)[0]  # Remove batch dimension
    return np.clip(denoised_frame, 0, 1)  # Ensure pixel values are in [0, 1]

if __name__ == "__main__":
    # Example usage
    frame_path = "example_frame.jpg"  # Replace with your video frame path
    frame = preprocess_video_frame(frame_path)

    # Add noise to the frame
    noisy_frame = add_noise(frame)

    # Build and compile the DVDNet model
    dvdnet_model = build_dvdnet_model()

    # Simulated training data (for demonstration purposes)
    train_data = np.expand_dims(noisy_frame, axis=0)
    train_labels = np.expand_dims(frame, axis=0)

    # Train the model
    print("Training DVDNet...")
    train_dvdnet(dvdnet_model, train_data, train_labels, epochs=10, batch_size=1)

    # Test the model
    print("Testing DVDNet...")
    denoised_frame = test_dvdnet(dvdnet_model, noisy_frame)

    # Save or display the output
    io.imsave("denoised_frame.jpg", (denoised_frame * 255).astype(np.uint8))
    print("Denoised frame saved as 'denoised_frame.jpg'")
