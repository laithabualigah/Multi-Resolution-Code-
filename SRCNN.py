import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from skimage import io, img_as_float
from skimage.transform import resize

# Data Pre-processing
def preprocess_image(image_path, upscale_factor=2):
    # Load and normalize the image
    image = img_as_float(io.imread(image_path))
    if len(image.shape) == 2:  # Convert grayscale to RGB
        image = np.stack((image,) * 3, axis=-1)

    # Downscale and then upscale the image to simulate low resolution
    height, width, _ = image.shape
    low_res = resize(image, (height // upscale_factor, width // upscale_factor), anti_aliasing=True)
    upscaled = resize(low_res, (height, width), anti_aliasing=True)

    return upscaled, image

# Define the SRCNN Model
def build_srcnn_model():
    model = models.Sequential([
        layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(None, None, 3)),
        layers.Conv2D(32, (1, 1), activation='relu', padding='same'),
        layers.Conv2D(3, (5, 5), activation='linear', padding='same')
    ])
    return model

# Train the SRCNN Model
def train_srcnn(model, train_data, train_labels, epochs=50, batch_size=16):
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=1)

# Test the SRCNN Model
def test_srcnn(model, input_image):
    input_tensor = np.expand_dims(input_image, axis=0)  # Add batch dimension
    output_image = model.predict(input_tensor)[0]  # Remove batch dimension
    return np.clip(output_image, 0, 1)  # Ensure pixel values are in [0, 1]

if __name__ == "__main__":
    # Example usage
    image_path = "example_image.jpg"  # Replace with your image path
    low_res, high_res = preprocess_image(image_path, upscale_factor=2)

    # Build and compile the SRCNN model
    srcnn_model = build_srcnn_model()

    # Simulated training data (for demonstration purposes)
    train_data = np.expand_dims(low_res, axis=0)
    train_labels = np.expand_dims(high_res, axis=0)

    # Train the model
    print("Training SRCNN...")
    train_srcnn(srcnn_model, train_data, train_labels, epochs=10, batch_size=1)

    # Test the model
    print("Testing SRCNN...")
    output_image = test_srcnn(srcnn_model, low_res)

    # Save or display the output
    io.imsave("output_image.jpg", (output_image * 255).astype(np.uint8))
    print("Output image saved as 'output_image.jpg'")
