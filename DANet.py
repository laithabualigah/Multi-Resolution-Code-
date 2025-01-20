import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from skimage import io, img_as_float
from skimage.transform import resize

# Data Pre-processing
def preprocess_image(image_path, target_size=(128, 128)):
    """Load and resize an image to the target size."""
    image = img_as_float(io.imread(image_path))
    if len(image.shape) == 2:  # Convert grayscale to RGB
        image = np.stack((image,) * 3, axis=-1)
    image_resized = resize(image, target_size, anti_aliasing=True)
    return image_resized

# Define the Attention Modules
def position_attention_module(x):
    """Position Attention Module for spatial relationships."""
    shape = tf.shape(x)
    b, h, w, c = shape[0], shape[1], shape[2], shape[3]

    # Reshape for matrix multiplication
    x_reshaped = tf.reshape(x, (b, h * w, c))
    x_transposed = tf.transpose(x_reshaped, perm=[0, 2, 1])

    # Attention weights
    attention = tf.nn.softmax(tf.matmul(x_reshaped, x_transposed))

    # Weighted combination
    out = tf.matmul(attention, x_reshaped)
    out = tf.reshape(out, shape)
    return out

def channel_attention_module(x):
    """Channel Attention Module for feature channels."""
    avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)

    shared_dense = layers.Dense(x.shape[-1] // 8, activation='relu')
    avg_out = shared_dense(avg_pool)
    max_out = shared_dense(max_pool)

    combined = tf.nn.sigmoid(avg_out + max_out)
    return x * combined

# Define the DANet Model
def build_danet(input_shape=(128, 128, 3)):
    """Build the Dual Attention Network (DANet)."""
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Attention Modules
    position_attention = position_attention_module(x)
    channel_attention = channel_attention_module(x)

    # Combine attention outputs
    combined = layers.Add()([position_attention, channel_attention])

    # Final convolution layers
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(combined)
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs)
    return model

# Train the DANet Model
def train_danet(model, train_data, train_labels, epochs=50, batch_size=8):
    """Train the DANet model."""
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=1)

# Test the DANet Model
def test_danet(model, input_image):
    """Apply the trained DANet model to an input image."""
    input_tensor = np.expand_dims(input_image, axis=0)  # Add batch dimension
    output_image = model.predict(input_tensor)[0]  # Remove batch dimension
    return np.clip(output_image, 0, 1)  # Ensure pixel values are in [0, 1]

if __name__ == "__main__":
    # Example usage
    image_path = "example_image.jpg"  # Replace with your image path
    image = preprocess_image(image_path)

    # Simulated training data
    train_data = np.expand_dims(image, axis=0)
    train_labels = np.expand_dims(image, axis=0)

    # Build the DANet model
    danet_model = build_danet()

    # Train the model
    print("Training DANet...")
    train_danet(danet_model, train_data, train_labels, epochs=10, batch_size=1)

    # Test the model
    print("Testing DANet...")
    output_image = test_danet(danet_model, image)

    # Save or display the output
    io.imsave("danet_output.jpg", (output_image * 255).astype(np.uint8))
    print("Output image saved as 'danet_output.jpg'")
