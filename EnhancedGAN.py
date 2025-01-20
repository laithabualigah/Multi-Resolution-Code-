import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from skimage import io, img_as_float
from skimage.util import random_noise
from skimage.transform import resize

# Data Pre-processing
def preprocess_image(image_path, target_size=(128, 128)):
    """Load and resize an image to the target size."""
    image = img_as_float(io.imread(image_path))
    if len(image.shape) == 2:  # Convert grayscale to RGB
        image = np.stack((image,) * 3, axis=-1)
    image_resized = resize(image, target_size, anti_aliasing=True)
    return image_resized

def add_noise(image, noise_type='gaussian', var=0.01):
    """Add noise to the image."""
    noisy_image = random_noise(image, mode=noise_type, var=var)
    return noisy_image

# Define the Generator Model
def build_generator():
    model = models.Sequential([
        layers.Dense(128 * 16 * 16, activation='relu', input_dim=100),
        layers.Reshape((16, 16, 128)),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])
    return model

# Define the Discriminator Model
def build_discriminator():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(128, 128, 3)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Combine Generator and Discriminator into a GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = models.Model(gan_input, gan_output)
    return gan

# Training the GAN
def train_gan(generator, discriminator, gan, real_images, epochs=10000, batch_size=32):
    """Train the Enhanced-GAN model."""
    for epoch in range(epochs):
        # Train Discriminator
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_images = generator.predict(noise)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        discriminator_loss_real = discriminator.train_on_batch(real_images, real_labels)
        discriminator_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        generator_loss = gan.train_on_batch(noise, real_labels)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Discriminator Loss = {discriminator_loss}, Generator Loss = {generator_loss}")

if __name__ == "__main__":
    # Example usage
    image_path = "example_image.jpg"  # Replace with your image path
    image = preprocess_image(image_path)
    noisy_image = add_noise(image)

    # Simulated training data
    real_images = np.expand_dims(noisy_image, axis=0)

    # Build Generator and Discriminator
    generator = build_generator()
    discriminator = build_discriminator()

    # Compile Discriminator
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Build and compile the GAN
    gan = build_gan(generator, discriminator)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    # Train the GAN
    print("Training Enhanced-GAN...")
    train_gan(generator, discriminator, gan, real_images, epochs=1000, batch_size=1)

    # Generate and save an enhanced image
    noise = np.random.normal(0, 1, (1, 100))
    enhanced_image = generator.predict(noise)[0]
    io.imsave("enhanced_image.jpg", (enhanced_image * 255).astype(np.uint8))
    print("Enhanced image saved as 'enhanced_image.jpg'")
