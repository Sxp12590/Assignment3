import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load the MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((-1, 784))  # Flatten the images
x_test = x_test.reshape((-1, 784))    # Flatten the images

# 2. Define the Autoencoder (Fully Connected)
def build_autoencoder(latent_dim=32):
    # Encoder
    input_img = layers.Input(shape=(784,))
    encoded = layers.Dense(latent_dim, activation='relu')(input_img)

    # Decoder
    decoded = layers.Dense(784, activation='sigmoid')(encoded)

    # Autoencoder Model
    autoencoder = models.Model(input_img, decoded)

    # Encoder Model (to extract encoded data)
    encoder = models.Model(input_img, encoded)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder, encoder

# 3. Compile and train the autoencoder
latent_dim = 32  
autoencoder, encoder = build_autoencoder(latent_dim=latent_dim)

# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_data=(x_test, x_test))

# 4. Plot original vs reconstructed images
def plot_comparison(original, reconstructed, num_images=10):
    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        # Display original
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap="gray")
        ax.axis('off')

        # Display reconstruction
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        ax.axis('off')

    plt.show()

# Reconstruct images
reconstructed_images = autoencoder.predict(x_test)

# Plot comparison
plot_comparison(x_test, reconstructed_images)

# 5. Modify latent dimension and evaluate
latent_dims = [16, 64]
for dim in latent_dims:
    print(f"\nTraining autoencoder with latent dimension {dim}")
    autoencoder, encoder = build_autoencoder(latent_dim=dim)
    autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_data=(x_test, x_test))
    reconstructed_images = autoencoder.predict(x_test)
    plot_comparison(x_test, reconstructed_images, num_images=10)
