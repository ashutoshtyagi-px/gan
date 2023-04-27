import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import time

# Define the generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

# Define the discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
# Define the loss functions for the generator and discriminator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define the optimizers for the generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define a function to generate and save images during training
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
# Define the training loop
@tf.function
def train_step(images):
    # Generate random noise for the generator input
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    # Use the gradient tape to record operations for automatic differentiation
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate images from noise using the generator
        generated_images = generator(noise, training=True)

        # Evaluate the discriminator on real and fake images
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calculate the loss for the generator and discriminator
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Compute the gradients of the generator and discriminator loss functions
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply the gradients to the optimizer
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Train the GAN model
def train_gan(generator, discriminator, dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

        # Generate and save images every 10 epochs
        if epoch % 10 == 0:
            generate_and_save_images(generator, epoch, random_noise)

        # Print the loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Generator loss={gen_loss}, Discriminator loss={disc_loss}')

    # Generate and save a final set of images
    generate_and_save_images(generator, epochs, random_noise)

    # Generator model
    generator = build_generator(image_shape)
    generator.summary()

    # Discriminator model
    discriminator = build_discriminator(image_shape)
    discriminator.summary()

    # Compile the discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Set the discriminator to non-trainable
    discriminator.trainable = False

    # Define the GAN model
    gan_input = Input(shape=(latent_dim,))
    gan_output = generator(gan_input)
    gan_output_logits = discriminator(gan_output)
    gan = Model(gan_input, gan_output_logits)

    # Compile the GAN model
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    
