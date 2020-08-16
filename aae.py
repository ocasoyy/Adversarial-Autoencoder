# Tensorflow2 Implementation of Adversarial Autoencoder
# Supervised Learning

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from time import perf_counter

# Data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = 60000
batch_size = 32
test_size = 10000

train_dataset = (tf.data.Dataset.from_tensor_slices(
    (tf.cast(train_images, tf.float32), tf.cast(train_labels, tf.float32)))
                 .shuffle(train_size).batch(batch_size))



class AAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(AAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28*28 + 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=512, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(units=256, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=latent_dim),
            ])

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim+1)),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(units=256, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(units=512, activation='relu'),
                tf.keras.layers.Dense(units=784),
            ])

        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim+1)),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(units=128, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(units=1),
            ])

    @tf.function
    def encode(self, x, y):
        inputs = tf.concat([x, y], 1)
        z = self.encoder(inputs)
        return z

    def discriminate(self, z, y):
        inputs = tf.concat([z, y], 1)
        output = self.discriminator(inputs)
        return output

    @tf.function
    def decode(self, z, y, apply_sigmoid=False):
        inputs = tf.concat([z, y], 1)
        logits = self.decoder(inputs)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def compute_reconstruction_loss(x, x_logit):
    # Reconstruction Loss
    marginal_likelihood = tf.reduce_sum(x * tf.math.log(x_logit) + (1 - x) * tf.math.log(1 - x_logit), axis=[1])
    loglikelihood = tf.reduce_mean(marginal_likelihood)
    reconstruction_loss = -loglikelihood
    return reconstruction_loss

def compute_discriminator_loss(fake_output, true_output):
    # Discriminator Loss
    d_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=true_output,
                                                                         labels=tf.ones_like(true_output)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                         labels=tf.zeros_like(fake_output)))
    discriminator_loss = d_loss_true + d_loss_fake
    return discriminator_loss

def compute_generator_loss(fake_output):
    # Generator Loss
    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                            labels=tf.ones_like(fake_output)))
    return generator_loss


@tf.function
def train_step(model, x, y, r_optimizer, d_optimizer, g_optimizer):
    # Results
    x = tf.reshape(x, [-1, 784])
    y = tf.reshape(y, [-1, 1])

    # Propagation
    with tf.GradientTape() as tape:
        z = model.encode(x, y)
        x_logit = model.decode(z, y, True)
        x_logit = tf.clip_by_value(x_logit, 1e-8, 1 - 1e-8)
        reconstruction_loss = compute_reconstruction_loss(x, x_logit)
    r_gradients = tape.gradient(reconstruction_loss, model.trainable_variables)
    r_optimizer.apply_gradients(zip(r_gradients, model.trainable_variables))

    with tf.GradientTape() as tape:
        z = model.encode(x, y)
        true_z = tf.random.normal(shape=(z.shape))
        fake_output = model.discriminate(z, y)
        true_output = model.discriminate(true_z, y)
        discriminator_loss = compute_discriminator_loss(fake_output, true_output)
    d_gradients = tape.gradient(discriminator_loss, model.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradients, model.trainable_variables))

    for _ in range(2):
        with tf.GradientTape() as tape:
            z = model.encode(x, y)
            fake_output = model.discriminate(z, y)
            generator_loss = compute_generator_loss(fake_output)
        g_gradients = tape.gradient(generator_loss, model.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, model.trainable_variables))

    total_loss = reconstruction_loss + discriminator_loss + generator_loss

    return total_loss


epochs = 30
latent_dim = 2
model = AAE(latent_dim)

r_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4/5)
g_optimizer = tf.keras.optimizers.Adam(1e-4)

# Train
for epoch in range(1, epochs + 1):
    train_losses = []
    for x, y in train_dataset:
        total_loss = train_step(model, x, y, r_optimizer, d_optimizer, g_optimizer)
        train_losses.append(total_loss)

    print('Epoch: {}, Loss: {:.2f}'.format(epoch, np.mean(train_losses)))


# Test
def generate_images(model, test_x, test_y):
    test_x = tf.reshape(test_x, [-1, 784])
    test_y = tf.reshape(test_y, [-1, 1])

    z = model.encode(test_x, test_y)
    predictions = model.decode(z, test_y, True)
    predictions = tf.clip_by_value(predictions, 1e-8, 1 - 1e-8)
    predictions = tf.reshape(predictions, [-1, 28, 28, 1])

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.show()



test_dataset = (tf.data.Dataset.from_tensor_slices(
    (tf.cast(test_images, tf.float32), tf.cast(test_labels, tf.float32)))
                .shuffle(test_size).batch(1500))


num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
test_x, test_y = next(iter(test_dataset))
test_x, test_y = test_x[0:num_examples_to_generate, :, :, :], test_y[0:num_examples_to_generate, ]

for i in range(test_x.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_x[i, :, :, 0], cmap='gray')
    plt.axis('off')

plt.show()

generate_images(model, test_x, test_y)


#
x, y = next(iter(test_dataset))
x = tf.reshape(x, [-1, 784])
y = tf.reshape(y, [-1, 1])
z = model.encode(x, y)

# plt.figure(figsize=(8, 6))
plt.scatter(z[:, 0], z[:, 1], c=np.max(y, 1), marker='o', cmap='Paired')
plt.colorbar()

