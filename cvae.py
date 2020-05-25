import tensorflow as tf
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
import IPython
from datetime import datetime
from IPython import display

from model import CVAE, reparameterize
from logging_util import plot_to_image, plot_orig_with_recon

# ----------------------------------------------------------------------------------------------------------------------
# PARAMS
# ----------------------------------------------------------------------------------------------------------------------
TRAIN_BUF = 60000
BATCH_SIZE = 32

TEST_BUF = 10000

optimizer = tf.keras.optimizers.Adam(1e-4)
epochs = 500
latent_dim = 10
num_examples_to_generate = 16

# ----------------------------------------------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------------------------------------------
# Load images
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# Normalizing the images to the range of [0., 1.]
train_images /= 255.
test_images /= 255.

# Binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

sample_imgs_ds = tf.data.Dataset.from_tensor_slices(test_images[:5]).batch(5)


# ----------------------------------------------------------------------------------------------------------------------
# LOSS METHODS
# ----------------------------------------------------------------------------------------------------------------------
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x), x_logit


@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss, reconstruction = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# ----------------------------------------------------------------------------------------------------------------------
# IMAGE METHODS
# ----------------------------------------------------------------------------------------------------------------------
generate = False


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for ii in range(predictions.shape[0]):
        plt.subplot(4, 4, ii+1)
        plt.imshow(predictions[ii, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    plt.close(fig)


def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


# ----------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------
logdir = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
train_writer = tf.summary.create_file_writer(logdir + '/train')
test_writer = tf.summary.create_file_writer(logdir + '/test')

# ----------------------------------------------------------------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------------------------------------------------------------
# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

# Create images at epoch 0, i.e. without any training
if generate:
    generate_and_save_images(model, 0, random_vector_for_generation)

for sample_img_batch in sample_imgs_ds:
    _, reconstruction_batch = compute_loss(model, sample_img_batch)
    # for sample_img, reconstruction in zip(sample_img_batch, reconstruction_batch):
    #     comparison = plot_to_image(plot_orig_with_recon(sample_img, reconstruction))
    comparison_list = [plot_to_image(plot_orig_with_recon(sample_img, reconstruction), add_batch_dim=False)
                       for sample_img, reconstruction in zip(sample_img_batch, reconstruction_batch)]
    comparison_batch = tf.stack(comparison_list, axis=0)
    with test_writer.as_default():
        tf.summary.image('Reconstructions', comparison_batch, step=0, max_outputs=5)

# Start training
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        compute_apply_gradients(model, train_x, optimizer)
    end_time = time.time()

    if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            total_loss, _ = compute_loss(model, test_x)
            loss(total_loss)
        elbo = -loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, '
              'time elapse for current epoch {}'.format(epoch,
                                                        elbo,
                                                        end_time - start_time))
        if generate:
            generate_and_save_images(model, epoch, random_vector_for_generation)

    if epoch % 10 == 0:
        for sample_img_batch in sample_imgs_ds:
            _, reconstruction_batch = compute_loss(model, sample_img_batch)
            # for sample_img, reconstruction in zip(sample_img_batch, reconstruction_batch):
            #     comparison = plot_to_image(plot_orig_with_recon(sample_img, reconstruction))
            comparison_list = [plot_to_image(plot_orig_with_recon(sample_img, reconstruction), add_batch_dim=False)
                               for sample_img, reconstruction in zip(sample_img_batch, reconstruction_batch)]
            comparison_batch = tf.stack(comparison_list, axis=0)
            with test_writer.as_default():
                tf.summary.image('Reconstructions', comparison_batch, step=epoch, max_outputs=5)

if generate:
    plt.imshow(display_image(epochs))
    plt.axis('off')  # Display images

    anim_file = 'cvae.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    if IPython.version_info >= (6, 2, 0, ''):
        display.Image(filename=anim_file)
