import tensorflow as tf
import time
import matplotlib.pyplot as plt
import PIL
from datetime import datetime
from IPython import display

from model import CVAE
from logging_util import log_reconstruction_comparison

# ----------------------------------------------------------------------------------------------------------------------
# PARAMS
# ----------------------------------------------------------------------------------------------------------------------
TRAIN_BUF = 60000
BATCH_SIZE = 32

TEST_BUF = 10000

optimizer = tf.keras.optimizers.Adam(1e-4)
epochs = 50
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

# Split the formerly training set into training and test set
train_val_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF)
train_dataset = train_val_dataset.take(int(train_images.shape[0])*0.8).batch(BATCH_SIZE)
valid_dataset = train_val_dataset.skip(int(train_images.shape[0])*0.8).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

# Create a dataset of the first 5 images from the test set for visualizing the progress in reconstruction.
sample_imgs_ds = tf.data.Dataset.from_tensor_slices(test_images[:5]).batch(5)


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
valid_writer = tf.summary.create_file_writer(logdir + '/valid')
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

# First comparison of original and reconstructions before training begins
log_reconstruction_comparison(
            sample_images=sample_imgs_ds, model=model, epoch=0, summary_writer=test_writer
        )

# Start training
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        model.compute_apply_gradients(train_x, optimizer)
    end_time = time.time()

    if epoch % 10 == 0:
        loss = tf.keras.metrics.Mean()
        for valid_x in valid_dataset:
            total_loss, _ = model.compute_loss(valid_x)
            loss(total_loss)
        elbo = -loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Valid set ELBO: {}, '
              'time elapse for current epoch {}'.format(epoch, elbo, end_time - start_time))

        log_reconstruction_comparison(
            sample_images=sample_imgs_ds, model=model, epoch=epoch, summary_writer=test_writer
        )

        if generate:
            generate_and_save_images(model, epoch, random_vector_for_generation)

if generate:
    plt.imshow(display_image(epochs))
    plt.axis('off')  # Display images

# ToDo: Tensorboard hyperparameter tuning possible? Need to use model.fit + model.evaluate or custom ok?
