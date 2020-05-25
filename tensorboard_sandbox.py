from datetime import datetime
from pathlib import Path
import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Download the data. The data is already divided into train and test.
# The labels are integers representing classes.
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()

# Names of the integer classes, i.e., 0 -> T-short/top, 1 -> Trouser, etc.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Sets up a timestamped log directory.
logdir = Path("logs/train_data/", datetime.now().strftime("%Y%m%d-%H%M%S"))
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(str(logdir))

with file_writer.as_default():
    # Don't forget to reshape.
    images = np.reshape(train_images[0:25], (-1, 28, 28, 1))
    tf.summary.image("25 training data examples", images, max_outputs=25, step=0)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid():
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title=class_names[train_labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)

    return figure


def plot_orig_with_recon(original, reconstruction):
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.squeeze(original), cmap=plt.cm.binary)

    plt.subplot(2, 1, 2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.squeeze(reconstruction), cmap=plt.cm.binary)

    return fig


# Prepare the plot
figure = image_grid()
# Convert to image and log
with file_writer.as_default():
    tf.summary.image("Training data", plot_to_image(figure), step=0)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
