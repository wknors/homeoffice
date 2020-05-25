import matplotlib.pyplot as plt
import numpy as np
import io
import tensorflow as tf


def plot_orig_with_recon(original, reconstruction):
    fig = plt.figure(figsize=(4, 2))
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.squeeze(original), cmap=plt.cm.binary)

    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.squeeze(reconstruction), cmap=plt.cm.binary)

    return fig


def plot_to_image(figure, add_batch_dim=True):
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
    if add_batch_dim:
        image = tf.expand_dims(image, 0)
    return image
