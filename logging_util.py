import matplotlib.pyplot as plt
import io
import tensorflow as tf


def plot_orig_with_recon(original, reconstruction):
    fig = plt.figure(figsize=(4, 2))
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(tf.squeeze(original), cmap='gray')

    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(tf.squeeze(reconstruction), cmap='gray')

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


def log_reconstruction_comparison(sample_images, model, epoch, summary_writer, max_outputs=5):
    for sample_img_batch in sample_images:
        _, reconstruction_batch = model.compute_loss(sample_img_batch)
        comparison_list = [plot_to_image(plot_orig_with_recon(sample_img, reconstruction), add_batch_dim=False)
                           for sample_img, reconstruction in zip(sample_img_batch, reconstruction_batch)]
        comparison_batch = tf.stack(comparison_list, axis=0)
        with summary_writer.as_default():
            tf.summary.image('Reconstructions', comparison_batch, step=epoch, max_outputs=max_outputs)
