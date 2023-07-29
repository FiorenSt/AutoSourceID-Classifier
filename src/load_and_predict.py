import os
from tensorflow.keras.models import load_model
import tensorflow as tf

def load_config():
    """
    Returns a dictionary of configuration parameters for a machine learning model.

    Returns:
    config (dict): A dictionary containing the following key-value pairs:
        - 'batch_size': The number of samples that will be propagated through the network at a time.
        - 'model_name': The name of the model to be used.
        - 'epochs': The number of times the learning algorithm will work through the entire training dataset.
        - 'init_learning_rate': The initial learning rate for the optimizer.
        - 'lr_decay_rate': The learning rate decay rate.
        - 'optimizer': The optimization algorithm to be used.
        - 'loss_fn': The loss function to be used.
        - 'gamma', 'alpha': Parameters for the focal loss function.
        - 'metrics': The metrics to be used for evaluating the model's performance.
        - 'earlystopping_patience': The number of epochs with no improvement after which training will be stopped.
        - 'kernel_size': The size of the kernel to be used in the convolutional neural network.
        - 'padding': The type of padding to be used in the convolutional layers.
        - 'dropout': The dropout rate for regularization.
        - 'regularization': The regularization constant.
    """
    return dict(
        batch_size=2048,
        model_name='cnn+location FINAL',
        epochs=100,
        init_learning_rate=0.001,
        lr_decay_rate=0.1,
        optimizer='adam',
        loss_fn='focal_loss',
        gamma=0,
        alpha=1,
        metrics=[
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(curve='ROC'),
            tf.keras.metrics.AUC(curve='PR')
        ],
        earlystopping_patience=10,
        kernel_size=4,
        padding='same',
        dropout=0,
        regularization=0
    )



def load_and_predict(model_name, patches):
    """
    Loads a Keras model and uses it to make predictions.

    Parameters:
    model_name (str): The name of the model file (assumed to be in the '../Models' directory).
    patches (array-like): The input data to make predictions on.

    Returns:
    array: The predictions made by the model.
    """

    model_path = os.path.join('../Models', model_name)

    # Check if the model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")

    # Load the model
    model = load_model(model_path)

    # Make predictions
    try:
        predictions = model.predict(patches)
    except Exception as e:
        raise ValueError(f"An error occurred while making predictions: {e}")

    return predictions
