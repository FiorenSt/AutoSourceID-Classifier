import tensorflow as tf

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 1

inputs_img = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
inputs_location = tf.keras.layers.Input((2,))


config = {'batch_size': 2048, 'model_name': 'cnn+location FINAL', 'epochs': 100, 'init_learning_rate': 0.001,
          'lr_decay_rate': 0.1, 'optimizer': 'adam', 'loss_fn': 'focal_loss', 'gamma': 0, 'alpha': 1, 'metrics': [
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC(curve='ROC'),
        tf.keras.metrics.AUC(curve='PR')
    ], 'earlystopping_patience': 10, 'kernel_size': 4, 'padding': 'same', 'dropout': 0, 'regularization': 0}



def create_model(inputs_img, inputs_location, config):
    """
    Returns a Tensorflow model based on the configuration parameters.

    Parameters:
    inputs_img: Tensor, the input tensor for the images.
    inputs_location: Tensor, the input tensor for the locations.
    config (dict): A dictionary containing the configuration parameters for the model.

    Returns:
    model: A Tensorflow model.
    """
    x = inputs_img
    for _ in range(3):
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(config['kernel_size'], config['kernel_size']), padding=config['padding'], activation='relu', kernel_regularizer =tf.keras.regularizers.l2(config['regularization']))(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Dropout(config['dropout'])(x)

    x = tf.keras.layers.Flatten()(x)

    y = inputs_location
    for _ in range(2):
        y = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer =tf.keras.regularizers.l2(config['regularization']))(y)

    combined = tf.keras.layers.Concatenate()([x, y])
    z = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer =tf.keras.regularizers.l2(config['regularization']))(combined)
    z = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer =tf.keras.regularizers.l2(config['regularization']))(z)
    z = tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.Model(inputs=[inputs_img, inputs_location], outputs=z)
    return model

def train_model(train_dataset, valid_dataset, total_train, total_val, config):
    """
    Trains a Tensorflow model based on the given parameters.

    Parameters:
    train_dataset: Tensor, the training dataset.
    valid_dataset: Tensor, the validation dataset.
    total_train: Integer, the total number of training samples.
    total_val: Integer, the total number of validation samples.
    config (dict): A dictionary containing the configuration parameters for the model.
    """
    model = create_model(inputs_img, inputs_location, config)

    opt = tf.keras.optimizers.Adam(learning_rate=config['init_learning_rate'])

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=config['metrics'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=config['earlystopping_patience'], verbose=0, mode='auto',
        restore_best_weights=True
    )

    model.fit(x=train_dataset,
              epochs=config['epochs'],
              steps_per_epoch=total_train // config['batch_size'],
              validation_data=valid_dataset,
              validation_steps=total_val // config['batch_size'],
              callbacks=[early_stop],
              verbose=1)

    model.save(config['model_name'])
