import glob
import tensorflow as tf

def load_config_partI():
    return dict(
        batch_size = 2048,
        model_name = 'Part I',
        epochs = 200,
        init_learning_rate = 0.001,
        lr_decay_rate = 0.1,
        optimizer = 'adam',
        loss_fn = 'huber',
        metrics=['mse', 'mae', 'mre'],
        earlystopping_patience = 10,
        kernel_size=4,
        padding='same',
        dropout=0
    )


def load_config_partII():
    return dict(
        batch_size=2048,
        model_name='Part II ',
        epochs=200,
        init_learning_rate=0.001,
        lr_decay_rate=0.1,
        optimizer='adam',
        loss_fn='gauss_loss',
        metrics=['mse', 'mae', 'mre'],
        earlystopping_patience=8,
        kernel_size=4,
        padding='same',
        dropout=0,
        freezing='False',
        regularization=0.001
    )



def prepare_data(config):
    # Implement your data preparation logic here
    pass