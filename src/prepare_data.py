import tensorflow as tf
import glob

###################################
# Prepare Data
###################################

def _parse_function(example_proto):
    """Parses a single tf.Example into image, position, and label tensors."""
    keys_to_features = {'image': tf.io.FixedLenFeature((34,34), tf.float32),
                        'location': tf.io.FixedLenFeature((2,), tf.float32),
                        'label': tf.io.FixedLenFeature((), tf.int64)}
    parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
    return (parsed_features['image'],  parsed_features['location']), parsed_features['label']

def _augment(image, values):
    """Applies random cropping to the images in the batch."""
    image, position = image
    image = tf.reshape(image,(image.shape[0],image.shape[1],1))
    height = tf.random.uniform(shape=(1,),minval=-1,maxval=2,dtype=tf.int32)
    width = tf.random.uniform(shape=(1,), minval=-1, maxval=2, dtype=tf.int32)
    new_image = tf.image.crop_to_bounding_box(image,(height+1)[0],(width+1)[0],32,32)
    return (new_image, position), values

def _only_crop(image, values):
    """Crops the images in the batch to a fixed size."""
    image, position = image
    image = tf.reshape(image,(image.shape[0],image.shape[1],1))
    new_image = tf.image.crop_to_bounding_box(image,1,1,32,32)
    return (new_image, position), values

def get_dataset(filenames, batch_size=1024, augment=False):
    """Returns a tf.data.Dataset given TFRecord filenames."""
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(_parse_function, num_parallel_calls=AUTOTUNE)
        .map(_augment if augment else _only_crop, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .repeat()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset

def get_tfrecord_filenames(path):
    """Returns a list of filenames in the given directory."""
    return glob.glob(path)

def prepare_data(path_to_tfrecords, batch_size=1024, augment_train=False):
    """Prepares datasets for training and validation."""
    # Replace with your directory structure
    train_files = get_tfrecord_filenames(f"{path_to_tfrecords}/Training/*.*")
    val_files = get_tfrecord_filenames(f"{path_to_tfrecords}/Validation/*.*")
    # Add code for test and calibration datasets if needed

    train_dataset = get_dataset(train_files, batch_size, augment_train)
    val_dataset = get_dataset(val_files, batch_size)

    return train_dataset, val_dataset

def load_test_data(path_to_tfrecords, batch_size=1024):
    """Loads the test data from TFRecords.

    Parameters:
    path_to_tfrecords (str): The path to the directory containing the TFRecords.
    batch_size (int): The size of the batches.

    Returns:
    test_dataset: A tf.data.Dataset object containing the test data.
    """
    test_files = get_tfrecord_filenames(f"{path_to_tfrecords}/Test/*.*")
    test_dataset = get_dataset(test_files, batch_size)

    return test_dataset

# Usage:
# train_dataset, val_dataset = prepare_data("/path/to/TFRecord/datasets", 10, True)
