import sys
import os
sys.path.append(sys.path[0])

from unet import get_model, get_unet
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import pandas as pd
import tensorflow_io as tfio
import cv2
import rasterio
import numpy as np
from process_img import *

def main() :
    img_size = (1024, 1024) # H x W
    num_classes = 4
    df = pd.read_csv("varuna_1-holdout.csv")
    weight_name=f"test-varuna_1"
    # train(
    #     df=df,
    #     img_size=img_size, 
    #     num_classes=num_classes,
    #     weight_name=weight_name,
    # )

    train_2()

def get_training(df, path, img_size, batch_size=32) :

    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    training = datagen.flow_from_dataframe(
        dataframe=df,
        directory=os.path.join(path, "input"),
        x_col="input",
        y_col="input",
        batch_size=batch_size,
        seed=42,
        target_size=img_size,
        class_mode=None
    )

    mask = datagen.flow_from_dataframe(
        dataframe=df,
        directory=os.path.join(path, "label"),
        x_col="label",
        y_col="label",
        batch_size=batch_size,
        seed=42,
        target_size=img_size,
        class_mode=None,
        color_mode="grayscale",
    )

    while True :
        x = training.next()
        # y = mask.next()

        m = mask.next()

        m = one_hot_encode_masks(m, 4)

        yield x, m

    # autotune = tf.data.experimental.AUTOTUNE
    # x  = get_filepath_from_df(df=df, path=os.path.join(path, "input"), x_col="input")
    # y  = get_filepath_from_df(df=df, path=os.path.join(path, "label"), x_col="label")

    # ds = tf.data.Dataset.from_tensor_slices((x, y))
    # ds = ds.map(
    #     prepare_data,
    #     num_parallel_calls=autotune
    # )
    # ds = ds.batch(batch_size)

    # # .prefetch(autotune)
    # # .repeat()
    # return ds

classes = {
    "casava" : (255, 0, 0),
    "rice" : (0, 255, 0),
    "maize" : (0, 0, 255),
    "sugarcane" : (0, 255, 255),
}

def one_hot_encode_masks(masks, num_classes):
    """
    :param masks: Y_train patched mask dataset 
    :param num_classes: number of classes
    :return: 
    """
    # initialise list for integer encoded masks
    integer_encoded_labels = []

    # # iterate over each mask
    for mask in masks:

        # get image shape
        _img_height, _img_width, _img_channels = mask.shape

        # create new mask of zeros
        encoded_image = np.zeros((_img_height, _img_width, 1)).astype(int)

        for j, cls in enumerate(classes):
            encoded_image[np.all(mask == classes[cls], axis=-1)] = j

        # append encoded image
        integer_encoded_labels.append(encoded_image)

    # return one-hot encoded labels
    return tf.keras.utils.to_categorical(y=integer_encoded_labels, num_classes=num_classes)

@tf.function
def wrapper_load(x, y):
    img_x, img_y = tf.py_function(
        func=prepare_data,
        inp=[x, y],
        Tout=[tf.float32, tf.float32]
    )

    return img_x, img_y

def prepare_data(x, y) :
    img_x = parse_image(x)
    img_y = parse_image(y, mask=True)

    return img_x, img_y
    # return tf.convert_to_tensor(img_x, dtype='float32'), tf.convert_to_tensor(img_y, dtype='float32')

def parse_image(x, mask=False) :
    img_str = tf.io.read_file(x)
    if mask :
        img = tf.image.decode_png(img_str, channels=1)
        img.set_shape([None, None, 1])
        img = tf.image.resize(images=img, size=[256, 256])
    else :
        img = tf.image.decode_png(img_str, channels=3)
        img.set_shape([None, None, 3])
        img = tf.image.resize(images=img, size=[256, 256])
    
    # img = tfio.experimental.image.decode_tiff(img_str)
    # x = x.numpy().decode("utf-8")
    # img = cv2.imread(x)
    # img = cv2.resize(img, (256, 256))

    # with rasterio.open(x) as src :
    #     img = src.read()
    #     img = np.moveaxis(img, 0, 2)

    # img = tf.convert_to_tensor(img, dtype='float32')

    return img

def get_filepath_from_df(df, path, x_col) :
    img_path = df[x_col].apply(lambda x: os.path.join(path, x)).values.tolist()
    return img_path


def train(df, img_size, num_classes, weight_name) :
    epochs=100
    batch_size=1
    
    dataset_path = os.path.join("datasets", "varuna_1")

    train_df = df[df["holdout"] == "training"]
    vali_df = df[df["holdout"] == "validation"]
    train_gen = get_training(train_df, path=dataset_path, img_size=img_size, batch_size=batch_size)
    vali_gen = get_training(vali_df, path=dataset_path, img_size=img_size, batch_size=batch_size)

    # model = get_model(
    #     img_size=img_size,
    #     num_classes=num_classes,
    # )

    print("Train Dataset:", train_gen)
    print("Val Dataset:", vali_gen)

    model = get_unet(
        input_size=img_size,
        num_classes=num_classes
    )

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
        loss="categorical_crossentropy",
        metrics=["accuracy", jaccard_index],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{weight_name}.h5", mode='max', monitor='val_jaccard_index')
    ]

    hist = model.fit(
        train_gen,
        validation_data=vali_gen,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=(33//batch_size),
        validation_steps=(9//batch_size)
    )

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1])
    union = K.sum(y_true, axis=[1]) + K.sum(y_pred, axis=[1])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def jaccard_index(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def train_2() :
    data_dir = os.path.join("datasets", "varuna_1")
    X, Y = get_training_data(
        root_directory=data_dir,
    )
    


if __name__ == "__main__" :
    main()