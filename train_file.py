import sys
import os
print(sys.path)
sys.path.append(sys.path[0])

from unet import get_model, get_unet
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import tensorflow_io as tfio
import cv2
import rasterio
import numpy as np

def main() :
    img_size = (512, 512) # H x W
    num_classes = 4
    df = pd.read_csv("varuna_1.csv")
    weight_name=f"test-varuna_1"
    train(
        df=df,
        img_size=img_size, 
        num_classes=num_classes,
        weight_name=weight_name,
    )

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
        class_mode='raw'
    )

    mask_training = datagen.flow_from_dataframe(
        dataframe=df,
        directory=os.path.join(path, "label"),
        x_col="label",
        y_col="label",
        batch_size=batch_size,
        seed=42,
        target_size=img_size,
        class_mode='raw'
    )

    autotune = tf.data.experimental.AUTOTUNE
    x  = get_filepath_from_df(df=df, path=os.path.join(path, "input"), x_col="input")
    y  = get_filepath_from_df(df=df, path=os.path.join(path, "label"), x_col="label")

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = (
        ds.map(
            wrapper_load,
            num_parallel_calls=autotune
        )
    # .repeat()
    .batch(batch_size)
    .prefetch(autotune)
    )
    return ds

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
    img_y = parse_image(y)

    return tf.convert_to_tensor(img_x, dtype='float32'), tf.convert_to_tensor(img_y, dtype='float32')

def parse_image(x) :
    # img_str = tf.io.read_file(x)
    # img = tfio.experimental.image.decode_tiff(img_str)
    x = x.numpy().decode("utf-8")
    img = cv2.imread(x)
    img = cv2.resize(img, (512, 512))

    # with rasterio.open(x) as src :
    #     img = src.read()
    #     img = np.moveaxis(img, 0, 2)

    img = tf.convert_to_tensor(img, dtype='float32')
    

    return img

def get_filepath_from_df(df, path, x_col) :
    imgs = []
    img_path = df[x_col].apply(lambda x: os.path.join(path, x)).values.tolist()

    return img_path


def train(df, img_size, num_classes, weight_name) :
    epochs=100
    dataset_path = os.path.join("datasets", "varuna_1")
    # model = get_model(
    #     img_size=img_size,
    #     num_classes=num_classes,
    # )

    model = get_unet(
        input_size=img_size,
        num_classes=num_classes
    )

    model.compile(
        optimizer="adam", 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy'],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{weight_name}.h5", save_best_only=True)
    ]

    train_gen = get_training(df, path=dataset_path, img_size=img_size, batch_size=1)
    # mask_train_gen = get_training(df, path=dataset_path, img_size=img_size)
    # validation_gen = get_training(validaiton_df)

    hist = model.fit(
        train_gen,
        # validation_data=validation_gen,
        epochs=epochs,
        callbacks=callbacks,
    )

if __name__ == "__main__" :
    main()