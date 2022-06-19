import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow import keras
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import *
import random
# from tensorflow.keras import layers
seed=7
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def main() :
    # model = get_model(input_size=(128, ))

    x, y = get_data()
    input_size = len(x.columns)
    x = np.array(x)
    y = np.array(y)
    y = y-1
    input_size = (input_size, )

    class_weight = {
        0: 510,
        1: 240,
        2: 433,
        3: 134,
    }

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    # print(data)

    min_max_scaler = MinMaxScaler()
    scaler = StandardScaler()
    # x_train = min_max_scaler.fit_transform(x_train)
    x_train = scaler.fit_transform(x_train)

    model = get_model(input_size)
    model.compile(
        tf.optimizers.Adam(learning_rate=1e-4), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy()]
    )
    hist = model.fit(
        x_train, 
        y_train,
        epochs=1000,
        batch_size=1024,
        # class_weight=class_weight,
    )

    # x_test = min_max_scaler.transform(x_test)
    x_test = scaler.transform(x_test)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    get_score(y_test, y_pred)

    path = os.path.join("model_test1.h5")
    model.save(path)
    pass

def get_score(y_test, y_pred, average=None) :
    print(classification_report(y_test, y_pred))
    print(jaccard_score(y_test, y_pred, average=None))
    print(np.mean(jaccard_score(y_test, y_pred, average=None)))
    print(np.std(jaccard_score(y_test, y_pred, average=None)))

def get_model(input_size) :
    inputs = keras.Input(input_size)

    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    # x = Dense(256, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    # x = Dense(128, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dense(32, activation='relu')(x)
    # x = Dropout(0.3)(x)
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.25)(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(128, activation='sigmoid')(x)
    # x = Dropout(0.25)(x)
    # x = Dense(64, activation='relu')(x)
    
    # x = Conv1D(32, 3, activation='relu')(inputs)

    outputs = Dense(4, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    print(model.summary())
    return model

def f1(y_true, y_pred) :
    return f1_score(y_true, y_pred, average=None)

def get_data() :
    path = os.path.join("train_non_pred_NDVI_misra_b5I.csv")

    df = pd.read_csv(path)

    lst_col = []
    # for c in df.columns :
    #     if "years" != c or "crop_type" != c :
    #         lst_col.append(c)
    x = df[lst_col]
    x = df.drop(["crop_type", "years"], axis=1)
    y = df["crop_type"]
    return x, y

if __name__ == "__main__" :
    main()