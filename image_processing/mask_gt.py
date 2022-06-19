from tqdm import tqdm
from enum import Enum
import numpy as np

import tensorflow as tf

def main() :
    pass

# mask color codes
class MaskColorMap(Enum):
    # Unlabelled = (155, 155, 155)
    # Building = (60, 16, 152)
    # Land = (132, 41, 246)
    # Road = (110, 193, 228)
    # Vegetation = (254, 221, 58)
    # Water = (226, 169, 41)

    casava = (255, 0, 0)
    rice = (0, 255, 0)
    maize = (0, 0, 255)
    sugarcane = (0, 255, 255)


def one_hot_encode_masks(masks, num_classes):
    """
    :param masks: Y_train patched mask dataset 
    :param num_classes: number of classes
    :return: 
    """
    # initialise list for integer encoded masks
    integer_encoded_labels = []

    # iterate over each mask
    for mask in tqdm(masks):

        # get image shape
        _img_height, _img_width, _img_channels = mask.shape

        # create new mask of zeros
        encoded_image = np.zeros((_img_height, _img_width, 1)).astype(int)

        for j, cls in enumerate(MaskColorMap):
            encoded_image[np.all(mask == cls.value, axis=-1)] = j

        # append encoded image
        integer_encoded_labels.append(encoded_image)

    # return one-hot encoded labels
    return tf.keras.utils.to_categorical(y=integer_encoded_labels, num_classes=num_classes)

if __name__ == "__main__" :
    main()