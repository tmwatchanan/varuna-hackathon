import cv2
from skimage.io import *
from PIL import Image
import os
import rasterio
# import rasterio.mask
import geopandas as gpd

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, mapping
import shutil
# from rasterio import plot

def main() :
    # create_gt()
    # pull_tci_from_sentinel()
    move_file()
    pass

def create_gt() :

    path = os.path.join("20210101-RGB_masked.tiff")

    img = imread(path)

    mask_img = np.zeros(img.shape).astype('float32')
    print(img.shape)

    df = gpd.read_file("traindata.shp")

    print(df.head())

    color = {
        "1": [255, 0, 0],
        "2": [0, 255, 0],
        "3": [0, 0, 255],
        "4": [0, 255, 255],
    }

    for c in color :
        label = df[df["crop_type"] == c]
        label["coor"] = label["geometry"].apply(lambda x: mapping(x)["coordinates"][0])

        lst_label = label["coor"].values.tolist()
        # aa = []
        # for i, item in enumerate(lst_label) :
        #     bb = []
        #     for j, jtem in enumerate(lst_label[i]) :
        #         bb.append(jtem)
        #         # break
        #     aa.append(bb)
        # print(aa)

        geo = np.array(lst_label)
        # print(geo)
        # geo = np.array(label["coor"].values.tolist())
        cv2.fillPoly(mask_img, pts=[geo], color=color[c])

    imshow(mask_img)
    plt.show()

def pull_tci_from_sentinel() :
    path = os.path.join("sentinel-2 image", "2021")
    dst_path = os.path.join("sentinel-2 image", "2021", "rgb")

    for date in os.listdir(path) :
        if "rgb" == date :
            continue
        dst_date_path = os.path.join(dst_path, date)
        os.makedirs(dst_date_path, exist_ok=True)

        img_data_path = os.path.join(path, date, "IMG_DATA")
        for f in os.listdir(img_data_path) :
            if (".jp2" in f and "TCI" in f ) and (".xml" not in f):
                file_path =os.path.join(img_data_path, f)
                bands = rasterio.open(file_path).read()
                img = np.moveaxis(bands, 0, 2)
                new_file = f.replace(".jp2", ".png")
                imsave(os.path.join(dst_date_path, new_file), img)
                # break
        
        # break

def move_file() :
    file_path = os.path.join("sentinel-2 image", "2021", "rgb")
    dst_path = os.path.join("sentinel-2 image", "2021", "all_date")
    os.makedirs(dst_path, exist_ok=True)
    for date in os.listdir(file_path) :
        date_path = os.path.join(file_path, date)

        for f in os.listdir(date_path) :
            shutil.copyfile(os.path.join(date_path, f), os.path.join(dst_path, f))


if __name__ == "__main__" :
    main()