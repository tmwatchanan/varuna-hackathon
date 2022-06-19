import geopandas as gpd
import rasterio
import os

def main() :
    test()
    pass

def savi(img_b4, img_b8) :
    res = (img_b8 - img_b4) / (img_b8 + img_b4 + 0.428) * (1.428)
    return res

def ndwi(img_b3, img_b8) :
    res = (img_b3 - img_b8) / (img_b3 + img_b8)
    return res

def ndvi(img_b4, img_b8) :
    res = (img_b8 - img_b4) / (img_b8 + img_b4)
    return res

def read_band(date, band_name):
    path = os.path.join("sentinel-2 image", "2021")
    date_path = os.path.join(path, date)
    band_path = os.path.join(date_path, "IMG_DATA", f"47PQS_{date}_{band_name}.jp2")
    band = rasterio.open(band_path)
    return band

def test() :
    lst_band_4 = [
        read_band("20210804", "B04"),
        read_band("20210809", "B04"),
        read_band("20210819", "B04"),
        read_band("20210824", "B04"),
        read_band("20210829", "B04"),
    ]
    lst_band_8 = [
        read_band("20210804", "B08"),
        read_band("20210809", "B08"),
        read_band("20210819", "B08"),
        read_band("20210824", "B08"),
        read_band("20210829", "B08"),
    ]


    pass

if __name__ == "__main__" :
    main()