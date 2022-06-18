import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import mapping
from skimage.io import *
from skimage.transform import *
import matplotlib.pyplot as plt
import cv2
import os

from rasterio import plot

def get_pixel_from_coor(img, coor) :
	#### return pixel
	return img.index(coor[0], coor[1])

def get_pixel_polygon(img, polys) :
	p_poly = []
	for p in polys :
		x, y = get_pixel_from_coor(img, p)
		p_poly.append([y, x])
	return p_poly

def create_mask_from_coordinates(df, mask_tiff) :
	"""
	use this function to create mask of polygon from coordinate to pixel

	df - training, testing file
	"""
	mask = np.expand_dims(np.zeros(mask_tiff.shape), axis=-1).astype('uint8')
	mask = np.dstack((mask, mask, mask))

	#### color to use in create mask
	color = {
		"1": [255, 0, 0],
		"2": [0, 255, 0],
		"3": [0, 0, 255],
		"4": [0, 255, 255],
	}

	for c in color :
		sel_df = df[df["crop_type"] == c]

		sel_df["p_poly"] = sel_df["geometry"].apply(lambda x: get_pixel_polygon(mask_tiff, mapping(x)["coordinates"][0]))
		lst_poly = sel_df["p_poly"].values.tolist()
		for poly in lst_poly :
			poly = np.array(poly).astype(int)
			mask = cv2.fillPoly(mask, pts=[poly], color=color[c])
	
	#### visualize
	# plt.figure(figsize=(10, 10))
	# plt.title('mask')
	# plt.imshow(mask)
	# imshow(mask)
	return mask

def crate_mask() :
	path = os.path.join("sentinel-2 image", "2021", "all_date")

	dst_path = os.path.join("sentinel-2 image", "2021", "label")
	os.makedirs(dst_path, exist_ok=True)
	mask = np.zeros((2051, 2051))
	df = gpd.read_file("traindata.shp")
	for f in os.listdir(path) :
		mask_coor = create_mask_from_coordinates(df=df, mask_tiff=mask.copy())
		dst_file_path = os.path.join(dst_path, f)
		imsave(dst_file_path, mask_coor)

def get_std_from_polygon(poly) :
	return np.std(poly)

def get_mean_from_polygon(poly) :
	return np.mean(poly)

def get_col_array(x, col) :
	#### col use when select x, y in coordinates (0, 1)
	return np.array(x)[:, col]

def get_calculate_value(x, method, col) :

	x = get_col_array(x, col)

	if method == "std" :
		res = get_std_from_polygon(x)
	elif method == "mean" :
		res = get_mean_from_polygon(x)

	return res

def create_column_mean_std(df) :
	"""
	use this function to calculate mean, std from each point of polygon and split x, y
	"""

	# ((774604.0143275484, 1671240.0548827336),
	#  (774606.5055553364, 1671331.0924999372),
	#  (774679.5517776415, 1671330.1703888504),
	#  (774675.7978508549, 1671227.4928340595),
	#  (774666.4300546482, 1671219.0801929508),
	#  (774604.0143275484, 1671240.0548827336))
	df["mean_1"] = df["geometry"].apply(
		lambda x: get_calculate_value(
			x=mapping(x)["coordinates"][0], 
			method="mean", 
			col=0
		)
	)
	df["mean_2"] = df["geometry"].apply(
		lambda x: get_calculate_value(
			x=mapping(x)["coordinates"][0], 
			method="mean", 
			col=1
		)
	)

	df["std_1"] = df["geometry"].apply(
		lambda x: get_calculate_value(
			x=mapping(x)["coordinates"][0], 
			method="std", 
			col=0
		)
	)
	df["std_2"] = df["geometry"].apply(
		lambda x: get_calculate_value(
			x=mapping(x)["coordinates"][0], 
			method="std", 
			col=1
		)
	)

crate_mask()
# target_img = rasterize(
# 	shapes=
# )