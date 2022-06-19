import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import mapping
from skimage.io import *
from skimage.transform import *
import matplotlib.pyplot as plt
import cv2
import os
import rasterio

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
	print(df.geometry.bounds())
	# for f in os.listdir(path) :
	# 	mask_coor = create_mask_from_coordinates(df=df, mask_tiff=mask.copy())
	# 	dst_file_path = os.path.join(dst_path, f)
	# 	imsave(dst_file_path, mask_coor)

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

# crate_mask()
# target_img = rasterize(
# 	shapes=
# )

def create_patch() :
	path = os.path.join("sentinel-2 image", "2021", "all_date")

	# dst_path = os.path.join("sentinel-2 image", "2021", "label")
	# os.makedirs(dst_path, exist_ok=True)
	# mask = np.zeros((2051, 2051))
	df = gpd.read_file("traindata.shp")
	# print(df.geometry.bounds())

	# path = os.path.join("datasets", "varuna_1", "label", "gt.png")
	# img = cv2.imread(path)

	# img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# thresh = 100
	# #get threshold image
	# ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
	# thresh_img = ((thresh_img > 0) * 255.).astype('uint8')
	# #find contours
	# contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# img_contours = np.zeros(img.shape)

	# for i, cnt in enumerate(contours) :
		
	# 	x, y, w, h = cv2.boundingRect(cnt)
	# 	print(x, y, w, h)

	# 	cv2.rectangle(img_contours, (x, y), (x + w, y + h), 255)


	# 	# break
	# cv2.imshow('img1', thresh_img)
	# cv2.imshow('img', img_contours)
	# cv2.waitKey()
	# draw the contours on the empty image
	# cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)

	src_mask = rasterio.open('20210101-RGB_masked.tif')

	colors = {
		"1": (255, 0, 0),
		"2": (0, 255, 0),
		"3": (0, 0, 255),
		"4": (0, 255, 255),
	}

	patch_size = np.zeros((160, 160, 3)).astype('uint8')


	df[["minx", "miny", "maxx", "maxy"]] = df["geometry"].bounds

	dst_path = os.path.join("sentinel-2 image", "2021", "gt")
	os.makedirs(dst_path, exist_ok=True)

	df["p_poly"] = df["geometry"].apply(lambda x: get_pixel_polygon(src_mask, mapping(x)["coordinates"][0]))

	lst_bounds = []

	for i, row in df.iterrows() :
		pass
		color = colors[row["crop_type"]]
		mask_tiff = np.zeros((2051, 2051, 3)).astype('uint8')
		poly = np.array(row["p_poly"]).astype(int)
		mask_tiff = cv2.fillPoly(mask_tiff, pts=[poly], color=color)

		miny, minx = get_pixel_from_coor(src_mask, (row["minx"], row["miny"]))
		maxy, maxx = get_pixel_from_coor(src_mask, (row["maxx"], row["maxy"]))

		minx = minx if minx % 2 == 0 else minx + 1
		miny = miny if miny % 2 == 0 else miny + 1
		maxx = maxx if maxx % 2 == 0 else maxx + 1
		maxy = maxy if maxy % 2 == 0 else maxy + 1

		centerx = abs(minx - maxx) // 2
		centery = abs(miny - maxy) // 2
		min_b_x = ((patch_size.shape[1] // 2) - centerx)
		max_b_x = ((patch_size.shape[1] // 2) + centerx)

		min_b_y = ((patch_size.shape[0] // 2) - centery)
		max_b_y = ((patch_size.shape[0] // 2) + centery)

		size_x = abs(minx - maxx)
		size_y = abs(miny - maxy)
		# l_x = (centerx) - (patch_size.shape[1] // 2)
		l_x = (minx + centerx) - (patch_size.shape[1] // 2)
		# r_x = (centerx) + (patch_size.shape[1] // 2)
		r_x = (maxx - centerx) + (patch_size.shape[1] // 2)

		# l_y = (centery) - (patch_size.shape[0] // 2)
		# h_y = (centery) + (patch_size.shape[0] // 2)
		l_y = (miny + centery) - (patch_size.shape[1] // 2)
		h_y = (miny - centery) + (patch_size.shape[1] // 2)

		lst_bounds.append([minx, miny, maxx, maxy])

		aa = np.zeros(patch_size.shape).astype('uint8')
		# aa[min_b_y: max_b_y, min_b_x: max_b_x] = 
		
		file_name = f"gt-{i}.png"
		out = mask_tiff[maxy: miny, minx: maxx]
		imsave(os.path.join(dst_path, file_name), out)

	dst_path = os.path.join(dst_path, "input")
	os.makedirs(dst_path, exist_ok=True)
	for f in os.listdir(path) :
		os.makedirs(os.path.join(dst_path, f[:-4]), exist_ok=True)
		img = imread(os.path.join(path, f))
		# os.makedirs(os.path.join(dst_path, f), exist_ok=True)
		for i, item in enumerate(lst_bounds) :
			l, l_y, r, h_y = item
			# print(l, r, l_y, h_y)
			# cv2.imshow('img', img[h_y: l_y, l: r, :])

			# cv2.imshow('resize', cv2.resize(img, (1024, 1024)))
			# cv2.waitKey()
			
			# break
			out = img[h_y: l_y, l: r, :]
		
			dst_file_path = os.path.join(dst_path, f[:-4], f"{f[:-4]}-{i}.png")
			imsave(dst_file_path, out)
		# break


		# imsave(os.path.join(dst_path, file_name), aa)

		# bb = cv2.resize(mask_tiff, (1024, 1024))
		# cv2.imshow('mask_tiff', bb)
		# cv2.imshow('mask_tiff_crop', aa)
		# cv2.waitKey(0)
		# break
		# patch_size[]

	# for f in os.listdir(path) :
	# 	mask_coor = create_mask_from_coordinates(df=df, mask_tiff=mask.copy())
	# 	dst_file_path = os.path.join(dst_path, f)
	# 	imsave(dst_file_path, mask_coor)


create_patch()
