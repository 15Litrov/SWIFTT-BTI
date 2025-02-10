import os
import numpy as np
import pandas as pd
from osgeo import gdal

# root folder of train&validation dataset
IMAGE_DIR = r"SWIFTT_dataset_France"

LOAD_IMAGE_BIT = 1
LOAD_FTYPE_BIT = 2
LOAD_IMASK_BIT = 4
LOAD_ALL = LOAD_IMAGE_BIT | LOAD_FTYPE_BIT | LOAD_IMASK_BIT

def getBands(img):
	bands = np.empty((img.RasterCount, img.RasterYSize, img.RasterXSize))
	for i in range(img.RasterCount):
		bands[i, :, :] = img.GetRasterBand(i + 1).ReadAsArray()

	return bands

def loadImage(file_name, load_flags):
	json_index = file_name[8:10]
	if ('-' in json_index):
		json_index = json_index[0]
	json_index = json_index

	target_img = file_name.strip('\n')

	output_image, output_ftype, output_imask = None, None, None
	if (load_flags & LOAD_IMASK_BIT) == LOAD_IMASK_BIT:
		img = gdal.Open(os.path.join(IMAGE_DIR, json_index, f"geojson_{json_index}_mask.tif"))
		output_imask = img.GetRasterBand(1).ReadAsArray()
		img.FlushCache()

	if (load_flags & LOAD_FTYPE_BIT) == LOAD_FTYPE_BIT:
		img = gdal.Open(os.path.join(IMAGE_DIR, json_index, f"geojson_{json_index}_forest_type_2018.tif"))
		output_ftype = img.GetRasterBand(1).ReadAsArray()
		img.FlushCache()

	if (load_flags & LOAD_IMAGE_BIT) == LOAD_IMAGE_BIT:
		img = gdal.Open(os.path.join(IMAGE_DIR, json_index, target_img))
		output_image = getBands(img)
		img.FlushCache()

	return output_image, output_ftype, output_imask

def getStressedImagesNames(xlsx_name):
	df = pd.read_excel(xlsx_name)
	output_array = []
	for _, row in df.iterrows():
		json_index = row['geojson']
		folder_path = os.path.join(IMAGE_DIR, str(json_index))
		if not os.path.exists(folder_path):
			continue

		date_range = row['time range']
		parts = str.split(date_range, '" - "')
		begin = str.split(parts[0][1:], '-')
		end = str.split(parts[1][:-1], '-')
				
		# 2018-10-01  -> 20181001
		begin_int = int(begin[0]) * 100 * 100 + int(begin[1]) * 100 + int(begin[2])
		end_int = int(end[0]) * 100 * 100 + int(end[1]) * 100 + int(end[2])

		files = os.listdir(os.path.join(IMAGE_DIR, str(json_index)))
		for f in files:
			parts = str.split(f, '-')
			if len(parts) == 1 or (not str.endswith(f, '.tif')):
				continue

			begin = parts[1:4]
			end = parts[4:7]

			img_begin_int = int(begin[0]) * 100 * 100 + int(begin[1]) * 100 + int(begin[2])
			img_end_int = int(end[0]) * 100 * 100 + int(end[1]) * 100 + int(end[2])

			if begin_int <= img_begin_int and img_end_int <= end_int:
				output_array.append((json_index, f))
				break

	return output_array