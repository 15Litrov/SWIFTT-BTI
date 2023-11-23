import os
import numpy as np
import pandas as pd
from osgeo import gdal

IMAGE_DIR = r"D:\SWIFTT_dataset_France"

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
	df = pd.read_excel(os.path.join(IMAGE_DIR, xlsx_name))
	print(df.shape)
	output_array = [None]*df.shape[0]
	for i, row in df.iterrows():
		json_index = row['geojson']
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
				output_array[i] = (json_index, f)
				break

	return output_array

def splitImageIntoH_and_S(image, ftype, mask):
    # ravel image to 1d array of bands
    image = image.reshape((image.shape[0], image.shape[1] * image.shape[2]))
    # ravel ftype and mask to 1d array
    ftype = ftype.reshape((ftype.shape[0] * ftype.shape[1]))
    mask = mask.reshape((mask.shape[0] * mask.shape[1]))

    H = image[:, np.where((ftype == 2) & (mask == 0))]
    H = H.reshape((H.shape[0], H.shape[2]))
    S = image[:, np.where((ftype == 2) & (mask == 1))]
    S = S.reshape((S.shape[0], S.shape[2]))

    return H, S

def getH_and_S(dictOfImages, tiles):
    data_array = [None] * len(tiles)
    total_H, total_S = 0, 0
    for i in range(len(tiles)):
        tile = tiles[i]
        image, ftype, mask = loadImage(dictOfImages[tile], LOAD_ALL)
        H, S = splitImageIntoH_and_S(image, ftype, mask)
        total_H += H.shape[1]
        total_S += S.shape[1]
        data_array[i] = (H, S)

    offset_H, offset_S = 0, 0
    H = np.empty((data_array[0][0].shape[0], total_H))
    S = np.empty((data_array[0][1].shape[0], total_S))
    for i in range(len(tiles)):
        new_offset_H = offset_H + data_array[i][0].shape[1]
        H[:, offset_H:new_offset_H] = data_array[i][0]

        new_offset_S = offset_S + data_array[i][1].shape[1]
        S[:, offset_S:new_offset_S] = data_array[i][1]

        offset_H, offset_S = new_offset_H, new_offset_S

    return H, S

def joinData(H, S):
    buffer = np.empty((H.shape[0] + S.shape[0], H.shape[1]))
    buffer[:H.shape[0], :] = H
    buffer[H.shape[0]:, :] = S

    labels = np.empty((H.shape[0] + S.shape[0]), dtype="int8")
    labels[:H.shape[0]] = 0
    labels[H.shape[0]:] = 1

    return buffer, labels

def leaveFinite(data):
    return data[:, (~np.isnan(data).any(axis=0)) & (np.isfinite(data).all(axis=0))]

def prepareData(data):
    return leaveFinite(data).swapaxes(0, 1)