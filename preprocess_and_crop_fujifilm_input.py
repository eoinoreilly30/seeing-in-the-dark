import pickle
import rawpy
import numpy as np
import sys
import glob
import os

# reads in raw image, returns packed image
def pack_raw(path):
	# read in raw image, return raw object
	raw = rawpy.imread(path)
	
	# get image from raw object
	im = raw.raw_image_visible.astype(np.float32)
	
	# subtract the black level (1024), cap at 0
	im = np.maximum(im - 1024, 0)
	
	# normalize all pixels between 0 - 1
	# raw data is 14 bits
	# 2^14 = 16383 = max value
	im = im / (16383 - 1024)
	
	img_shape = im.shape
	H = (img_shape[0] // 6) * 6
	W = (img_shape[1] // 6) * 6
	
	out = np.zeros((H // 3, W // 3, 9))
	
	# 0 R
	out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
	out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
	out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
	out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]
	
	# 1 G
	out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
	out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
	out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
	out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]
	
	# 1 B
	out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
	out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
	out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
	out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]
	
	# 4 R
	out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
	out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
	out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
	out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]
	
	# 5 B
	out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
	out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
	out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
	out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]
	
	out[:, :, 5] = im[1:H:3, 0:W:3]
	out[:, :, 6] = im[1:H:3, 1:W:3]
	out[:, :, 7] = im[2:H:3, 0:W:3]
	out[:, :, 8] = im[2:H:3, 1:W:3]
	
	# crop image by factor of 2
	out = out[0:H/2, 0:W/2, :]
	
	return out

files_dir = './input/'
done_dir = './input_cropped/'

# get all files with .RAF extension
files = glob.glob(files_dir + '*.RAF')
done = glob.glob(done_dir + '*')

# get file ids
file_ids = []
for file in files:
	file_ids.append(os.path.basename(file))

# loop through all files, pack, then dump 
# back to disk
for id in file_ids:
	out = open(done_dir + id, 'wb')
	data = pack_raw(files_dir + id)
	pickle.dump(data, out)
	out.close()