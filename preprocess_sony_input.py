import pickle
import glob
import os
import numpy as np
import rawpy

# reads in raw image and returns packed image
def pack_raw(in_path):
	# read in raw image, return raw object
	raw = rawpy.imread(in_path)
	
	# get image from raw object
	im = raw.raw_image_visible.astype(np.float32)
	
	# subtract the black level (512), cap at 0
	im = np.maximum(im - 512, 0)
	
	# normalize all pixels between 0 - 1
	# raw data is 14 bits
	# 2^14 = 16383 = max value
	im = im / (16383 - 512)
	
	# add dimension to store colours RGBG
	im = np.expand_dims(im, axis=2)
	img_shape = im.shape
	H = img_shape[0]
	W = img_shape[1]
	
	# extract each colour
	out = np.concatenate(
	(im[0:H:2, 0:W:2, :],
	 im[0:H:2, 1:W:2, :],
	 im[1:H:2, 1:W:2, :],
	 im[1:H:2, 0:W:2, :]), axis=2)
	 
	return out

in_dir = './input/'
out_dir = './output/'

# get all files with .ARW (raw) extension
files = glob.glob(in_dir + '*.ARW')

# get file ids
file_ids = []
for file in files:
	file_ids.append(os.path.basename(file))

# loop through all files, pack, then dump 
# back to disk
for id in file_ids:
	out = open(out_dir + id, 'wb')
	data = pack_raw(in_dir + id)
	pickle.dump(data, out)
	out.close()