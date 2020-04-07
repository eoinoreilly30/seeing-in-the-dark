import pickle
import rawpy
import numpy as np
import glob
import os

# reads in raw image
# returns postprocessed image cropped by 2
def scale(path):
	# read in raw image, returns raw object
	gt_raw = rawpy.imread(path)
	
	# postprocess raw image to RGB image
	im = gt_raw.postprocess(use_camera_wb=True, 
		half_size=False, no_auto_bright=True, output_bps=16)
	
	# crop image by 2 in each axis
	H = im.shape[0]
	W = im.shape[1]
	im = im[0:H/2, 0:W/2, :]
	
	return im

files_dir = './gt_raw/'
done_dir = './output/'

# get all files with .RAF extension
files = glob.glob(files_dir + '*.RAF')

# get file ids
file_ids = []
for file in files:
	file_ids.append(os.path.basename(file))

# loop through all files, crop
# dump back to disk
for id in left_ids:
	out = open(done_dir + id, 'wb')
	data = scale(files_dir + id)
	pickle.dump(data, out)
	out.close()