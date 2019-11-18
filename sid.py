import rawpy
import csv
import imageio
import os
import numpy as np
from matplotlib import pyplot as plt

checkpoint_dir = '.\\result_Sony\\'
result_dir = '.\\result_Sony\\'

train_list = csv.reader(open('train_list_pngs.txt'), delimiter=" ")

input_image_paths = []
gt_image_paths = []

# extract paths from train list
for line in train_list:
    input_image_paths.append(os.path.abspath(line[0]))
    gt_image_paths.append(os.path.abspath(line[1]))

# load input images
input_images = []
for path in input_image_paths[0:1]:
    raw_image = rawpy.imread(path)

    # pack Bayer image to 4 channels
    im = raw_image.raw_image_visible.astype(np.float32)

    plt.imshow(im)

    im = np.maximum(im - 512, 0) / (16383 - 512)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)

    input_images.append(out)

    # gt_images.append(imageio.imread(gt_path))


ps = 512
save_freq = 500

# data generator here

# model here
