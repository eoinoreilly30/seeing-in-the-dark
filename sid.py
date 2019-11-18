import rawpy
import csv
import imageio
import os
import numpy as np

checkpoint_dir = '.\\result_Sony\\'
result_dir = '.\\result_Sony\\'

train_list = csv.reader(open('Sony_train_list.txt'), delimiter=" ")

input_images = []
gt_images = []

new_list = []

for line in train_list:
    l = line[1].split('.')
    line = l[1] + '.png'


# for line in train_list:
#     in_path = os.path.abspath(line[0])
#     raw_image = rawpy.imread(in_path)
#
#     # pack Bayer image to 4 channels
#     im = raw_image.raw_image_visible.astype(np.float32)
#     im = np.maximum(im - 512, 0) / (16383 - 512)
#
#     im = np.expand_dims(im, axis=2)
#     img_shape = im.shape
#     H = img_shape[0]
#     W = img_shape[1]
#
#     out = np.concatenate((im[0:H:2, 0:W:2, :],
#                           im[0:H:2, 1:W:2, :],
#                           im[1:H:2, 1:W:2, :],
#                           im[1:H:2, 0:W:2, :]), axis=2)
#
#     input_images.append(out)
#
#     gt_path = os.path.abspath(line[1])
#     # gt_images.append(imageio.imread(gt_path))



ps = 512
save_freq = 500

# data generator here

# model here

