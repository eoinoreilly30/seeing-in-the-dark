import rawpy
import csv
import os
import numpy as np

checkpoint_dir = '.\\result_Sony\\'
result_dir = '.\\result_Sony\\'

train_list = csv.reader(open('train_list_pngs.txt'), delimiter=" ")

input_image_paths = []
gt_image_dict = {}

# extract paths from train list
for line in train_list:
    input_image_paths.append(os.path.abspath(line[0]))
    gt_image_dict[os.path.abspath(line[0])] = os.path.abspath(line[1])

ps = 512
save_freq = 500

# data generator here

# model here
