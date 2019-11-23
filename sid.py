import csv
import os
import tensorflow as tf

from data_generator import DataGenerator
from subpixel import SubpixelConv2D
from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Conv2DTranspose, concatenate

checkpoint_dir = '.\\result_Sony\\'
result_dir = '.\\result_Sony\\'

train_list = csv.reader(open('train_list_pngs.txt'), delimiter=" ")

input_image_paths = []
gt_image_dict = {}

# extract paths from train list
for line in train_list:
    input_image_paths.append(os.path.abspath(line[0]))
    gt_image_dict[os.path.abspath(line[0])] = os.path.abspath(line[1])

data_gen_params = {'input_dim': (2120, 1416),  # 4240/2, 2832/2
                   'input_channels': 4,
                   'output_dim': (4240, 2832),
                   'output_channels': 3,
                   'batch_size': 32,
                   'shuffle': True}

training_generator = DataGenerator(input_image_paths, gt_image_dict, **data_gen_params)


# def UpsampleAndConcat(x1, x2, input_channels, output_channels):
#     def upsample_and_concat(x):
#         pool_size = 2
#         deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, input_channels], stddev=0.02))
#         deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
#
#         deconv_output = tf.concat([deconv, x2], 3)
#         deconv_output.set_shape([None, None, None, output_channels * 2])
#
#         return deconv_output
#
#     def output_shape():
#         return [None, None, None, output_channels * 2]
#
#     return Lambda(upsample_and_concat, output_shape=output_shape, name='upsample_and_concat')


# functional model
input_layer = Input(shape=(2120, 1416, 4))

conv1 = Conv2D(32, (3, 3), input_shape=(2120, 1416, 4), activation="relu")(input_layer)
conv1 = Conv2D(32, (3, 3), activation="relu")(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv1)

conv2 = Conv2D(64, (3, 3), activation="relu")(pool1)
conv2 = Conv2D(64, (3, 3), activation="relu")(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv2)

conv3 = Conv2D(128, (3, 3), activation="relu")(pool2)
conv3 = Conv2D(128, (3, 3), activation="relu")(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv3)

conv4 = Conv2D(256, (3, 3), activation="relu")(pool3)
conv4 = Conv2D(256, (3, 3), activation="relu")(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv4)

conv5 = Conv2D(512, (3, 3), activation="relu")(pool4)
conv5 = Conv2D(512, (3, 3), activation="relu")(conv5)

# Conv2DTranspose
# concat
# convolution
print(conv4.shape)
up6 = Conv2DTranspose(256, (3, 3), strides=(2, 2))(conv5)
# concat6 = concatenate([up6, conv4], axis=3)

model = Model(input=input_layer, output=up6)
model.summary()

conv6 = Conv2D(256, (3, 3), activation="relu")(concat6)
conv6 = Conv2D(256, (3, 3), activation="relu")(conv6)

up7 = Conv2DTranspose(128, (3, 3), strides=(2, 2))(conv6)
concat7 = concatenate([up7, conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation="relu")(concat7)
conv7 = Conv2D(128, (3, 3), activation="relu")(conv7)

up8 = Conv2DTranspose(64, (3, 3), strides=(2, 2))(conv7)
concat8 = concatenate([up8, conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation="relu")(concat8)
conv8 = Conv2D(64, (3, 3), activation="relu")(conv8)

up9 = Conv2DTranspose(32, (3, 3), strides=(2, 2))(conv8)
concat9 = concatenate([up9, conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation="relu")(concat9)
conv9 = Conv2D(32, (3, 3), activation="relu")(conv9)

conv10 = Conv2D(12, (1, 1))(conv9)

output_layer = SubpixelConv2D(conv10.shape, 2)(conv10)

