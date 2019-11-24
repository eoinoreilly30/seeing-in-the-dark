import csv
import os
import tensorflow as tf

from data_generator import DataGenerator
from subpixel import SubpixelConv2D
from keras import Model
from keras.layers import Layer, Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.initializers import TruncatedNormal
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

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


class UpSampleAndConcat(Layer):
    def __init__(self, x2, input_channels, output_channels, pool_size=2, **kwargs):
        self.x2 = x2
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.pool_size = pool_size
        super(UpSampleAndConcat, self).__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        # Create a trainable weight variable for this layer.
        self.deconv_filter = self.add_weight(name="deconv_filter",
                                             shape=(self.pool_size, self.pool_size,
                                                    self.output_channels, self.input_channels),
                                             initializer=TruncatedNormal(stddev=0.02),
                                             trainable=True)
        super(UpSampleAndConcat, self).build(input_shape)

    def call(self, x1, **kwargs):
        deconv = tf.nn.conv2d_transpose(x1, self.deconv_filter,
                                        tf.shape(self.x2), strides=[1, self.pool_size, self.pool_size, 1])
        deconv_output = tf.concat([deconv, self.x2], 3)
        deconv_output.set_shape([None, None, None, self.output_channels*2])
        return deconv_output

    def compute_output_shape(self, input_shape):
        h = self.x2.get_shape().as_list()[1]
        w = self.x2.get_shape().as_list()[2]
        return input_shape[0], h, w, self.output_channels*2


# functional model
input_layer = Input(shape=(2120, 1416, 4))

conv1 = Conv2D(32, (3, 3), activation="relu")(input_layer)
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

up6 = UpSampleAndConcat(conv4, 512, 256)(conv5)
# up6 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", output_padding=4)(conv5)
# concat6 = concatenate([up6, conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation="relu")(up6)
conv6 = Conv2D(256, (3, 3), activation="relu")(conv6)

# up7 = Conv2DTranspose(128, (3, 3), strides=(2, 2))(conv6)
# concat7 = concatenate([up7, conv3], axis=3)
up7 = UpSampleAndConcat(conv3, 256, 128)(conv6)
conv7 = Conv2D(128, (3, 3), activation="relu")(up7)
conv7 = Conv2D(128, (3, 3), activation="relu")(conv7)

# up8 = Conv2DTranspose(64, (3, 3), strides=(2, 2))(conv7)
# concat8 = concatenate([up8, conv2], axis=3)
up8 = UpSampleAndConcat(conv2, 128, 64)(conv7)
conv8 = Conv2D(64, (3, 3), activation="relu")(up8)
conv8 = Conv2D(64, (3, 3), activation="relu")(conv8)

# up9 = Conv2DTranspose(32, (3, 3), strides=(2, 2))(conv8)
# concat9 = concatenate([up9, conv1], axis=3)
up9 = UpSampleAndConcat(conv1, 64, 32)(conv8)
conv9 = Conv2D(32, (3, 3), activation="relu")(up9)
conv9 = Conv2D(32, (3, 3), activation="relu")(conv9)

conv10 = Conv2D(12, (1, 1))(conv9)

output_layer = SubpixelConv2D(conv10.shape, 2)(conv10)

model = Model(input=input_layer, output=output_layer)
model.summary()

model.compile(Adam(lr=0.0001), loss="mean_absolute_error", metrics=['accuracy'])

# use tensorboard
file_path = "weights.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
model.fit_generator(generator=training_generator,
                    epochs=4000,
                    callbacks=[checkpoint],
                    workers=6,
                    shuffle=True)


