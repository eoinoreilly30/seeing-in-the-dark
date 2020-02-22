"""fuji

Original file is located at
    https://colab.research.google.com/drive/15pBEE6WoeCFBNOiOLapzPD0F8egBJj0B
"""

import csv
import numpy as np
import tensorflow as tf
import math
import pickle
import glob
import time

from datetime import date

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Lambda
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, Callback
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence

tf.keras.backend.clear_session()

ps = 512

TESTING = False


def read_in_pickle(path):
    print('reading in ' + path)
    infile = open(path, 'rb')
    image = pickle.load(infile)
    infile.close()
    return image


input_dir = './input/'
gt_dir = './gt/'
result_dir = './result/'

# read in training/validation/testing list
if TESTING:
    input_list = csv.reader(open('/content/drive/My Drive/FYP/fuji/test_list_no_burst.txt'), delimiter=" ")
else:
    input_list = csv.reader(open('/content/drive/My Drive/FYP/fuji/train_list_no_burst.txt'), delimiter=" ")
    validation_list = csv.reader(open('/content/drive/My Drive/FYP/fuji/validation_list_no_burst.txt'), delimiter=" ")


input_fns = []
input_gt_fns = []
validation_fns = []
validation_gt_fns = []

# extract ids from input list
for line in input_list:
    input_fns.append(line[0])
    input_gt_fns.append(line[1])

# extract ids from validation list
if not TESTING:
    for line in validation_list:
        validation_fns.append(line[0])
        validation_gt_fns.append(line[1])

# shorten dataset for testing
# input_fns = input_fns[0:5]
# input_gt_fns = input_gt_fns[0:5]
# validation_fns = validation_fns[0:1]
# validation_gt_fns = validation_gt_fns[0:1]

input_gt_fns = [fn.split('.')[0]+'.RAF' for fn in input_gt_fns]
validation_gt_fns = [fn.split('.')[0]+'.RAF' for fn in validation_gt_fns]

# load in input images
print("loading in input images...")
input_map = {}
for fn in input_fns:
    input_map[fn] = read_in_pickle(input_dir + fn)

print('done')

# load in gt images
print("loading in gt images...")
gt_map = {}
for fn in input_gt_fns:
    gt_map[fn] = read_in_pickle(gt_dir + fn)

print('done')

if not TESTING:
    print("loading in validation images...")
    for fn in validation_fns:
        input_map[fn] = read_in_pickle(input_dir + fn)

    print('done')

    print("loading in validation gt images...")
    for fn in validation_gt_fns:
        gt_map[fn] = read_in_pickle(gt_dir + fn)

    print('done')

# keras layer that implements tf.depth_to_space
def SubpixelConv2D(input_shape, scale):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape, name='subpixel')


def augment_images(input_images, gt_images):
    H = input_images.shape[1]
    W = input_images.shape[2]

    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)
    input_patch = input_images[:, yy:yy + ps, xx:xx + ps, :]
    gt_patch = gt_images[:, yy * 3:yy * 3 + ps * 3, xx * 3:xx * 3 + ps * 3, :]

    if np.random.randint(2, size=1)[0] == 1:  # random flip
        input_patch = np.flip(input_patch, axis=1)
        gt_patch = np.flip(gt_patch, axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=2)
        gt_patch = np.flip(gt_patch, axis=2)
    if np.random.randint(2, size=1)[0] == 1:  # random transpose
        input_patch = np.transpose(input_patch, (0, 2, 1, 3))
        gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

    input_patch = np.minimum(input_patch, 1.0)

    return input_patch, gt_patch


# dynamically load in images
class DataGenerator(Sequence):
    def __init__(self, input_fns, gt_fns, batch_size):
        self.input_fns = input_fns
        self.gt_fns = gt_fns
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.input_fns) / self.batch_size)

    def __getitem__(self, i):
        input_fns_tmp = self.input_fns[i*self.batch_size : (i+1)*self.batch_size]
        gt_fns_tmp = self.gt_fns[i*self.batch_size : (i+1)*self.batch_size]
        
        # ratio only handles batch size of 1
        in_exposure = float(input_fns_tmp[0][9:-5])
        gt_exposure = float(gt_fns_tmp[0][9:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        
        # change to np.expanddims
        input_images = np.array([input_map[fn] for fn in input_fns_tmp])
        gt_images = np.array([gt_map[fn] for fn in gt_fns_tmp])

        # batch = augment_images(input_images, gt_images, ratio)
        input_images *= ratio

        if TESTING:
            input_images = np.minimum(input_images, 1.0)
            input_images = input_images[:, :, 0:992, :] # divisible by 32
            gt_images = gt_images[:, :, 0:2976, :]
            batch = (input_images, gt_images)
        else:
            batch = augment_images(input_images, gt_images)

        return batch


# functional model
if TESTING:
    input_layer = Input(shape=(672, 992, 9))
else:
    input_layer = Input(shape=(ps, ps, 9))

conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

up6 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(conv5)
concat6 = concatenate([up6, conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(concat6)
conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

up7 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(conv6)
concat7 = concatenate([up7, conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(concat7)
conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

up8 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(conv7)
concat8 = concatenate([up8, conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(concat8)
conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

up9 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(conv8)
concat9 = concatenate([up9, conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(concat9)
conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

conv10 = Conv2D(27, (1, 1), padding="same")(conv9)

output_layer = SubpixelConv2D(conv10.shape, 3)(conv10)

model = Model(inputs=input_layer, outputs=output_layer)

# model.summary()

model.load_weights('/content/drive/My Drive/FYP/fuji/weights_after_pack/weights.0805-0.6983.hdf5')

model.compile(Adam(lr=1e-4), loss="mean_absolute_error", metrics=['accuracy'])

checkpoint = ModelCheckpoint('/content/drive/My Drive/FYP/fuji/weights_after_pack/weights.{epoch:04d}-{val_acc:.4f}.hdf5',
                             monitor='val_acc', 
                             verbose=0,
                             save_best_only=True)

class TimerCallback(tf.keras.callbacks.Callback):

    def __init__(self, start_time, start_date):
        self.start_time = start_time
        self.start_date = start_date

    def on_epoch_end(self, epoch, logs=None):
        total_time = str(time.time() - self.start_time)
        time_file = open('/content/drive/My Drive/FYP/fuji/train_time.txt', 'a')
        time_file.write('Colab, Date: ' + self.start_date + ' ' + total_time + '\n')
        time_file.close()


def scheduler(epoch):
    if epoch >= 2000:
        return 1e-5
    else:
        return 1e-4

learning_rate_reduce = LearningRateScheduler(scheduler, verbose=0)

today = str(date.today())
csv_logger = CSVLogger('/content/drive/My Drive/FYP/fuji/logs/training-' + today + '.log')

batch_size = 1

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if TESTING:
    [loss, acc] = model.evaluate(DataGenerator(input_fns, input_gt_fns, batch_size),
                                 use_multiprocessing=True,
                                 workers=4)
    
    print('Loss: ' + str(loss) + ' | Accuracy: ' + str(acc*100))

    out = model.predict(DataGenerator(input_fns, input_gt_fns, batch_size),
                        use_multiprocessing=True,
                        workers=4,
                        verbose=1)
    
    for idx, im in enumerate(out):
        # Rescale to 0-255 and convert to uint8
        rescaled = (255.0 / im.max() * (im - im.min())).astype(np.uint8)
        image = Image.fromarray(rescaled)
        filename = './result/' + input_fns[idx][0:-4] + '.png'
        print('saving ' + filename)
        image.save(filename)
        image.close()
else:
    start_time = time.time()
    time_callback = TimerCallback(start_time, today)

    history = model.fit(DataGenerator(input_fns, input_gt_fns, batch_size),
                        validation_data=DataGenerator(validation_fns, validation_gt_fns, batch_size),
                        validation_freq=1,
                        epochs=4000,
                        initial_epoch=887,
                        use_multiprocessing=True,
                        shuffle=True,
                        workers=4,
                        max_queue_size=10,
                        callbacks=[checkpoint, learning_rate_reduce, csv_logger, time_callback])
    
    history_file = open('/content/drive/My Drive/FYP/fuji/history/' + today + '.pickle', 'wb')
    pickle.dump(history, history_file)
    history_file.close()
    
    total_time = time.time() - start_time
    time_file = open('/content/drive/My Drive/FYP/fuji/train_time.txt', 'a')
    time_file.write('Colab, Date: ' + today + ' ' + total_time + '\n')
    time_file.close()