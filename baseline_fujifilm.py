import csv
import numpy as np
import tensorflow as tf
import math
import pickle
import glob
import time
import skimage.measure

from PIL import Image
from datetime import date

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Lambda
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, Callback
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence

# reset backend and clear local variables
tf.keras.backend.clear_session()

# boolean used to switch between training or testing
TESTING = True

#############################################################
# Load in dataset
#############################################################

input_dir = './input/'
gt_dir = './gt/'
result_dir = './result/'

# read in training/validation/testing list
if TESTING:
    input_list = csv.reader(open('test_list.txt'), delimiter=" ")
else:
    input_list = csv.reader(open('train_list.txt'), delimiter=" ")
    validation_list = csv.reader(open('validation_list.txt'), delimiter=" ")

# arrays to store filenames
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

# returns unpickled (deserialized) image
def read_in_pickle(path):
    print('reading in ' + path)
    infile = open(path, 'rb')
    image = pickle.load(infile)
    infile.close()
    return image

# load in input images
print("loading in input images...")
input_map = {}
for fn in input_fns:
    input_map[fn] = read_in_pickle(input_dir + fn)

print('done')

# load in ground truth images
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
    

#############################################################
# Helper function definitions
#############################################################

# called at the end of each epoch
# returns the learning rate for next epoch
def learningRateReduce(epoch):
    if epoch >= 2000:
        return 1e-5
    else:
        return 1e-4

# called at end of each epoch
# keeps track of elapsed time by writing to 
# text file
class TimerCallback(Callback):

    def __init__(self, start_time, start_date):
        self.start_time = start_time
        self.start_date = start_date

    def on_epoch_end(self, epoch, logs=None):
        total_time = str(time.time() - self.start_time)
        time_file = open('train_time.txt', 'a')
        time_file.write('Colab, Date: ' + self.start_date + ' ' 
                        + total_time + '\n')
        time_file.close()

# keras layer that implements tf.nn.depth_to_space()
def SubpixelConv2D(input_shape, scale):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape, name='subpixel')


ps = 512
# randomly crops a patch for training
def crop_images(input_image, gt_image):
    # get input height and width
    H = input_image.shape[1]
    W = input_image.shape[2]

    # random index to start cropping at
    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)

    # crop input image
    input_patch = input_image[:, yy:yy + ps, xx:xx + ps, :]

    # crop ground truth image
    # triple the patch size as input image is one third
    # size of output image
    gt_patch = gt_images[:, yy * 3:yy * 3 + ps * 3,
                         xx * 3:xx * 3 + ps * 3, :]

    return input_patch, gt_patch

# randomly flip or transpose images
def augment_images(input_patch, gt_patch):
    # random flip x-axis
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=1)
        gt_patch = np.flip(gt_patch, axis=1)

    # random flip y-axis
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=2)
        gt_patch = np.flip(gt_patch, axis=2)
    
    # random transpose
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.transpose(input_patch, (0, 2, 1, 3))
        gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

    return input_patch, gt_patch


# dynamically loads images and applies augmentations
class DataGenerator(Sequence):
    def __init__(self, input_fns, gt_fns, validation=False):
        self.input_fns = input_fns
        self.gt_fns = gt_fns
        self.validation = validation

    # calculates the number of batches in dataset
    # in this case batch size = 1, just return
    # length of dataset
    def __len__(self):
        return len(self.input_fns)

    # called at the beginning of each batch
    # returns batch as tuple (input, ground_truth)
    def __getitem__(self, i):
        input_fn = self.input_fns[i]
        gt_fn = self.gt_fns[i]

        # extract exposure value from filename
        in_exposure = float(input_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])

        # calculate amplification ratio
        ratio = np.minimum(gt_exposure/in_exposure, 300)

        # add batch axis
        input_image = np.expand_dims(input_map[input_fn], axis=0)
        gt_image = np.expand_dims(gt_map[gt_fn], axis=0)
        
        # apply amplification ratio
        input_image *= ratio

        # if testing, don't apply cropping or data augmentations
        if TESTING:
            input_image = np.minimum(input_image, 1.0)
            # each axis must be divisible by 32
            input_image = input_image[:, :, 0:992, :]
            gt_image = gt_image[:, :, 0:2976, :]
            batch = (input_image, gt_image)
        # if validating, don't apply augmentations
        elif self.validation:
            batch = crop_images(input_image, gt_image)
        else:
            input_image, gt_image = crop_images(input_image,
                                                gt_image)
            batch = augment_images(input_image, gt_image)
        
        return batch


#############################################################
# Model definition
#############################################################

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

# end contractive path, begin expansive path

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

# construct model
model = Model(inputs=input_layer, outputs=output_layer)

# check model is defined correctly
# model.summary()

# load weights if resuming training
model.load_weights('weights.2064-0.6992.hdf5')

# compile model
model.compile(Adam(lr=1e-4), 
              loss="mean_absolute_error",
              metrics=['acc'])

#############################################################
# Run training/testing
#############################################################

# save weights when loss is improved
checkpoint = ModelCheckpoint('{epoch:04d}-{val_loss:.4f}.hdf5',
                             monitor='val_loss', 
                             verbose=0,
                             save_best_only=True)

# reduce learning rate after 2000 epochs
learning_rate_reduce = LearningRateScheduler(learningRateReduce, verbose=0)

# record logs after every epoch
today = str(date.today())
csv_logger = CSVLogger('training-' + today + '.log')

if TESTING:
    # record test loss (MAE)
    [loss, acc] = model.evaluate(DataGenerator(input_fns, 
                                        input_gt_fns))
    
    print('Loss: ' + str(loss))

    # predict output images on test set
    out = model.predict(DataGenerator(input_fns,
                                      input_gt_fns),
                        verbose=1)
    
    psnr = []
    ssim = []
    
    # record metrics and also save images to disk
    for idx, im in enumerate(out):
        # get corresponding ground truth image
        gt_image = gt_map[input_gt_fns[idx]]
        
        # Rescale ground truth back to 0-255 and convert to uint8
        rescaled_gt = (255.0 / gt_image.max() * 
                       (gt_image - gt_image.min()))
        rescaled_gt = rescaled_gt.astype(np.uint8)

        # Rescale output back to 0-255 and convert to uint8
        rescaled_output = (255.0 / im.max() * (im - im.min()))
        rescaled_output = rescaled_output.astype(np.uint8)

        # record metrics
        psnr.append(skimage.measure.compare_psnr(rescaled_gt, 
                                                 rescaled_output))
        ssim.append(skimage.measure.compare_ssim(rescaled_gt,
                                                 rescaled_output,
                                                 multichannel=True))

        # extract id and create new filename
        filename = result_dir + input_fns[idx][0:-4] + '.png'
        
        # save image to disk
        print('saving ' + filename)
        image = Image.fromarray(rescaled_output)
        image.save(filename)

    # print results
    print('Mean PSNR: ' + str(np.mean(psnr)) + 
          ' | Mean SSIM: ' + str(np.mean(ssim)))
else:
    start_time = time.time()
    time_callback = TimerCallback(start_time, today)

    # begin training
    model.fit(DataGenerator(input_fns, input_gt_fns),
              validation_data=DataGenerator(validation_fns, 
                                            validation_gt_fns),
              validation_freq=1,
              epochs=4000,
              callbacks=[checkpoint, learning_rate_reduce, 
                         csv_logger, time_callback])