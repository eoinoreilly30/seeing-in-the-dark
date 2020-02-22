import numpy as np
import keras
import rawpy
import imageio


class DataGenerator(keras.utils.Sequence):

    def __init__(self, input_file_names, gt_image_dict, input_dim, input_channels,
                 output_dim, output_channels, batch_size, shuffle=True):
        self.input_file_names = input_file_names
        self.gt_image_dict = gt_image_dict
        self.input_dim = input_dim
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.output_channels = output_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.input_file_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.input_file_names) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one (item) batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        input_file_names_temp = [self.input_file_names[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(input_file_names_temp)

        return x, y

    def __data_generation(self, input_file_names_temp):
        """Generates data containing batch_size samples"""  # x : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, *self.input_dim, self.input_channels))
        y = np.empty((self.batch_size, *self.output_dim, self.output_channels))

        # Generate data
        for i, filename in enumerate(input_file_names_temp):
            # Store input image
            x[i, ] = self.__read_in_raw_image(filename)

            # Store ground truth image
            y[i, ] = self.__read_in_ground_truth_image(filename)

        return x, y

    @staticmethod
    def __read_in_raw_image(filename):
        raw = rawpy.imread(filename)

        # pack Bayer image to 4 channels
        im = raw.raw_image_visible.astype(np.float32)
        im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        h = img_shape[0]
        w = img_shape[1]

        out = np.concatenate((im[0:h:2, 0:w:2, :],
                              im[0:h:2, 1:w:2, :],
                              im[1:h:2, 1:w:2, :],
                              im[1:h:2, 0:w:2, :]), axis=2)
        return out

    def __read_in_ground_truth_image(self, filename):
        gt_filename = self.gt_image_dict[filename]
        return imageio.imread(gt_filename)