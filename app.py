'''

CS445 Final Project, UIUC - Spring 2020
@Authors: 
    - Pranav Velamakanni (pranavv2@illinois.edu)
    - Tarik Koric (koric1@illinois.edu)

Summary:
    Requirements: Python 3+
    Modules: see README.md for a complete list

This project aims to isolate backgrounds from images.
The prediction is based on 3 models trained by us using a 3 sets of images compiled by us.
A 4th model is also used which is trained on all three data sets combined.
The training process can take several hours so the models have been exported as H5 and loaded here.
Refer to TrainModel.ipynb for the training code and data pre-processing.
NOTE: Training all 4 models takes around 8 hours on a non-GPU enabled system.

The following code provides an API to process new images.

'''

# Standard python imports
import logging
import argparse
import os
import warnings
import ntpath

# Third party libraries. 
# NOTE: See readme.md for a detailed guide on requirements to train and run the model.
import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image as IMG
from scipy.misc import imresize
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator

# Suppress warning from imresize
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class Classify:
    '''
    Initializes the models for prediction.
    '''

    def __init__(self, mode = 'all'):

        self.models = {
            'people' : 'models/people.h5',
            'cat' : 'models/cat.h5',
            'dog' : 'models/dog.h5',
            'all' : 'models/all.h5'
        }

        self.model = self.import_model(mode)

    def jaccard(self, actual, predicted, smooth = 1):
        '''
        Jaccrd index of the predicted and actual used as an indicator for the image segmentation model.
        '''

        inter = K.sum(K.abs(actual * predicted), axis = [1,2,3])
        union = K.sum(actual, [1,2,3]) + K.sum(predicted, [1,2,3]) - inter
        
        return K.mean((inter + smooth) / (union + smooth), axis = 0)


    def dice_coef(self, actual, predicted, smooth = 1):
        '''
        Dice coefficient (F1 score) of the predicted and actual used as an indicator for the image segmentation model.
        '''
        
        inter = K.sum(actual * predicted, axis = [1,2,3])
        union = K.sum(actual, axis = [1,2,3]) + K.sum(predicted, axis = [1,2,3])
        
        return K.mean((2. * inter + smooth) / (union + smooth), axis = 0)

    def import_model(self, model):
        '''
        Loads the saved H5 model file.
        '''

        logging.info('Loading model: {}'.format(model))
        return load_model(self.models.get(model), custom_objects={'jaccard':self.jaccard, 'dice_coef': self.dice_coef})

    def predict(self, data):
        '''
        Uses the corresponding model to predict.
        '''
        
        result = None

        if data.mode == 'dir':
            result = list()
            for img in data.data:
                result.append(self._predict(img))

        if data.mode == 'file' or data.mode == 'url':
            result = self._predict(data.data)

        return result

    def _predict(self, image):
        '''
        Helper function that wraps the prediction and post-processing.
        '''

        image_p = Image(image)
        return self.process_prediction(image_p.raw, self.model.predict(image_p.data))

    def process_prediction(self, image, prediction):
        '''
        Takes the raw image and the predicted mask to crop the background.
        '''

        return cv2.bitwise_and(image, image, mask = prediction[0][:, :, 0].astype('uint8'))

    def visualize(self, result):
        '''
        Displays the predicted cropped image. 
        NOTE: Ideal for single file mode only.
        '''

        if isinstance(result, list):
            for res in result:
                self._visualize(res)
        else:
            self._visualize(result)

    def _visualize(self, result):
        '''
        Helper function for visualize. Displays the final image.
        '''

        fig = plt.figure()
        ax = plt.axes()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(result)
        plt.show()

    def write_to_file(self, result, dir):
        '''
        Saves the final image to a file.
        NOTE: The file is saved in the results directory with the same file as the input file.
        '''

        if isinstance(result, list):
            for res, file in zip(result, os.listdir(dir)):
                self._write_to_file(res, file)
        else:
            self._write_to_file(result, dir[-1])

    def _write_to_file(self, result, file):
        '''
        Helper function for the write to file operation.
        '''
        
        if not os.path.exists('results'):
            os.makedirs('results')

        cv2.imwrite(os.path.join('results', file), cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        logging.info('Image saved as: {}'.format(os.path.join('results', file)))

    def process_input(self, mode, data):
        '''
        Processes the command line input.
        '''
        
        return Input(mode, data)


class Image:
    '''
    Class containing image related operations. Prepares the input to be fed into the model for prediction.
    '''

    def __init__(self, data, shape = [256, 256, 3]):

        self.shape = shape
        self.raw = None
        self.data = self.prepare_image(data)

    def prepare_image(self, image):
        '''
        Prepares the image by reshaping and modified into a vector.
        '''
        
        if len(image.shape) < 3:
            image = self.grey2rgb(image)

        res_img = np.array(imresize(image, self.shape))
        self.raw = res_img
        res = np.zeros(shape = [1] + self.shape)
        res[0] = res_img

        logging.info('Image processed for prediction')
        return res

    def grey2rgb(self, img):
        '''
        Converts a greyscale image to RGB by duplicating the image to create 3 channels.
        This function ensures that the input passed to the model retains the RGB structure.
        '''

        res = []
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                res.append(list(img[x][y]) * 3)

        return np.array(res).reshape(img.shape[0], img.shape[1], 3)


class Input:
    '''
    Class to manage and process user input.
    '''

    def __init__(self, mode = None, data = None):

        self.modes = {
            'dir' : self.process_dir,
            'file' : self.process_file,
            'url' : self.process_url,
        }

        self.data = None
        self.mode = None
        self.files = list()
        if mode and data:
            self.data = self.modes.get(mode)(data)
            self.mode = mode

    def process(self, mode, data):
        '''
        Processes the user data with the corresponding mode.
        '''

        self.mode = mode
        self.data = self.data = self.modes.get(mode)(data)

    def process_file(self, file):
        '''
        Processes single file inputs.
        '''
        
        assert os.path.isfile(file)

        self.files.append(ntpath.basename(file))
        return img_to_array(load_img(file)).astype('uint8')

    def process_dir(self, dir):
        '''
        Processes directory inputs.
        '''
        
        assert os.path.isdir(dir)

        result = list()
        for file in os.listdir(dir):
            result.append(self.process_file(os.path.join(dir, file)))

        return np.array(result)

    def process_url(self, url):
        '''
        Processes URL inputs.
        NOTE: The expected URL is a direct link to an image.
        '''
        
        self.files.append('URLSample.png')
        return io.imread(url)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Remove image backgrounds using u-net')
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument('--file', '-f', type=str, default=None, help='Full path of the image file')
    group.add_argument('--url', '-u', type=str, default=None, help='Direct URL of the image file')
    group.add_argument('--dir', '-d', type=str, default=None, help='Directory containing image files')
    parser.add_argument('--mode', type=str, choices=['people', 'dog', 'cat', 'all'], default='all', help='Model to use for prediction')
    parser.add_argument('--summary', action='store_true', default=False, help='Displays summary of the trained model')
    parser.add_argument('--visualize', action='store_true', default=False, help='Displays the final prediction')
    parser.add_argument('--write-to-file', action='store_true', default=False, help='Saves the final image to a file in the results directory')

    args = parser.parse_args()

    # Initialize the model
    model = Classify(mode = args.mode)

    # Parse the input
    if args.file:
        data = Input(mode = 'file', data = args.file)
    if args.dir:
        data = Input(mode = 'dir', data = args.dir)
    if args.url:
        data = Input(mode = 'url', data = args.url)

    # Perform prediction
    prediction = model.predict(data)

    if args.write_to_file:
        model.write_to_file(prediction, dir = data.files)

    if args.visualize:
        model.visualize(prediction)
    
    if args.summary:
        print(model.model.summary())
