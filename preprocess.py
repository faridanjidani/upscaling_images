import tensorflow as tf


import random
import os
import math
import numpy as np
import tensorflow_datasets as tfds
# import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
# from IPython.display import display
import matplotlib.pyplot as plt
import keras.backend as K
from scipy.io import loadmat, savemat

with tf.device('/device:GPU:1'):
    def get_dataset(batch_size, split):
        dataset = tfds.load(name='div2k',data_dir='./dataset')

        return dataset
    def gaussian_blur(img, kernel_size=11, sigma=5):
        def gauss_kernel(channels, kernel_size, sigma):
            ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
            xx, yy = tf.meshgrid(ax, ax)
            kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
            kernel = kernel / tf.reduce_sum(kernel)
            kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
            return kernel

        gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
        gaussian_kernel = gaussian_kernel[..., tf.newaxis]

        return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                    padding='SAME', data_format='NHWC')

    def process_input(input, input_size_x, input_size_y, network_type):
        input = tf.image.rgb_to_grayscale(input)


        # input = tf.squeeze(input,3)
        print("process1:",input.shape) 

        # input3 = tf.image.random_jpeg_quality(input[0], 80, 95)
        # gamma = np.random.choice([1,5,10,20])
        # gauss_kernel = gaussian_kernel( 3, 0, gamma )
        # input = tf.expand_dims(
        #     input, 0, name=None
        # )
        # input /= 255.0
        # print("process2:",input.shape) 
        # input = gaussian_blur(input, 3, gamma)
        # input = tf.cast(input, tf.float32)
        # input = tf.squeeze(input,0)

        print("process3:",input.shape) 
        if (network_type == 2):
            input1 = tf.image.resize(input, [input_size_x, input_size_y], method="area")
            input2 = tf.image.resize(input, [input_size_x, input_size_y], antialias=True, method="bicubic")
        elif (network_type == 1):
            input1 = tf.image.resize(input, [input_size_x*3, input_size_y], method="area")
            input2 = tf.image.resize(input, [input_size_x*3, input_size_y], antialias=True, method="bicubic")


        # input = tf.cast(input, tf.int8)
        

        
        return input2
        # return tf.divide(tf.add (input1, input2),2)
        # return tf.image.resize(input, [input_size_x, input_size_y], antialias=True, method="bicubic")


    def process_target(input, crop_size_x, crop_size_y):
        input = tf.image.rgb_to_grayscale(input)
        # input = tf.cast(input, tf.float32)
        # input /= 255.0
        # input = tf.image.rgb_to_yuv(input)
        # last_dimension_axis = len(input.shape) - 1
        # y, u, v = tf.split(input, 3, axis=last_dimension_axis)
        # return y
        return input
    def run(network_type):

        seed_value= 0

        # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
        os.environ['PYTHONHASHSEED']=str(seed_value)

        # 2. Set the `python` built-in pseudo-random generator at a fixed value

        random.seed(seed_value)

        # 3. Set the `numpy` pseudo-random generator at a fixed value

        np.random.seed(seed_value)

        # 4. Set the `tensorflow` pseudo-random generator at a fixed value

        tf.random.set_seed(seed_value)
        crop_size_x = 1200
        crop_size_y = 300
        upscale_factor = 3
        input_size_x = crop_size_x // upscale_factor
        input_size_y = crop_size_y // upscale_factor

        batch_size = 4
        dataset = get_dataset(batch_size, 'train')



        train_dataset = dataset['train']
        valid_ds = dataset['validation']
        matrix3 = loadmat('./1pwenvelop.mat')
        das = matrix3["envelope_dasIQdata"] / np.max(matrix3["envelope_dasIQdata"])
        das = das.astype("float32")
        if (network_type == 2):
            input_das =das[0:1777:3, 0:385:3]
        elif (network_type == 1):
            input_das =das[:, 0:385:3]
        input_das = np.expand_dims(input_das, axis=0)
        print(input_das.shape)
        input_das = input_das / np.max(input_das) *255.0
        matrix3 = loadmat("./75pw_envelop")
        das = matrix3["envelope_dasIQdata"] / np.max(matrix3["envelope_dasIQdata"])
        das = das.astype("float32")
        out_das = np.expand_dims(das, axis=0)
        out_das = tf.expand_dims(out_das, axis=3)
        out_das = out_das / np.max(out_das)
        print(out_das.shape)
        def resize(input, crop_size_x, crop_size_y):
            input = input['hr']
            return tf.image.resize(input, [crop_size_x, crop_size_y],antialias=True, method="area")

        train_dataset = train_dataset.map(
        lambda x: (resize(x,crop_size_x, crop_size_y))
        )
        valid_ds = valid_ds.map(
            lambda x: resize(x,crop_size_x, crop_size_y)
        )
        train_dataset = train_dataset.batch(4)
        valid_ds = valid_ds.batch(4)
        for i in train_dataset.take(1):
            for j in i:
            #     plt.figure()
                print("max:", np.max(j.numpy()))
            #     # k= k+1
            #     plt.imshow(j.numpy())
        train_dataset = train_dataset.map(
        lambda x: (process_input(x, input_size_x, input_size_y, network_type), process_target(x, crop_size_x, crop_size_y))
        )


        valid_ds = valid_ds.map(
        lambda x: (process_input(x, input_size_x, input_size_y, network_type), process_target(x,crop_size_x, crop_size_y))
        )

        for i in train_dataset.take(1):
            for j in i[0]:
            #     plt.figure()
                print("max0:", np.max(j.numpy()))
            for j in i[1]:
            #     plt.figure()
                print("max1:", np.max(j.numpy()))
            #     # k= k+1
            #     plt.imshow(j.numpy())


        return train_dataset, valid_ds, input_das, out_das