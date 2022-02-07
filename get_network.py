import os
import math
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
# from IPython.display import display
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

with tf.device('/device:GPU:1'):
    class dsubpixel(keras.layers.Layer):
        def __init__(self):
            super(dsubpixel, self).__init__()
            # self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

        def call(self, inputs):
            # self.total.assign_add(tf.reduce_sum(inputs, axis=0))
            bsize, a, b, c = inputs.get_shape().as_list()
            # print(bsize)
            bsize = K.shape(inputs)[0]
            a = K.shape(inputs)[1]
            b = K.shape(inputs)[2]
            c = K.shape(inputs)[3]
            # print(bsize,a,b,c) 
            self.total = K.reshape(inputs, [bsize, a, b*c ,1])
            # self.total = tf.transpose(inputs, perm=[0, 2, 1])
            # self.total = tf.zeros((4,4))
            return self.total

        # def get_config (self):

        #   config = super().get_config().copy()
        #    config.update({
        #         'input_dim': self.input_dim,
        #    })

    # with strategy.scope():
    # with tf.device('/device:GPU:0'):
    def squeeze_excite_block(tensor, ratio=16):
        init = tensor
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)

        se = layers.GlobalAveragePooling2D()(init)
        se = layers.Reshape(se_shape)(se)
        se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        if K.image_data_format() == 'channels_first':
            se = layers.Permute((3, 1, 2))(se)

        x = layers.multiply([init, se])
        return x
    def get_model(params = "5k", upscale_factor=3, channels=1):
        # conv_args = {
      #     "activation": "tanh",
      #     "kernel_initializer": "Orthogonal",
      #     "padding": "same",
      # }
        if(params == "5k"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))
            x = layers.SeparableConv2D(64, (5,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            x = layers.SeparableConv2D(32, (3,3), strides=(1, 1), padding='same', activation= "relu")(x)
            y = layers.SeparableConv2D(32, (3,3), strides=(1, 1), padding='same', activation= "relu")(x)
            z1 = layers.SeparableConv2D(channels * (upscale_factor**2), (3,3), padding='same',kernel_initializer = initializer)(y)
            outputs = tf.nn.depth_to_space(z1, 3)
            info = [2,"5k_2Df_paper"]

        elif(params == "13k"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))
            x = layers.SeparableConv2D(128, (5,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            x = layers.SeparableConv2D(64, (3,3), strides=(1, 1), padding='same', activation= "relu")(x)
            y = layers.SeparableConv2D(32, (3,3), strides=(1, 1), padding='same', activation= "relu")(x)
            z1 = layers.SeparableConv2D(channels * (upscale_factor**2), (3,3), padding='same',kernel_initializer = initializer)(y)
            outputs = tf.nn.depth_to_space(z1, 3)
            info = [2,"13k_2Df_paper"]
        elif(params == "98k"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))
            x = layers.Conv2D(128, (5,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            x = layers.Conv2D(64, (3,3), strides=(1, 1), padding='same', activation= "relu")(x)
            y = layers.Conv2D(32, (3,3), strides=(1, 1), padding='same', activation= "relu")(x)
            z1 = layers.Conv2D(channels * (upscale_factor**2), (3,3), padding='same',kernel_initializer = initializer)(y)
            outputs = tf.nn.depth_to_space(z1, 3)
            info = [2, "98k_2D_paper"]

        elif(params == "33k_1DF"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))
            x = layers.Conv2D(128, (1,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            x = layers.Conv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
            y = layers.Conv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
            z1 = layers.Conv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(y)
            # outputs = tf.nn.depth_to_space(z1, 3)
            outputs = dsubpixel()(z1)
            info = [1,"33k_1DF"]

        elif(params == "13k_1DF"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))
            x = layers.SeparableConv2D(128, (1,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            x = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
            y = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(y)
            # outputs = tf.nn.depth_to_space(z1, 3)
            outputs = dsubpixel()(z1)
            info = [1,"13k_1DF"]

        elif(params == "4k_1DF"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))
            x = layers.SeparableConv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            x = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
            y = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(y)
            # outputs = tf.nn.depth_to_space(z1, 3)
            outputs = dsubpixel()(z1)
            info = [1,"4k_1DF"]


        elif(params == "13k_1DRes"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))
            in_3 = layers.Concatenate()([inputs,inputs,inputs])
            x = layers.SeparableConv2D(128, (1,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            x = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
            y = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(y)
            # outputs = tf.nn.depth_to_space(z1, 3)
            z1 = layers.add([z1, in_3])
            outputs = dsubpixel()(z1)
            info = [1,"13k_1DRes"]

        elif(params == "13k_1DSqX"):
            # initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None) ## change the initilizer adding the drop out
            initializer = tf.keras.initializers.HeNormal(seed=None)
            inputs = keras.Input(shape=(None, None, channels))

            in_3 = layers.Concatenate()([inputs,inputs,inputs])
            x1 = layers.Conv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(inputs)
            x = layers.Dropout(.05)(x1)
            y1 = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(x1) 
            y1 = layers.Dropout(.05)(y1)
            y1 = squeeze_excite_block(y1)
            z1 = layers.add([y1, x1])

            z1 = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(z1)
            z1 = layers.Dropout(.05)(z1)
            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(z1)
            z1 = squeeze_excite_block(z1)
            # outputs = tf.nn.depth_to_space(z1, 3)
            s1 = layers.add([z1, in_3])
            outputs = dsubpixel()(s1)
            info = [1,"13k_1DSqX"]  

        elif(params == "13k_1DSqXNoRes"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))

            
            # in_3 = layers.Concatenate()([inputs,inputs,inputs])
            x1 = layers.Conv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            
            y1 = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu")(x1) 
            y1 = squeeze_excite_block(y1)
            z1 = layers.add([y1, x1])

            z1 = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(z1)

            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3),activation= "relu", padding='same',kernel_initializer = initializer)(z1)
            z1 = squeeze_excite_block(z1)
            outputs = dsubpixel()(z1)
            info = [1,"13k_1DSqXNoRes"]  

        elif(params == "4k_1DRes"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))
            in_3 = layers.Concatenate()([inputs,inputs,inputs])
            x = layers.SeparableConv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            x = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
            y = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(y)
            # outputs = tf.nn.depth_to_space(z1, 3)
            z1 = layers.add([z1, in_3])
            outputs = dsubpixel()(z1)
            info = [1,"4k_1DRes"]

        elif(params == "4k_1DSqX"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))
            in_3 = layers.Concatenate()([inputs,inputs,inputs])
            x = layers.SeparableConv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            x = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
            y = squeeze_excite_block(x)
            y = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(y)
            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(y)
            # outputs = tf.nn.depth_to_space(z1, 3)
            z1 = layers.add([z1, in_3])
            outputs = dsubpixel()(z1)
            info = [1,"4k_1DSqX"]    
        elif(params == "13k_transposed"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))

            
            in_3 = layers.Concatenate()([inputs,inputs,inputs])
            x1 = layers.Conv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            
            y1 = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu")(x1) 
            y1 = squeeze_excite_block(y1)
            z1 = layers.add([y1, x1])

            z1 = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(z1)

            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(z1)
            z1 = squeeze_excite_block(z1)
            # outputs = tf.nn.depth_to_space(z1, 3)
            s1 = layers.add([z1, in_3])
            s1 = dsubpixel()(s1)

            enhanced_in = layers.Conv2DTranspose(1, (1,3), strides=(1,3),padding='same')(inputs)
            x2 = layers.Conv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            
            y2 = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu")(x2) 
            y2 = squeeze_excite_block(y2)
            z2 = layers.add([y2, x2])

            z2 = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(z2)
            z2 = squeeze_excite_block(z2)
            z2 = layers.Conv2DTranspose(32, (1,3), strides=(1,upscale_factor),padding='same')(z2)
            z2 = layers.SeparableConv2D(channels, (1,3), padding='same',kernel_initializer = initializer)(z2)

            s2 = layers.add([z2, enhanced_in])
            outputs = layers.Average()([s1, s2])
            
            info = [1,"13k_transposed"]
        elif(params == "13k_transposed_2D"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))

            
            in_3 = layers.Concatenate()([inputs,inputs,inputs])
            in_9 = layers.Concatenate()([in_3,in_3,in_3])
            x1 = layers.Conv2D(64, (5,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            
            y1 = layers.SeparableConv2D(64, (3,3), strides=(1, 1), padding='same', activation= "relu")(x1) 
            y1 = squeeze_excite_block(y1)
            z1 = layers.add([y1, x1])

            z1 = layers.SeparableConv2D(32, (3,3), strides=(1, 1), padding='same', activation= "relu")(z1)

            z1 = layers.SeparableConv2D(channels * (upscale_factor **2), (3,3), padding='same',kernel_initializer = initializer)(z1)
            z1 = squeeze_excite_block(z1)

            s1 = layers.add([z1, in_9])
            s1 = tf.nn.depth_to_space(s1, 3)
            # s1 = dsubpixel()(s1)

            enhanced_in = layers.Conv2DTranspose(1, (3,3), strides=(3,3),padding='same')(inputs)
            x2 = layers.Conv2D(64, (5,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            
            y2 = layers.SeparableConv2D(64, (3,3), strides=(1, 1), padding='same', activation= "relu")(x2) 
            y2 = squeeze_excite_block(y2)
            z2 = layers.add([y2, x2])

            z2 = layers.SeparableConv2D(32, (3,3), strides=(1, 1), padding='same', activation= "relu")(z2)
            z2 = squeeze_excite_block(z2)
            z2 = layers.Conv2DTranspose(32, (3,3), strides=(upscale_factor,upscale_factor),padding='same')(z2)
            z2 = layers.SeparableConv2D(channels, (3,3), padding='same',kernel_initializer = initializer)(z2)

            s2 = layers.add([z2, enhanced_in])
            outputs = layers.Average()([s1, s2])
            
            info = [2,"13k_transposed_2D"]


        elif(params == "13k_transposed_only"):
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))

            
            # in_3 = layers.Concatenate()([inputs,inputs,inputs])
            # x1 = layers.Conv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            
            # y1 = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu")(x1) 
            # y1 = squeeze_excite_block(y1)
            # z1 = layers.add([y1, x1])

            # z1 = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(z1)

            # z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(z1)
            # z1 = squeeze_excite_block(z1)
            # # outputs = tf.nn.depth_to_space(z1, 3)
            # s1 = layers.add([z1, in_3])
            # s1 = dsubpixel()(s1)

            enhanced_in = layers.Conv2DTranspose(1, (1,3), strides=(1,3),padding='same')(inputs)
            x2 = layers.Conv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
            
            y2 = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu")(x2) 
            y2 = squeeze_excite_block(y2)
            z2 = layers.add([y2, x2])

            z2 = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(z2)
            z2 = squeeze_excite_block(z2)
            z2 = layers.Conv2DTranspose(32, (1,3), strides=(1,upscale_factor),padding='same')(z2)
            z2 = layers.SeparableConv2D(channels, (1,3), padding='same',kernel_initializer = initializer)(z2)

            outputs = layers.add([z2, enhanced_in])
            # outputs = layers.Average()([s1, s2])
            
            info = [1,"13k_transposed_only"]

        elif(params == "test1"):### different dropout
            # initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None) ## change the initilizer adding the drop out
            initializer = tf.keras.initializers.HeNormal(seed=None)
            inputs = keras.Input(shape=(None, None, channels))

            in_3 = layers.Concatenate()([inputs,inputs,inputs])
            x1 = layers.Conv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(inputs)
            x = layers.Dropout(.15)(x1)
            y1 = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(x1) 
            y1 = layers.Dropout(.15)(y1)
            y1 = squeeze_excite_block(y1)
            z1 = layers.add([y1, x1])

            z1 = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(z1)
            z1 = layers.Dropout(.15)(z1)
            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(z1)
            z1 = squeeze_excite_block(z1)
            # outputs = tf.nn.depth_to_space(z1, 3)
            s1 = layers.add([z1, in_3])
            outputs = dsubpixel()(s1)
            info = [1,"13k_1DSqX_t1"]  
            print(info[1])
        elif(params == "test2"): ## different initilizer
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None) ## change the initilizer adding the drop out
            # initializer = tf.keras.initializers.HeNormal(seed=None)
            inputs = keras.Input(shape=(None, None, channels))

            in_3 = layers.Concatenate()([inputs,inputs,inputs])
            x1 = layers.Conv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(inputs)
            x = layers.Dropout(.05)(x1)
            y1 = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(x1) 
            y1 = layers.Dropout(.05)(y1)
            y1 = squeeze_excite_block(y1)
            z1 = layers.add([y1, x1])

            z1 = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(z1)
            z1 = layers.Dropout(.05)(z1)
            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(z1)
            z1 = squeeze_excite_block(z1)
            # outputs = tf.nn.depth_to_space(z1, 3)
            s1 = layers.add([z1, in_3])
            outputs = dsubpixel()(s1)
            info = [1,"13k_1DSqX_t2"]
            print(info[1])
        elif(params == "test3"): ## no res
            # initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None) ## change the initilizer adding the drop out
            initializer = tf.keras.initializers.HeNormal(seed=None)
            inputs = keras.Input(shape=(None, None, channels))

            in_3 = layers.Concatenate()([inputs,inputs,inputs])
            x1 = layers.Conv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(inputs)
            x = layers.Dropout(.05)(x1)
            y1 = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(x1) 
            y1 = layers.Dropout(.05)(y1)
            y1 = squeeze_excite_block(y1)
            z1 = layers.add([y1, x1])

            z1 = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(z1)
            z1 = layers.Dropout(.05)(z1)
            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(z1)
            z1 = squeeze_excite_block(z1)
            # outputs = tf.nn.depth_to_space(z1, 3)
            # s1 = layers.add([z1, in_3])
            outputs = dsubpixel()(z1)
            info = [1,"13k_1DSqX_t3"]
            print(info[1])

        elif(params == "test4"): ## no res 0.15 dropout
            # initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None) ## change the initilizer adding the drop out
            initializer = tf.keras.initializers.HeNormal(seed=None)
            inputs = keras.Input(shape=(None, None, channels))

            in_3 = layers.Concatenate()([inputs,inputs,inputs])
            x1 = layers.Conv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(inputs)
            x = layers.Dropout(.15)(x1)
            y1 = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(x1) 
            y1 = layers.Dropout(.15)(y1)
            y1 = squeeze_excite_block(y1)
            z1 = layers.add([y1, x1])

            z1 = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(z1)
            z1 = layers.Dropout(.15)(z1)
            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(z1)
            z1 = squeeze_excite_block(z1)
            # outputs = tf.nn.depth_to_space(z1, 3)
            # s1 = layers.add([z1, in_3])
            outputs = dsubpixel()(z1)
            info = [1,"13k_1DSqX_t4_sqloss"]
            print(info[1])
        elif(params == "test5"): ## no res
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None) ## change the initilizer adding the drop out
            # initializer = tf.keras.initializers.HeNormal(seed=None)
            inputs = keras.Input(shape=(None, None, channels))

            in_3 = layers.Concatenate()([inputs,inputs,inputs])
            x1 = layers.Conv2D(64, (1,5), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(inputs)
            x = layers.Dropout(.15)(x1)
            y1 = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(x1) 
            y1 = layers.Dropout(.15)(y1)
            y1 = squeeze_excite_block(y1)
            z1 = layers.add([y1, x1])

            z1 = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(z1)
            z1 = layers.Dropout(.15)(z1)
            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(z1)
            z1 = squeeze_excite_block(z1)
            # outputs = tf.nn.depth_to_space(z1, 3)
            # s1 = layers.add([z1, in_3])
            outputs = dsubpixel()(z1)
            info = [1,"13k_1DSqX_t5sqloss"]
            print(info[1])

        elif(params == "test6"): ## no res
            # initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None) ## change the initilizer adding the drop out
            # initializer = tf.keras.initializers.HeNormal(seed=None)
            initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
            inputs = keras.Input(shape=(None, None, channels))
            x = layers.SeparableConv2D(128, (1,5), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(inputs)
            x = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(x)
            y = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu",kernel_initializer = initializer)(x)
            z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(y)
            # outputs = tf.nn.depth_to_space(z1, 3)
            outputs = dsubpixel()(z1)
            info = [1,"13k_1D_allorth"]
            print(info[1])
        # elif(params == "test7"): ## no res
        #     # initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None) ## change the initilizer adding the drop out
        #     # initializer = tf.keras.initializers.HeNormal(seed=None)
        #     initializer = tf.keras.initializers.Orthogonal(gain=2, seed=None)
        #     inputs = keras.Input(shape=(None, None, channels))
        #     x = layers.SeparableConv2D(128, (1,5), strides=(1, 1), padding='same', activation= "relu")(inputs)
        #     x = layers.SeparableConv2D(64, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
        #     y = layers.SeparableConv2D(32, (1,3), strides=(1, 1), padding='same', activation= "relu")(x)
        #     z1 = layers.SeparableConv2D(channels * (upscale_factor), (1,3), padding='same',kernel_initializer = initializer)(y)
        #     # outputs = tf.nn.depth_to_space(z1, 3)
        #     outputs = dsubpixel()(z1)
        #     info = [1,"13k_1D_allorth"]
        #     print(info[1])
        return keras.Model(inputs, outputs), info
