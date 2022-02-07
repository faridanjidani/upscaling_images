import os
import math
import numpy as np
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import array_to_img
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# from IPython.display import display
import matplotlib.pyplot as plt
import keras.backend as K
from scipy.io import loadmat, savemat
import get_network as gn
import preprocess
import tensorflow as tf
import argparse
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


parser = argparse.ArgumentParser()
parser.add_argument('--network-type', type=str, required=True)
args = parser.parse_args()

network_type = args.network_type

model,info = gn.get_model( network_type ,upscale_factor=3, channels=1)
train_dataset, valid_ds, input_das, out_das = preprocess.run(info[0])
with tf.device('/device:GPU:1'):

    # global m_psnr= []
    # global m_ssim = []
    class ESPCNCallback(keras.callbacks.Callback):
      m_psnr= []
      m_ssim = []
      models = []
      def __init__(self):
          super(ESPCNCallback, self).__init__()

          # best_psnr =0


          # self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)

      # Store PSNR value in each epoch.
      def on_epoch_begin(self, epoch, logs=None):
          self.psnr = []

      def on_epoch_end(self, epoch, logs=None):
          print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
          # input_das = tf.expand_dims(input_das, axis=2)
          predicted_out = self.model.predict(input_das)
        #   if (info[0] ==1  ):
        #     predicted_out = tf.expand_dims(predicted_out, axis=3)
          if (info[0] ==2):
            predicted_out = np.delete(predicted_out, slice(1777, 1779), 1)

          #predicted_out = tf.expand_dims(predicted_out, axis=3)

          # print(predicted_out.shape)
          # print(out_das.shape)
          
          predicted_out = np.abs(predicted_out[0] / np.max(predicted_out)) # Based on the matlab file
        #   out_das = out_das / np.max(out_das)
          print("max:", np.max(predicted_out),np.min(predicted_out) ,predicted_out.shape )
          print("max:", np.max(out_das),np.min(out_das))
          ssim = tf.image.ssim(predicted_out, out_das, 1)
          psnr = tf.image.psnr(predicted_out, out_das, 1)
          self.m_psnr.append(psnr.numpy()[0])
          self.m_ssim.append(ssim.numpy()[0])
          self.models.append(self.model)
          print("psnr:", self.m_psnr)
          print("ssim:", self.m_ssim)
      def on_test_batch_end(self, batch, logs=None):
          self.psnr.append(10 * math.log10(1 / (logs["loss"])))

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
    # checkpoint_filepath = "gs://fariddata"



    checkpoint_path = os.path.join("./paper_model", info[1], "save_at_{epoch}")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=False,
    )
    model.summary()
    callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]
    loss_fn1 = keras.losses.MeanSquaredError()
    loss_fn2 = keras.losses.MeanAbsoluteError()
    loss_fn3 = keras.losses.MeanSquaredLogarithmicError()

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    epochs = 100
    # model.compile(
    #     optimizer=optimizer, loss=[loss_fn2, loss_fn3]  , loss_weights = [0.3, 0.7]
    # )

    model.compile(
        optimizer=optimizer, loss=loss_fn3
    )
    model.fit(
        train_dataset, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2
    )


    with open('./paper_model/'+info[1]+'m_psnr.txt', 'w') as f:
        for item in ESPCNCallback.m_psnr:
            f.write("%s\n" % item)
    with open('./paper_model/'+info[1]+'m_ssim.txt', 'w') as f:
        for item in ESPCNCallback.m_ssim:
            f.write("%s\n" % item)


