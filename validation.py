import os
import math
import numpy as np
import tensorflow as tf
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
import get_network as gn
with tf.device('/device:GPU:0'):
   


    model,info = gn.get_model( "13k_1DF" ,upscale_factor=3, channels=1)

    model.summary()

    ## pasth to folder
    path = "./paper_model/13k_1DF"



def upscale_image(model, gray):
    """Predict the result based on input image and restore the image as RGB."""

    # y, cb, cr = ycbcr.split()
    gray = gray.astype("float32") 
    input = np.expand_dims(gray, axis=0)
    print(tf.shape(input))
    out = model.predict(input)
    # tf.clip_by_value(
    # out, -1.0, 1.0, name=None
    # )

    out_img_y = out[0]
    # out_img_y *= 255.0
    # out_img_y = out_img_y.clip(0, 1)


    # Restore the image in RGB color space.
    # out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    # out_img_y =np.uint8(out_img_y)
    # out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    # out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    # out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        # "RGB"
    # )
    print(out_img_y.shape)
    return out_img_y

# matrix1 = loadmat('/content/drive/MyDrive/real_imag/realpart.mat')
# matrix1 = loadmat('realpart.mat')
# lr1 = matrix1["r"]
# lr1 = lr1.astype(np.float32)




# if (info[0]== 1):
#     matrix3 = loadmat('env_width_type1.mat')
#     lr1 = matrix3["env"] / np.max(matrix3["env"]) *255.0
# elif (info[0]== 2):
#     matrix3 = loadmat('env_width_type1.mat')
#     # lr1 = matrix3["env_ds"] / np.max(matrix3["env_ds"]) *255.0
#     lr1 = matrix3["env"] / np.max(matrix3["env"]) *255.0
#     lr1 = lr1[0:1536:3,:]

# if (info[0]== 1):
#     matrix3 = loadmat('env_type2.mat')
#     lr1 = matrix3["IQdata_abs"] / np.max(matrix3["IQdata_abs"]) *255.0
# elif (info[0]== 2):
#     matrix3 = loadmat('env_type2.mat')
#     # lr1 = matrix3["env_ds"] / np.max(matrix3["env_ds"]) *255.0
#     lr1 = matrix3["IQdata_abs"] / np.max(matrix3["IQdata_abs"]) *255.0
#     lr1 = lr1[0:1536:3,:]

if (info[0]== 1):
    matrix3 = loadmat('./env_sim.mat')
    lr1 = matrix3["IQdata_abs"] / np.max(matrix3["IQdata_abs"]) * 255.0
    # predicted_out = np.delete(predicted_out, slice(1777, 1779), 1)
elif (info[0]== 2):
    matrix3 = loadmat('./env_sim.mat')
    lr1 = matrix3["IQdata_abs"] / np.max(matrix3["IQdata_abs"]) * 255.0
    lr1 = lr1[0:-1:3,:]


print("Maxpred ",np.max(lr1))
print("minpred ",np.min(lr1))
try:
    os.makedirs('matlab_files'+'/'+info[1])
except OSError as error: 
    print(error)  
for i in range (1, 101):
    model.load_weights(path+'/save_at_'+str(i))
    preds1 = upscale_image(model, lr1)
# preds2 = upscale_image(model, lr2)

# pred3 = upscale_image(model, env)

# final_output = preds1 + 1j * preds2
    final_output = preds1 /255.0

# plt.imshow(abs(final_output), cmap='gray')
    if(len(final_output[:,0])== 1776):
        final_output = np.append(final_output, [final_output[-1]], axis = 0)
    savemat('matlab_files/'+info[1]+'/sim_' +str(i)+'.mat', {"foutput": final_output})
    print("Maxpred ",np.max(final_output))
    print("minpred ",np.min(final_output))
    print(np.sum(final_output))