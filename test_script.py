import numpy as np
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import applications
import pickle
from PIL import Image
import os

directory = './'
top_model_weights_path = directory+'/bottleneck_fc_model.h5'
img_width, img_height, channels = 160, 160, 3

# build the VGG16 network
base_model = applications.mobilenet.MobileNet(
        input_shape=(img_width, img_height, channels), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', classes=3)

# load top model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(3, activation='sigmoid'))

top_model.load_weights(top_model_weights_path)

# combine base and top model
final_input = Input(shape=(160,160, 3))
x = base_model(final_input)
result = top_model(x)
final_model = Model(input=final_input, output=result)
print(final_model.summary())

# predict class of image
im_path = './data/test/nevus/'
img = Image.open(os.path.join(im_path,os.listdir(im_path)[0]))
img = img.resize((160,160))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
x /= 255
preds = final_model.predict(x)
print(preds)