from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Input, GlobalMaxPooling2D, Reshape
from keras import backend as k
from tqdm import tqdm

from keras.applications.imagenet_utils import

import matplotlib.pyplot as plt
import cv2, numpy as np




#prebuilt model eith prebuilt weights on imagenet
input_tensor = Input(shape=(256, 1600, 3))
base_model = ResNet50(include_top=False)


base_model.summary()
model = base_model.output
model.add(Reshape((-1, 64, 64, 1)))

model = GlobalMaxPooling2D()(model)

model = Dense(1024, activation='relu')(model)

predictions = Dense(200, activation='sigmoid')(model)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model = model.load_weights()

for i, layer in enumerate(tqdm(base_model.layers)):
    print(i, layer.name)
# model.compile(optimizer=Adam(), loss='categorical_crossentropy')
# history = model.fit()
