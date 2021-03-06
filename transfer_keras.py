import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import pickle
# dimensions of our images.
img_width, img_height, channels = 160, 160, 3
train_y, validation_y, test_y = pickle.load(open('labels.pkl', 'rb'))

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = './data/train'
validation_data_dir = './data/validation'
test_data_dir = './data/test'

nb_train_samples = len(train_y)
nb_validation_samples = len(validation_y)
nb_test_samples = len(test_y)
epochs = 50
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=40)

    model = applications.mobilenet.MobileNet(
        input_shape=(img_width, img_height, channels), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', classes=3)

    # generator = datagen.flow_from_directory(
    #     train_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=True)
    # bottleneck_features_train = model.predict_generator(
    #     generator, nb_train_samples // batch_size)
    # np.save(open('bottleneck_features_train.npy', 'wb'),
    #         bottleneck_features_train)

    # generator = datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)
    # bottleneck_features_validation = model.predict_generator(
    #     generator, nb_validation_samples // batch_size)
    # np.save(open('bottleneck_features_validation.npy', 'wb'),
    #         bottleneck_features_validation)

    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_test = model.predict_generator(
        generator, len(test_y) // batch_size)
    np.save(open('bottleneck_features_test.npy', 'wb'),
            bottleneck_features_test)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_y,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_y))
    model.save_weights(top_model_weights_path)
    test_data = np.load(open('bottleneck_features_test.npy', 'rb'))
    model.evaluate_generator(test_data, 32)


def test_model():

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

# save_bottlebeck_features()
train_top_model()
# est_model()
