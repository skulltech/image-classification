import argparse
import os
import sys
import time
import datetime

import numpy as np
import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import keras
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Activation, Input, ZeroPadding2D, Concatenate, AveragePooling2D, GlobalAveragePooling2D, Dropout, Multiply, Add, multiply
from keras.models import Sequential, Model, load_model
from keras.optimizers import rmsprop, adam
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from keras import regularizers
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator


TTL = 6000
class LimitTrainingTime(Callback):
    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, batch, logs={}):
        time_elapsed = time.time() - self.start_time
        hr = str(datetime.timedelta(seconds=time_elapsed))
        print(f'[*] Time elapsed: {hr}')
        if time_elapsed > TTL:
            self.model.stop_training = True


def senet_layer(x, nb_channels, ratio):
    xd = GlobalAveragePooling2D()(x)
    xd = Dense(int(nb_channels / ratio), activation='relu')(xd)
    xd = Dense(nb_channels, activation='sigmoid')(xd)
    return multiply([x, xd])



def densenet_layer(x, nb_channels):
    x = BatchNormalization(gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(x)
    return x

def transition_layer(x, nb_channels):
    x = BatchNormalization(gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(nb_channels, (1, 1), padding='same', use_bias=False, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

def densenet_block(x, nb_channels, growth_rate, nb_layers):
    x_list = [x]
    for _ in range(nb_layers):
        x = densenet_layer(x, nb_channels)
        x_list.append(x)
        x = Concatenate()(x_list)
        nb_channels += growth_rate
    return x, nb_channels

def densenet(input_shape, growth_rate=12, dense_blocks=3, dense_layers=12):
    nb_channels = growth_rate * 2

    inputs = Input(shape=input_shape)
    x = Conv2D(nb_channels, (3,3), padding='same', strides=(1,1), use_bias=False, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(inputs)

    for i in range(dense_blocks):
        x, nb_channels = densenet_block(x, nb_channels=nb_channels, growth_rate=growth_rate, nb_layers=dense_layers)
        if i < dense_blocks - 1:
            x = transition_layer(x, nb_channels)
            x = senet_layer(x, nb_channels, 0.3)
    
    x = BatchNormalization(gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation='softmax', kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4), kernel_initializer='he_normal', bias_initializer='he_normal')(x)

    return Model(inputs, x, name='DenseNet')



def standard_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', input_shape=input_shape, kernel_initializer='he_normal', bias_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'))
    model.add(Dropout(0.25))
    model.add(Dense(100, kernel_initializer='he_normal', bias_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model


# Resnet

def identity_block(X, f, F1, F2, F3):
    X_shortcut = X
    X = Dropout(0.3)(X)
    X = Conv2D(F1, (1, 1), strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Dropout(0.3)(X)
    X = Conv2D(F2, (f, f), strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Dropout(0.3)(X)
    X = Conv2D(F3, (1, 1), strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)

    X = Add()([X, X_shortcut])

    X = Activation('relu')(X)

    return X

def convolution_block(X, f, F1, F2, F3, s=2):
    X_shortcut = X

    X = Dropout(0.3)(X)
    X = Conv2D(F1, (1, 1), strides=(s, s), padding="valid")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Dropout(0.3)(X)
    X = Conv2D(F2, (f, f), strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Dropout(0.3)(X)
    X = Conv2D(F3, (1, 1), strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)

    X_shortcut = Dropout(0.3)(X_shortcut)
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), padding="valid")(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(64, (7, 7), strides = (1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolution_block(X, 3, 64, 64, 256, s=1)
    X = identity_block(X, 3, 64, 64, 256)
    X = identity_block(X, 3, 64, 64, 256)

    X = convolution_block(X, 3, 128, 128, 512, s=2)
    X = identity_block(X, 3, 128, 128, 512)
    X = identity_block(X, 3, 128, 128, 512)

    X = convolution_block(X, 3, 256, 256, 1024, s=2)
    X = identity_block(X, 3, 256, 256, 1024)
    X = identity_block(X, 3, 256, 256, 1024)
    X = identity_block(X, 3, 256, 256, 1024)
    X = identity_block(X, 3, 256, 256, 1024)
    X = identity_block(X, 3, 256, 256, 1024)
    X = identity_block(X, 3, 256, 256, 1024)

    X = convolution_block(X, 3, 512, 512, 2048, s=2)
    X = identity_block(X, 3, 512, 512, 2048)
    X = identity_block(X, 3, 512, 512, 2048)
    
    X = AveragePooling2D(pool_size=(2,2))(X)

    X = Flatten(X)
    X = Dense(100, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X)

    return model

def claim_90():
    model = Sequential()
    weight_decay = 1e-4
    model.add(Conv2D(32, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    model.add(Conv2D(32, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    model.add(Conv2D(32, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    # model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())

    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(BatchNormalization())

    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model



def keras_cnn(args):
    with open(args.trainfile) as f:
        train = np.loadtxt(f, delimiter=' ')

    x = train[:, :-2]
    y = train[:, -1:]
    x = np.reshape(x, (x.shape[0], 3, 32, 32))
    x = np.swapaxes(x, 1, 2)
    x = np.swapaxes(x, 2, 3)
    lbl = LabelBinarizer()
    y = lbl.fit_transform(y)

    x, x_valid, y, y_valid = x[:45000, :], x[45000:, :], y[:45000, :], y[45000:, :]

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True)
    datagen.fit(x)

    # model = claim_90()
    model = densenet(x.shape[1:], dense_layers=4, growth_rate=8)

    # opt = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=True)
    opt = adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    lmt = LimitTrainingTime()
    es = EarlyStopping(monitor='val_acc', patience=10)
    mc = ModelCheckpoint('checkpoint_4', monitor='val_acc', save_best_only=True)
    model.fit_generator(datagen.flow(x, y, batch_size=128), validation_data=(x_valid, y_valid), callbacks=[lmt, es, mc], steps_per_epoch=int(len(x) / 32), epochs=100)
    # model.fit(x, y, validation_split=0.1, epochs=100, batch_size=128, callbacks=[lmt, es, mc])
    model = load_model('checkpoint')
    
    with open(args.testfile) as f:
        test = np.loadtxt(f, delimiter=' ')
    x = test[:, :-2]
    x = np.reshape(x, (x.shape[0], 3, 32, 32))
    x = np.swapaxes(x, 1, 2)
    x = np.swapaxes(x, 2, 3)
        
    probs = model.predict(x)
    preds = np.argmax(probs, axis=1)
    np.savetxt(args.outputfile, preds, fmt='%i')



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('trainfile', type=str)
    parser.add_argument('testfile', type=str)
    parser.add_argument('outputfile', type=str)
    parser.set_defaults(func=keras_cnn)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    args.func(args)



if __name__=='__main__':
    main()
