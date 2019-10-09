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
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Activation, Input, Concatenate, AveragePooling2D, GlobalAveragePooling2D, Dropout, Multiply
from keras.models import Sequential, Model, load_model
from keras.optimizers import rmsprop, adam
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from keras.regularizers import l2



TTL = 3000
class LimitTrainingTime(Callback):
    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, batch, logs={}):
        time_elapsed = time.time() - self.start_time
        hr = str(datetime.timedelta(seconds=time_elapsed))
        print(f'[*] Time elapsed: {hr}')
        if time_elapsed > TTL:
            self.model.stop_training = True



def densenet_layer(x, nb_channels):
    x = BatchNormalization(gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(x)
    return x

def transition_layer(x, nb_channels):
    x = BatchNormalization(gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
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



def senet_layer(x, nb_channels, ratio):
    xd = GlobalAveragePooling2D()(x)
    xd = Dense(nb_channels // ratio, activation='relu')
    xd = Dense(nb_channels, activation='sigmoid')
    return Multiply()([x, xd])



def keras_cnn(args):
    with open(args.trainfile) as f:
        train = np.loadtxt(f, delimiter=' ')

    x = train[:, :-2]
    y = train[:, -1:]
    x = np.reshape(x, (x.shape[0], 32, 32, 3))
    lbl = LabelBinarizer()
    y = lbl.fit_transform(y)

    model = standard_model(x.shape[1:])
    # model = densenet(x.shape[1:], dense_layers=4, growth_rate=8)

    opt = adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    lmt = LimitTrainingTime()
    es = EarlyStopping(monitor='val_acc', patience=3)
    mc = ModelCheckpoint('checkpoint', monitor='val_acc', save_best_only=True)
    model.fit(x, y, validation_split=0.1, epochs=100, batch_size=128, callbacks=[lmt, es, mc])
    model = load_model('checkpoint')
    
    with open(args.testfile) as f:
        test = np.loadtxt(f, delimiter=' ')
    x = test[:, :-2]
    x = np.reshape(x, (x.shape[0], 32, 32, 3))
    
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
