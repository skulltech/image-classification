import argparse
import datetime
import sys
import time

import numpy as np
from keras import regularizers
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Activation, Input, ZeroPadding2D, \
    Concatenate, AveragePooling2D, GlobalAveragePooling2D, Dropout, Add, multiply
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.preprocessing import LabelBinarizer

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
    x = Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4),
               kernel_initializer='he_normal')(x)
    return x


def transition_layer(x, nb_channels):
    x = BatchNormalization(gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(nb_channels, (1, 1), padding='same', use_bias=False, kernel_regularizer=l2(1e-4),
               kernel_initializer='he_normal')(x)
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
    x = Conv2D(nb_channels, (3, 3), padding='same', strides=(1, 1), use_bias=False, kernel_regularizer=l2(1e-4),
               kernel_initializer='he_normal')(inputs)

    for i in range(dense_blocks):
        x, nb_channels = densenet_block(x, nb_channels=nb_channels, growth_rate=growth_rate, nb_layers=dense_layers)
        if i < dense_blocks - 1:
            x = transition_layer(x, nb_channels)
            x = senet_layer(x, nb_channels, 0.3)

    x = BatchNormalization(gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation='softmax', kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4),
              kernel_initializer='he_normal', bias_initializer='he_normal')(x)

    return Model(inputs, x, name='DenseNet')


def standard_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', input_shape=input_shape,
                     kernel_initializer='he_normal', bias_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal',
                     bias_initializer='he_normal'))
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
    X = Conv2D(64, (7, 7), strides=(1, 1))(X)
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

    X = AveragePooling2D(pool_size=(2, 2))(X)
    X = Flatten(X)
    X = Dense(100, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X)
    return model


def claim_90():
    model = Sequential()
    weight_decay = 1e-4
    model.add(
        Conv2D(32, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay),
               input_shape=(32, 32, 3)))
    model.add(BatchNormalization())

    model.add(
        Conv2D(32, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())

    model.add(
        Conv2D(32, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(
        Conv2D(64, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())

    model.add(
        Conv2D(64, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())

    model.add(
        Conv2D(64, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(128, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())

    model.add(
        Conv2D(128, kernel_size=3, padding="same", activation="elu", kernel_regularizer=regularizers.l2(weight_decay)))
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


def main_block(x, filters, n, strides, dropout):
    x_res = Conv2D(filters, (3, 3), strides=strides, padding="same", kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(filters, (1, 1), strides=strides)(x)

    x_res = BatchNormalization()(x_res)
    x_res = Activation('relu')(x_res)

    x_res = Conv2D(filters, (3, 3), padding="same")(x_res)
    x = Add()([x_res, x])

    for i in range(n - 1):
        x_res = BatchNormalization()(x)
        x_res = Activation('relu')(x_res)
        x_res = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(5e-4))(x_res)
        if dropout:
            x_res = Dropout(dropout)(x_res)

        x_res = BatchNormalization()(x_res)
        x_res = Activation('relu')(x_res)
        x_res = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(5e-4))(x_res)
        x = Add()([x, x_res])

    return x


def wide_resnet(input_dims, output_dim, n, k, act="relu", dropout=None):
    n = (n - 4) // 6
    inputs = Input(input_dims)

    x = Conv2D(16, (3, 3), padding="same", kernel_regularizer=l2(5e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = main_block(x, 16 * k, n, (1, 1), dropout)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = main_block(x, 32 * k, n, (2, 2), dropout)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = main_block(x, 64 * k, n, (2, 2), dropout)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    outputs = Dense(output_dim, activation="softmax")(x)

    model = Model(inputs, outputs)
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

    x_train, x_valid = x[:45000, :], x[45000:, :]
    y_train, y_valid = y[:45000, :], y[45000:, :]

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)

    datagen.fit(x_train)
    model = wide_resnet(x.shape[1:], 100, 22, 8)

    sgd = SGD(lr=0.1, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    lmt = LimitTrainingTime()
    mc = ModelCheckpoint('checkpoint', monitor='val_acc', save_best_only=True)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                        steps_per_epoch=x_train.shape[0] // 128,
                        epochs=200,
                        validation_data=(x_valid, y_valid), callbacks=[lmt, mc])

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


if __name__ == '__main__':
    main()
