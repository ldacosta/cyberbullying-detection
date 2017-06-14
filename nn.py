import numpy as np
import keras
from keras.optimizers import SGD
from keras.layers import Flatten, Input, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
import warnings

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        elif current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

callbacks = [
    EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1),
    ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True),
    # ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
]

class CyberbullyingDetectionnNN(object):


    def __create_conv_plus_max_pooling__(self, a_size: int, inputs):
        # 2D 3x3 convolution followed by a maxpool
        a_branch =Conv2D(
                filters=100,
                kernel_size=(a_size, a_size),
                activation='relu',
                input_shape=(self.n_words_in_review, self.n_features_in_word, self.input_channels),
                strides=(1, 1),
                padding='same',
                name="conv2D_{}x{}".format(a_size, a_size))(inputs)
        a_branch = MaxPooling2D(pool_size=(2, 1), strides = (1,1), name="max_pooling_2x1_of_conv2D_{}x{}".format(a_size, a_size))(a_branch)
        a_branch = Dropout(0.25, name="dropout_0.25_of_max_pooling_2x1_of_conv2D_{}x{}".format(a_size, a_size))(a_branch)
        return a_branch


    def __init__(self, features_in_words, words_in_review):
        self.n_words_in_review = words_in_review
        self.n_features_in_word = features_in_words
        self.input_channels = 1
        # inputs
        inputs = Input(shape=(self.n_words_in_review, self.n_features_in_word, self.input_channels))
        # intermediate results
        # 2D 3x3 convolution followed by a maxpool
        branch1 = self.__create_conv_plus_max_pooling__(a_size = 1, inputs = inputs)
        branch2 = self.__create_conv_plus_max_pooling__(a_size = 2, inputs = inputs)
        branch3 = self.__create_conv_plus_max_pooling__(a_size = 3, inputs = inputs)
        # now let's go fully-connected to a 2-way classification:
        merged_branches = keras.layers.concatenate([branch1, branch2, branch3], axis = 1, name="merged_convolutions") # , axis=-1)
        merged_branches = Dropout(0.5, name="dropout_{}_of_merged_convolutions".format(0.5))(merged_branches)

        # ok. done
        categorization = Flatten()(merged_branches)
        categorization = Dense(2, activation='softmax')(categorization)

        # all good. Let's build the model, then
        self.model = Model(inputs=inputs, outputs=categorization)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer=sgd)

    def summary(self):
        return self.model.summary()

    def fit(self, x_train, y_train, batch_size, epochs):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split = 0.1, callbacks=callbacks)

    def evaluate(self, x, y, batch_size):
        self.model.load_weights()
        keras.engine.training.Model.load_weights()
        return self.model.evaluate(x, y, batch_size=batch_size)


def sanity_check():
    cnn_k = CyberbullyingDetectionnNN(features_in_words=300, words_in_review=10)
    cnn_k.summary()
    num_examples_training = 98
    x_train = np.random.random((num_examples_training, cnn_k.n_words_in_review, cnn_k.n_features_in_word, cnn_k.input_channels))
    y_train = keras.utils.to_categorical(np.random.randint(2, size=(num_examples_training, 1)), num_classes=2)
    num_examples_testing = 17
    x_test = np.random.random((num_examples_testing, cnn_k.n_words_in_review, cnn_k.n_features_in_word, cnn_k.input_channels))
    y_test = keras.utils.to_categorical(np.random.randint(2, size=(num_examples_testing, 1)), num_classes=2)
    cnn_k.fit(x_train, y_train, batch_size = 16, epochs = 3)
    score = cnn_k.evaluate(x_test, y_test, batch_size = 16)
    print("\n\n ====> score is {}".format(score))

if __name__ == "__main__":
    sanity_check()
