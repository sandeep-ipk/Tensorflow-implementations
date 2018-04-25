# python dependencies
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import decimate
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Keras, tf dependencies
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPool1D, GlobalAvgPool1D, Dropout, BatchNormalization, Dense, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.utils import np_utils
from keras.regularizers import l2

INPUT_LIB = '../project6/'
SAMPLE_RATE = 44100
CLASSES1 = ['extrastole','normal', 'murmur']
CLASSES = ['normal', 'murmur']
CODE_BOOK = {x:i for i,x in enumerate(CLASSES)}
NB_CLASSES = len(CLASSES)


# loading wav files
def load_wav_file(name, path):
    _, b = wavfile.read(path + name)
    #assert _ == SAMPLE_RATE
    return b


# this function helps in making all the wav files of equal length.
def repeat_to_length(arr, length):
    # Repeats the numpy 1D array to given length, and makes datatype float
    result = np.empty((length, ), dtype = 'float32')
    l = len(arr)
    pos = 0
    while pos + l <= length:
        result[pos:pos+l] = arr
        pos += l
    if pos < length:
        result[pos:length] = arr[:length-pos]
    return result


# creating dataframes using pandas for train set
df_train = pd.read_csv(INPUT_LIB + 'train_data1.csv')

df_train['time_series'] = df_train['fname'].apply(load_wav_file, path=INPUT_LIB +'train_data1/')
df_train['len_series'] = df_train['time_series'].apply(len)

MAX_LEN = max(df_train['len_series']) # will be used for repeat_to_length()
df_train['time_series'] = df_train['time_series'].apply(repeat_to_length, length=MAX_LEN)
df_train['classes'] = [CODE_BOOK[label] for label in df_train['label']]

x_train = np.stack(df_train['time_series'].values, axis=0)
y_train = np.array(df_train['classes'], dtype='int')


# creating dataframes using pandas for test set
df_test = pd.read_csv(INPUT_LIB + 'test_data1.csv')

df_test['time_series'] = df_test['fname'].apply(load_wav_file, path=INPUT_LIB +'test_data1/')
df_test['len_series'] = df_test['time_series'].apply(len)

MAX_LEN = max(df_train['len_series']) # will be used for repeat_to_length()
df_test['time_series'] = df_test['time_series'].apply(repeat_to_length, length=MAX_LEN)
df_test['classes'] = [CODE_BOOK[label] for label in df_test['label']]

x_test = np.stack(df_test['time_series'].values, axis=0)
y_test = np.array(df_test['classes'], dtype='int')

y_test_ = y_test
y_train_ = y_train
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#print(len(x_test), len(x_train), len(y_test), len(y_train))

x_train = decimate(x_train, 8, axis=1, zero_phase=True)
#x_train = decimate(x_train, 8, axis=1, zero_phase=True)
#x_train = decimate(x_train, 4, axis=1, zero_phase=True)
x_test = decimate(x_test, 8, axis=1, zero_phase=True)
#x_test = decimate(x_test, 8, axis=1, zero_phase=True)
#x_test = decimate(x_test, 4, axis=1, zero_phase=True)

# Scale each observation to unit variance, it should already have mean close to zero.
x_train_ = x_train / np.std(x_train, axis=1).reshape(-1,1)
x_test_ = x_test / np.std(x_test, axis=1).reshape(-1,1)

x_train = x_train_[:,:,np.newaxis]
x_test = x_test_[:,:,np.newaxis]

dropout = 0.5


class Model:

    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=4, kernel_size=9, padding='same',
                        input_shape = x_train.shape[1:],
                        kernel_regularizer = l2(0.025))) # , kernel_initializer='glorot_normal'
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(MaxPool1D(strides=4))
        self.model.add(BatchNormalization())

        self.model.add(Conv1D(filters=8, kernel_size=9, padding='same',
                        kernel_regularizer = l2(0.05)))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(MaxPool1D(strides=4))
        self.model.add(BatchNormalization())

        self.model.add(Conv1D(filters=8, kernel_size=9, padding='same',
                         kernel_regularizer = l2(0.1)))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(MaxPool1D(strides=4))
        self.model.add(BatchNormalization())

        self.model.add(Conv1D(filters=16, kernel_size=7, padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(MaxPool1D(strides=4))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout/2))

        self.model.add(Conv1D(filters=16, kernel_size=7, padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(MaxPool1D(strides=4))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout/2))

        self.model.add(Conv1D(filters=32, kernel_size=4, padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(MaxPool1D(strides=4))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout))

        self.model.add(Conv1D(filters=32, kernel_size=4, padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(MaxPool1D(strides=4))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout))

        self.model.add(Conv1D(filters=64, kernel_size=1, padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.75))
        self.model.add(GlobalAvgPool1D())

        self.model.add(Dense(NB_CLASSES, activation='softmax'))


class Svm:
    def __init__(self):
        self.clf = SVC(kernel='linear', max_iter=1000)


class LR:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)


def batch_generator(x_train, y_train, batch_size):
    """
    Rotates the time series randomly in time
    """
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')
    full_idx = range(x_train.shape[0])

    while True:
        batch_idx = np.random.choice(full_idx, batch_size)
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]

        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis=0)

        yield x_batch, y_batch


def train_batch_n(model_, epoch=25, steps_per_epoch=500, batch_size=16, save=False):

    weight_saver = ModelCheckpoint('set_weights.h5', monitor='val_loss',
                                   save_best_only=True, save_weights_only=True)

    model_.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8**x)

    hist = model_.fit_generator(batch_generator(x_train, y_train, batch_size),
                       epochs=epoch, steps_per_epoch=steps_per_epoch,
                       validation_data=(x_test, y_test),
                       callbacks=[weight_saver, annealer],
                       verbose=2)

    if save:
        file_name = 'sep_3_model_' + str(epoch) + '_' + str(steps_per_epoch) + '_' + str(1) + '.h5'
        model_.save(filepath='../project6/models/' + file_name)

    return model_, hist


def train_ensemble(model_=None, num_networks=5, epoch=25, steps_per_epoch=500, batch_size=16, save=False):
    ensemble_ = []
    if not model_:
        for i in range(num_networks):
            model_ = Model()
            model_i, _ = train_batch_n(model_=model_.model, epoch=epoch, steps_per_epoch=steps_per_epoch, batch_size=batch_size)
            file_name = 'sep_ensemble_model_' + str(epoch) + '_' + str(steps_per_epoch) + '_' + str(i+1) + '.h5'
            if save:
                model_i.save(filepath='../project6/models/' + file_name)

            print('trained model: ', i + 1)
            print()

            ensemble_.append(model_i)

    print()
    return ensemble_


def predict_from_ensemble(ensemble_, confusion=False):
    predictions = []
    for i in range(len(ensemble_)):
        prediction = ensemble_[i].predict(x_test)
        predictions.append(prediction)

    mean_prediction = np.mean(predictions, axis=0)
    ensemble_cls_pred = np.argmax(mean_prediction, axis=1)
    model_metrics(cls_pred=ensemble_cls_pred, type=1, confusion=confusion)


def model_metrics(model=None, hist=None, cls_pred=None, type=1, confusion=False):
    if model is not None:
        if type == 1:
            prediction = model.predict(x_test)
            cls_pred = np.argmax(prediction, axis=1)

        elif type == 0:
            prediction = model.predict(x_test_)
            cls_pred = np.around(prediction, decimals=1)
            #cls_pred = 1 - cls_pred
            print(prediction[:10])


    correct = (cls_pred == y_test_)
    incorrect = np.logical_not(correct)
    total_correct = np.sum(correct)

    precision, recall, _, _ = precision_recall_fscore_support(y_test_, cls_pred, average='binary')
    f1_score = 2*(precision*recall)/(precision+recall)
    print()
    print()
    print('accuracy, precision, recall, f1_score: ', total_correct / len(y_test_), precision, recall, f1_score)

    if hist is not None:
        plt.plot(hist.history['loss'], color='b', label='train loss')
        plt.plot(hist.history['val_loss'], color='r', label='test loss')
        plt.xlabel('num epochs')
        plt.ylabel('loss')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.show()

        plt.plot(hist.history['acc'], color='b', label='train acc')
        plt.plot(hist.history['val_acc'], color='r', label='test acc')
        plt.xlabel('num epochs')
        plt.ylabel('accuracy')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.show()
    if confusion:
        plot_confusion_matrix(cls_pred=cls_pred)


def plot_confusion_matrix(cls_pred):

    cls_true = y_test_
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    print(cm)
    plt.matshow(cm)

    plt.colorbar()
    tick_marks = np.arange(NB_CLASSES)
    plt.xticks(tick_marks, range(NB_CLASSES))
    plt.yticks(tick_marks, range(NB_CLASSES))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()

'''
model1 = load_model('model3.h5')
loss, acc = model1.evaluate(x_test, y_test, batch_size=32)
print(acc)
'''

#model_ = Model().model
#model_ = load_model('../project6/models/'+'3_model_30_500_1.h5')
#model_, hist = train_batch_n(model_=model_, epoch=30, steps_per_epoch=500, save=True)
#model_metrics(model=model_, hist=hist, type=1, confusion=True)
ensemble = train_ensemble(epoch=30, steps_per_epoch=500, save=True)
#ensemble = [load_model('../project6/models/'+'ensemble_model_30_500_' + str(i+1) + '.h5') for i in range(5)]
predict_from_ensemble(ensemble_=ensemble, confusion=True)


'''
svm = Svm().clf
svm.fit(x_train_, y_train_)
lr = LR().clf
lr.fit(x_train_, y_train_)
model_metrics(model=svm, type=0)
'''

'''
model1 = load_model('../project6/models/'+'3_model_20_500_1.h5')
model1.summary()
'''
