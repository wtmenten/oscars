import numpy as np

# np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Masking,Highway,Flatten,Reshape
from keras.layers.convolutional import Convolution1D,Convolution2D
from keras.layers.recurrent import LSTM
from keras.layers.pooling import MaxPooling1D
import pandas as pd
from random import random
import matplotlib.pyplot as plt
import pickle
import itertools

import os, sys, inspect
cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

base_hyperparams = dict(
    time_series_len = 20,
    l1_drop=0.1,
    l1_neurons = 50,
    l2_neurons = 50,
)


def _load_data():
    df = pd.read_csv("AA_4_computed.csv")
    print df.columns
    ## for single row version of model
    X = df[[5, 6, 8, 10, 17]].fillna(0).values.tolist()
    X = np.array(X).T
    print X[1][0]
    X[1] = np.log(X[1]+1)
    # X[2] = np.log(np.sqrt(100 - X[2] + 1) + .01)

    # X[3] = np.log(np.sqrt(10-X[3]+.01)+.01)
    # X[5] = np.log(np.sqrt(10-X[5]+.01)+.01)

    # X[4] = np.log(np.sqrt(100 - X[4] + 1) + .01)

    X = np.array([(row - row.min()) / (row.max() - row.min()) for row in X])
    train_set_x = X.T.tolist()
    train_set_y = df[[2]].values.tolist()
    return (train_set_x, train_set_y)

    ## for covolutional seasonal version
    # g = df.groupby(['year']).groups.items()
    # seasons = len(g)
    # maxSeasonlen = 12
    # np.zeros_like((seasons, maxSeasonlen))
    # out = (np.zeros((seasons, maxSeasonlen, 5)),np.zeros((seasons, maxSeasonlen, 1)))
    # # print out[0][1]
    # for index, (s, idxs) in enumerate(g):
    #     season = [df.fillna(0).iloc[i].values for i in idxs]
    #     newx = [[m[5],m[6],m[8],m[10],m[17],] for m in season]
    #     newy = [[m[2]] for m in season]
    #     out[0][index][:len(newx)] = newx
    #     out[1][index][:len(newx)] = newy
    # maxSeasonlen = max(map(len, out[1]))
    # # print out[0][1]
    # # print(maxSeasonlen)
    # return out


# TODO make generator for this
def _prepare_data(df, series_len):
    # print(df)
    x,y = df
    docX, docY = [], []
    for i in range(len(df[0]) - series_len):
        docX.append(df[0][i:i + series_len])
        docY.append(df[1][i + series_len])
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    assert len(df[0]) == len(df[1])

    ntrn = int(round(len(df[0]) * (1 - test_size)))
    X_train, y_train = (df[0][0:ntrn], df[1][0:ntrn])
    X_test, y_test = (df[0][ntrn:], df[1][ntrn:])
    return (X_train, y_train), (X_test, y_test)

def save_obj(obj, name ):
    with open('tests/pickels/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('tests/pickels/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def run_model(data, hyperParams):
    num_of_vectors = hyperParams["time_series_len"]  # time series length
    input_dim = hyperParams['input_dim']  # input length
    out_neurons = hyperParams['out_neurons']  # output length
    l1_neurons = hyperParams["l1_neurons"]
    l1_drop = hyperParams["l1_drop"]
    l2_neurons = hyperParams["l2_neurons"]
    model = Sequential()
    ## for single movie pred version
    model.add(Dense((100), input_dim=(input_dim)))
    model.add(Dropout(.3))
    model.add(Activation('relu'))
    model.add(Dense((100)))
    model.add(Dropout(.1))
    model.add(Activation('relu'))
    model.add(Dense((out_neurons)))
    model.add(Activation('hard_sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="Adam")

    # # for seasonal version
    # model.add(Convolution1D(1, 0, input_shape=(12, 5)))
    #
    # # model.add(MaxPooling1D(1, input_shape=(12,input_dim)))
    # # model.add(LSTM(l1_neurons, return_sequences=True, input_shape=(num_of_vectors, input_dim)))
    # # model.add(Dropout(l1_drop))
    # # model.add(Activation('tanh'))
    # # model.add(Flatten())
    # # model.add(Reshape((78,12)))
    # # model.add(LSTM(l1_neurons, return_sequences=False, go_backwards=True, input_shape=(78, 12, 1)))
    # # model.add(Activation('relu'))
    # model.add(Dense((12)))
    #
    # model.add(Dense((300)))
    # model.add(Dropout(.3))
    # model.add(Activation('relu'))
    # model.add(Dense((out_neurons)))
    # model.add(Activation('hard_sigmoid'))
    # model.compile(loss="categorical_crossentropy", optimizer="adam")


    # data2 = _prepare_data(data[:2], num_of_vectors)
    (X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
    # and now train the model
    # batch_size should be appropriate to your memory size
    # number of epochs should be higher for real world problems
    model.fit(X_train, y_train, batch_size=450, nb_epoch=150, validation_split=0.05)

    predicted = model.predict(X_test)
    return model, (predicted, y_test)


data = _load_data()
optimalHyperParams = {'l1_drop': 0.5, 'l1_neurons': 200, 'time_series_len': 100, 'l2_neurons': 250}
currentHyperParams = optimalHyperParams
currentHyperParams['out_neurons'] = 1
# currentHyperParams['out_neurons'] = 12
currentHyperParams['input_dim'] = 5
tests = []

for i in range(10):

    model, (predicted, y_test) = run_model(data, currentHyperParams)
    rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
    tests.append([rmse, predicted, y_test])
    print("Error for round %s: %s" % (i,rmse))
    # plt.figure()
    # plt.hist([x[0] for x in tests])
    # plt.savefig('tests/graphs/error_dist_%s.png' % i)
    plt.figure()
    plt.scatter(range(len(predicted)), predicted,c="r")
    plt.scatter(range(len(y_test)), y_test,c="g")
    plt.savefig('tests/graphs/pred_%s.png' % i)
    plt.figure()
    plt.scatter(range(len(predicted)), y_test - predicted, c="b")
    plt.savefig('tests/graphs/residuals_%s.png' % i)
    # model.save("tests/models/model_%s" % i)
    save_obj(tests, "tuned_testing")
