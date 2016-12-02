# % matplotlib inline
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Masking,Highway,Flatten,Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU, SReLU
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import rcParams, pyplot as plt

import os, sys, inspect
cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)


rcParams['figure.figsize'] = [9, 6]

seed = 7


def shuffle_in_unison(a, b):
    # np.random.seed()
    np.random.seed(seed)

    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b

def _no_group_data():
    df = pd.read_csv('AA_4_computed_new.csv', header=None).fillna(0)
    ds = df.values
    titles = ds[0,:]
    ds = ds[1:, :]
    X = ds[:,[0, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]] #manual selection of numeric fields
    Y = np.reshape(ds[:, 2], (len(ds[:, 2]), 1))

    encoder_r = LabelEncoder()
    encoder_p = LabelEncoder()
    encoder_y = LabelEncoder()
    encoder_r.fit(Y)
    encoder_p.fit(X[:, 1])
    encoder_y.fit(X[:, 0])
    Y = encoder_r.transform(Y)
    X[:, 1] = encoder_p.transform(X[:, 1])
    X[:, 0] = encoder_y.transform(X[:, 0])
    return X, Y

def _group_data():
    np.random.seed(seed)

    df = pd.read_csv('AA_4_computed_new.csv', header=None).fillna(0)
    ds = df.values
    titles = ds[0, :]
    ds = ds[1:, :]
    fields = [2, 0, 4, 5, 6, 7, 8, 9, 10, 17]
    ds = ds[:, fields]  # manual selection of numeric fields
    # Y = np.reshape(ds[:, 2], (len(ds[:, 2]), 1))
    # print(ds[-3:])
    encoder_r = LabelEncoder()
    encoder_p = LabelEncoder()
    encoder_y = LabelEncoder()
    encoder_r.fit(ds[:,0])
    encoder_p.fit(ds[:, 2])
    encoder_y.fit(ds[:, 1])
    ds[:, 0] = encoder_r.transform(ds[:,0])
    ds[:, 2] = encoder_p.transform(ds[:, 2])
    ds[:, 1] = encoder_y.transform(ds[:, 1])

    field_subset = pd.DataFrame(ds)
    g = field_subset.groupby([1]).groups.items()
    # print list(g)
    seasons = len(g)
    maxSeasonlen = 12
    feature_len = len(fields[1:])
    # print feature_len
    np.zeros_like((seasons, maxSeasonlen))
    out = (np.zeros((seasons, maxSeasonlen, feature_len)),np.zeros((seasons, maxSeasonlen)))
    # print out[0][1]
    for index, (season_year, group_indexes) in enumerate(g):
        season = [field_subset.iloc[i].values for i in group_indexes]

        # print season
        x = [m[1:] for m in season]
        y = [m[0] for m in season]
        # print x
        out[0][index][:len(x)] = x
        out[1][index][:len(y)] = y
        # shuffles the season so that there is no bias against the later channels which are sparse more often
        out[0][index], out[1][index] = shuffle_in_unison(out[0][index],out[1][index])
    return out[0], out[1]


# print(X, np.unique(X[:, 0], return_counts=True))
# print(Y)
# print(X[0,:])
# print(X[0,:])

def create_baseline():
    model = Sequential()
    model.add(Dense(len(X[0, :]), input_dim=len(X[0, :]), init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_smaller():
    model = Sequential()
    model.add(Dense(len(X[0, :]) / 2, input_dim = len(X[0, :]), init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_larger():
    model = Sequential()
    model.add(Dense(len(X[0, :])*.75, input_dim=len(X[0, :]), init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(.05))
    # model.add(SReLU())
    model.add(Dense(len(X[0, :]) / 2, init='normal', activation='relu'))
    model.add(Dense(len(X[0, :]) / 4, init='normal', activation='tanh'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
    return model

def create_grouped():
    np.random.seed(seed)

    feature_len = len(X[0, 0,:])
    season_len = len(X[0,:])
    model = Sequential()
    node_stretch = 8
    model.add(GaussianNoise(.20,input_shape=(season_len,feature_len)))
    # model.add(Flatten(input_shape=(season_len,feature_len)))
    model.add(Convolution1D(1, 1,input_shape=(season_len,feature_len), input_length=feature_len,bias=True))

    model.add(Flatten())

    model.add(Dense(int(feature_len *.5 * node_stretch), init='normal',))
    model.add(Activation('relu'))
    model.add(Dropout(.25))
    # model.add(SReLU())
    model.add(Dense(int(feature_len *.5 * node_stretch), init='normal', activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(int(feature_len *.20 * node_stretch), init='normal', activation='tanh'))
    model.add(Dropout(.2))
    # model.add(Dense(1, init='normal', activation='sigmoid'))
    model.add(Dense(12, init='normal', activation='sigmoid'))
    model.add(Dense(12, init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
    return model


def make_plots(test, preds, acc, loss, trial):
    # lets see our predictions!
    fig = plt.figure()
    # print(test)
    # print(preds)
    # return
    # plt.scatter(range(len(test)), test, c="g")
    # plt.scatter(range(len(preds)), preds, c="r")
    # plt.xlim(-2, len(preds) + 2)
    # plt.ylim(-0.05, 1.05)
    # plt.title('Predictions and actual values for trial %s' % (trial))
    # plt.legend(['actual', 'prediction'], loc='center')
    # # plt.show()
    # plt.savefig('tests/graphs/pred_%s.png' % trial)
    # plt.close(fig)
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(acc)
    #     plt.plot(hist.history['val_acc']) #only generate these metrics when using val keyword in fit()
    plt.title('model accuracy trial: %s' % (trial))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    #     plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('tests/graphs/acc_%s.png' % trial)
    plt.close(fig)
    # summarize history for loss
    fig = plt.figure()
    plt.plot(loss)
    #     plt.plot(hist.history['val_loss'])
    plt.title('model loss trial: %s' % (trial))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #     plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('tests/graphs/loss_%s.png' % trial)
    plt.close(fig)


def test_fold(train, test, trial, lasttest=None):
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X[train])
    X_test = X_scaler.transform(X[test])
    hist = model.fit(X_train, Y[train], nb_epoch=750, batch_size=50, verbose=0)
    # if lasttest is not None:
        # X_train = X_scaler.fit_transform(X[lasttest])
        # hist = model.fit(X_train, Y[lasttest], nb_epoch=750, batch_size=50, verbose=0)
    preds = model.predict_classes(X_test, batch_size=50, verbose=0)
    scores = model.evaluate(X_test, Y[test], verbose=0)
    accs.append(scores[1] * 100)
    precs.append(scores[2] * 100)
    recs.append(scores[3] * 100)
    fscores.append(scores[4] * 100)
    print("_-_-_-_-_-_-_-_-_-_-_-_-_-_")
    print(scores, model.metrics_names)
    #     print(hist.history.keys())
    #     print(preds)
    make_plots(test, preds, hist.history['acc'], hist.history['loss'], trial)
    return test

def train_group(train, vaild, trial):
    np.random.seed(seed)

    X_train = np.array([x for x in train[0]])
    X_valid = np.array([x for x in vaild[0]])
    hist = model.fit(X_train, train[1], nb_epoch=500, batch_size=10, verbose=0)
    preds = model.predict_classes(X_valid, batch_size=10, verbose=0)
    # preds = model.predict(X_test, batch_size=50, verbose=0)
    scores = model.evaluate(X_valid, vaild[1], verbose=0)
    accs.append(scores[1] * 100)
    precs.append(scores[2] * 100)
    recs.append(scores[3] * 100)
    fscores.append(scores[4] * 100)
    print("_-_-_-_-_-_-_-_-_-_-_-_-_-_")
    print(scores, model.metrics_names)
    #     print(hist.history.keys())
    #     print(preds)
    make_plots(vaild[1], preds, hist.history['acc'], hist.history['loss'], trial)

def test_group(test):
    np.random.seed(seed)

    preds = model.predict_classes(test[0], batch_size=10, verbose=0)
    # preds = model.predict(X_test, batch_size=50, verbose=0)
    scores = model.evaluate(test[0], test[1], verbose=0)
    accs.append(scores[1] * 100)
    precs.append(scores[2] * 100)
    recs.append(scores[3] * 100)
    fscores.append(scores[4] * 100)
    print("_-_-_-_-TEST-_-_-_-_-_")
    print(scores, model.metrics_names)
    print(preds)
    print(test[1])
    #     print(hist.history.keys())
    #     print(preds)
    # make_plots(vaild[1], preds, hist.history['acc'], hist.history['loss'], 0)
# print X[0]
# print Y[0]
# assert 1 ==0
np.random.seed(seed)
X,Y = _group_data()
X_scaler = StandardScaler()
X = np.array([X_scaler.fit_transform(x) for x in X])
X,Y = shuffle_in_unison(X,Y)
X = X.astype(float)
split_point = int(len(X)*.8)
x_train, x_test = [X[:split_point],X[split_point:]]
y_train, y_test = [Y[:split_point],Y[split_point:]]
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
kfold = KFold(n_splits=10, shuffle=True)
trial = 1
accs = []
precs = []
recs = []
fscores = []
model = create_grouped()
print model.layers[0].output_shape
print model.layers[1].output_shape
print model.layers[2].output_shape
folds = list(kfold.split(x_train, y_train))
# print Y[folds[0][1]]
# print Y[folds[1][1]]
# assert 1 ==0
# test_group((X[folds[0][0]],Y[folds[0][0]]),(X[folds[0][1]],Y[folds[0][1]]),trial)
# test_group((X[:-2],Y[:-2]),(X[-2:],Y[-2:]),trial)
# trial += 1
# test_fold(folds[1][0],folds[1][1],trial)
# trial += 1
lasttest = None
for train, vaild in folds:
    lasttest = train_group((x_train[train],y_train[train]),(x_train[vaild],y_train[vaild]),trial)
    # lasttest = test_fold(train, test,trial, lasttest=lasttest)
    trial += 1
test_group((x_test,y_test))

for i, scs in enumerate(zip(accs, precs, recs, fscores)):
    print('Stat report for round %s: Acc. %.2f%% | Prec. %.2f%% | Rec. %.2f%% | Fscore %.2f%%' % (i + 1, scs[0], scs[1], scs[2], scs[3]))
print('Average accuracy over all rounds: %.2f%% (+/- %.2f%%)' % (np.mean(accs), np.std(accs)))
print('Average precision over all rounds: %.2f%% (+/- %.2f%%)' % (np.mean(precs), np.std(precs)))
print('Average recall over all rounds: %.2f%% (+/- %.2f%%)' % (np.mean(recs), np.std(recs)))
print('Average Fscore over all rounds: %.2f%% (+/- %.2f%%)' % (np.mean(fscores), np.std(fscores)))