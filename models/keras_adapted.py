# % matplotlib inline
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Masking,Highway,Flatten,Reshape
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU, SReLU
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
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

df = pd.read_csv('AA_4_computed.csv', header=None).fillna(0)
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
# print(X, np.unique(X[:, 0], return_counts=True))
# print(Y)

# print(X[0,:])
X = X.astype(float)
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
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
    return model

#
# # TESTS
# np.random.seed(seed)
# estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, nb_epoch=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(build_fn=create_larger, nb_epoch=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# so we think the large network is the best, although the difference is minimal. all imporvements may really be due
# to random chance and the earlier standardaziation of the inputs.

def make_plots(test, preds, acc, loss, trial):
    # lets see our predictions!
    fig = plt.figure()
    plt.scatter(range(len(Y[test])), Y[test], c="g")
    plt.scatter(range(len(preds)), preds, c="r")
    plt.xlim(-2, len(preds) + 2)
    plt.ylim(-0.05, 1.05)
    plt.title('Predictions and actual values for trial %s' % (trial))
    plt.legend(['actual', 'prediction'], loc='center')
    # plt.show()
    plt.savefig('tests/graphs/pred_%s.png' % trial)
    plt.close(fig)
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



np.random.seed(seed)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
trial = 1
accs = []
precs = []
recs = []
fscores = []
for train, test in kfold.split(X, Y):
    model = create_larger()
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X[train])
    X_test = X_scaler.transform(X[test])
    hist = model.fit(X_train, Y[train], nb_epoch=750, batch_size=50, verbose=0)
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

    trial += 1