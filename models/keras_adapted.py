# % matplotlib inline
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Masking,Highway,Flatten,Reshape,Merge
from keras.layers.convolutional import Convolution1D
from keras.layers.noise import GaussianNoise
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams, pyplot as plt

# adds repo root to the path
import os, sys, inspect
cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)


rcParams['figure.figsize'] = [9, 6]

# for determinism
seed = 7
# np.random.seed(seed)


# this method takes two arrays and shuffles them using the same random state. This preserves the pairs.
def shuffle_in_unison(a, b):
    # np.random.seed()
    # np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b

# scales the features
def scale_data(X, Y):
    X_scaler = StandardScaler()
    X = np.array(X_scaler.fit_transform(X.astype(float)))
    X, Y = shuffle_in_unison(X, Y)
    X = X.astype(float)
    return X, Y

def _encode_fit(col):
    encoder = LabelEncoder()
    encoder.fit(col)
    return encoder.transform(col)

# prepares the data for the ungrouped models
def _no_group_data():
    df = pd.read_csv('AA_4_computed_new.csv', header=None).fillna(0)
    ds = df.values
    titles = ds[0,:]
    ds = ds[1:, :]
    X = ds[:,[0, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]] #manual selection of features
    Y = np.reshape(ds[:, 2], (len(ds[:, 2]),))

    # encode categorical cols
    Y = _encode_fit(Y) # encodes winner
    X[:, 0] = _encode_fit(X[:, 0]) # encodes year
    X[:, 1] = _encode_fit(X[:, 1]) # encodes producer

    return scale_data(X,Y)

# prepares the data for the grouped models
def _group_data():
    # np.random.seed(seed)

    df = pd.read_csv('AA_4_computed_new.csv', header=None).fillna(0)
    ds = df.values
    titles = ds[0, :]
    ds = ds[1:, :]
    fields = [2, 0, 4, 5, 6, 7, 8, 9, 10, 17]
    ds = ds[:, fields]  # manual selection of features

    ds[:, 0] = _encode_fit(ds[:, 0])  # encodes Winner
    ds[:, 1] = _encode_fit(ds[:, 1])  # encodes Year
    ds[:, 2] = _encode_fit(ds[:, 2])  # encodes Producer

    field_subset = pd.DataFrame(ds)
    g = field_subset.groupby([1]).groups.items() # groups data by year (season)
    seasons = len(g)
    maxSeasonlen = 12
    feature_len = len(fields[1:])

    # initialize empty arrays
    np.zeros_like((seasons, maxSeasonlen))
    X, Y = [np.zeros((seasons, maxSeasonlen, feature_len)), np.zeros((seasons, maxSeasonlen))]

    for index, (season_year, group_indexes) in enumerate(g):
        season = [field_subset.iloc[i].values for i in group_indexes]
        x = [m[1:] for m in season]
        y = [m[0] for m in season]

        X[index][:len(x)] = x
        Y[index][:len(y)] = y
        # shuffles the movies in the season so that there is no bias against the later channels which are sparse more often
        X[index], Y[index] = shuffle_in_unison(X[index], Y[index])

    # scales the features
    X_scaler = StandardScaler()
    X = np.array([X_scaler.fit_transform(x) for x in X])
    # shuffles the seasons
    X, Y = shuffle_in_unison(X, Y)
    X = X.astype(float)
    return X, Y

# small by movie model
def create_baseline():
    model = Sequential()
    model.add(Dense(len(X[0, :]), input_dim=len(X[0, :]), init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# large by movie model
def create_larger():
    model = Sequential()
    model.add(Dense(len(X[0, :])*.75, input_dim=len(X[0, :]), init='normal', activation='relu'))
    model.add(Dropout(.05))
    model.add(Dense(len(X[0, :]) / 2, init='normal', activation='relu'))
    model.add(Dense(len(X[0, :]) / 4, init='normal', activation='tanh'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
    return model

# first seasonal model. params were initally scaled by number of features
# was converted into hardcoded values for presentation
def create_grouped():
    # np.random.seed(seed)
    feature_len = len(X[0, 0,:])
    season_len = len(X[0,:])
    model = Sequential()
    model.add(GaussianNoise(.05, input_shape=(season_len, feature_len)))  # output:(None, 12, 9)
    model.add(Convolution1D(40, 1))  # (None, 12, 40)
    model.add(Flatten())  # (None, 480)
    model.add(Dense(int(40), init='normal', activation='relu'))  # (None, 40)
    model.add(Dropout(.25))
    model.add(Dense(int(40), init='normal', activation='relu', bias=True))
    model.add(Dropout(.25))
    model.add(Dense(int(20), init='normal', activation='tanh'))  # (None, 20)
    model.add(Dropout(.2))
    model.add(Dense(12, init='normal', activation='softmax'))  # (None, 12)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
    return model

# runs two models using a sigmoid or tanh layer
# merges models into a final model with a sigmoid and final softmax layer
def create_mergegrouped(base_params):
    # np.random.seed(seed)

    feature_len = len(X[0, 0, :])
    season_len = len(X[0, :])
    node_stretch = base_params['layer_scale']
    noise_std = base_params['noise']
    conv_size = base_params['conv_size']

    # sized based on number of features times scale factor
    l1_size = int(feature_len * node_stretch)
    l2_size = int(feature_len * node_stretch)

    left_model = Sequential()
    right_model = Sequential()

    left_model.add(GaussianNoise(noise_std, input_shape=(season_len, feature_len)))
    right_model.add(GaussianNoise(noise_std, input_shape=(season_len, feature_len)))
    left_model.add(Convolution1D(conv_size, 1, input_shape=(season_len, feature_len), input_length=feature_len))
    right_model.add(Convolution1D(conv_size, 1, input_shape=(season_len, feature_len), input_length=feature_len))
    left_model.add(Flatten())
    right_model.add(Flatten())

    left_model.add(Dense(l1_size, init='normal', activation='sigmoid'))
    left_model.add(Dropout(.15))

    right_model.add(Dense(l1_size, init='normal', activation='tanh'))
    right_model.add(Dropout(.15))

    final_model = Sequential()
    merged = Merge([left_model,right_model], mode='ave')
    final_model.add(merged)
    final_model.add(Dense(l2_size, init='normal', activation='sigmoid'))
    final_model.add(Dropout(.1))
    final_model.add(Dense(12, init='normal', activation='softmax', bias=True))
    final_model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
    return final_model

# helper for plotting the data from each validation and test
def make_plots(test, preds, acc, loss, trial):
    ## this portion was useful in initally seeing how the model was predicting when outputting ratio predictions.
    # fig = plt.figure()
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
    plt.title('model accuracy trial: %s' % (trial))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig('tests/graphs/acc_%s.png' % trial)
    plt.close(fig)
    # summarize history for loss
    fig = plt.figure()
    plt.plot(loss)
    plt.title('model loss trial: %s' % (trial))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('tests/graphs/loss_%s.png' % trial)
    plt.close(fig)

# helper for tracking scores and storing trained models
def _score(scores, model, test=False):
    if not test:
        accs.append(scores[1] * 100)
        precs.append(scores[2] * 100)
        recs.append(scores[3] * 100)
        fscores.append(scores[4] * 100)
        trained_models.append(model)
        print("_-_-_-_-_-_-_-_-_-_-_-_-_-_")
        print(scores, model.metrics_names)
    else:
        print("_-_-_-_-_Test Scores_-_-_-_-_")
        print("Acc.  %s" % (scores[1] * 100))
        print("prec.  %s" % (scores[2] * 100))
        print("rec.  %s" % (scores[3] * 100))
        print("F.  %s" % (scores[4] * 100))

# training method for non-seasonal models
def train_nogroup(train, test, trial):
    hist = model.fit(train[0], train[1], nb_epoch=750, batch_size=50, verbose=0)
    preds = model.predict_classes(train[0], batch_size=50, verbose=0)
    scores = model.evaluate(test[0], test[1], verbose=0)
    _score(scores,model)
    make_plots(test, preds, hist.history['acc'], hist.history['loss'], trial)

# training method for seasonal model
def train_group(train, valid, trial):
    # np.random.seed(seed)
    hist = model.fit(train[0], train[1], nb_epoch=500, batch_size=10, verbose=0)
    preds = model.predict_classes(valid[0], batch_size=10, verbose=0)
    # preds = model.predict(X_test, batch_size=50, verbose=0)
    scores = model.evaluate(valid[0], valid[1], verbose=0)
    _score(scores,model)
    make_plots(valid[1], preds, hist.history['acc'], hist.history['loss'], trial)

# training method for merged-seasonal model (requires two copies of the inputs)
def train_mergegroup(train, valid, trial):
    # np.random.seed(seed)
    hist = model.fit([train[0],train[0]], train[1], nb_epoch=600, batch_size=10, verbose=0)
    preds = model.predict_classes([valid[0],valid[0]], batch_size=10, verbose=0)
    scores = model.evaluate([valid[0],valid[0]], valid[1], verbose=0)
    _score(scores,model)
    make_plots(valid[1], preds, hist.history['acc'], hist.history['loss'], trial)

# associated test methods
def test_nogroup(test):
    preds = model.predict_classes(test[0], batch_size=50, verbose=0)
    scores = model.evaluate(test[0], test[1], verbose=0)
    _score(scores,model, test=True)
    # make_plots(test, preds, hist.history['acc'], hist.history['loss'], trial)

def test_group(test):
    preds = model.predict_classes(test[0], batch_size=10, verbose=0)
    # preds = model.predict(X_test, batch_size=50, verbose=0)
    scores = model.evaluate(test[0], test[1], verbose=0)
    _score(scores,model, test=True)
    # make_plots(test[1], preds, hist.history['acc'], hist.history['loss'], 0)

def test_mergegroup(test):
    preds = model.predict_classes([test[0],test[0]], batch_size=10, verbose=0)
    scores = model.evaluate([test[0],test[0]], test[1], verbose=0)
    _score(scores, model, test=True)
    # make_plots(vaild[1], preds, hist.history['acc'], hist.history['loss'], 0)



## start of script execution

## for determinism
np.random.seed(seed)

# prompts for a type of model
model_types = ['movie', 'season', 'merged_season']
print("Choose a model type. Options %s" % model_types)
model_type = str(raw_input("model: "))

print("preparing data for a %s model" % model_type)
X, Y = [], []
if model_type == 'movie':
    X,Y = _no_group_data()
    # can stratify the non grouped data
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
else :
    X,Y = _group_data()
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# splits into training and test samples
split_point = int(len(X)*.8)
x_train, x_test = [X[:split_point],X[split_point:]]
y_train, y_test = [Y[:split_point],Y[split_point:]]
folds = list(kfold.split(x_train, y_train))

# init Kfold scoring params
trial = 1
accs = []
precs = []
recs = []
fscores = []

# these param are for auto testing different values of a hyper parameter during KFold Cross Validation
p_scale = 15
p_const = 1
p_name = 'conv_size'
param_values = [p_const+x*p_scale for x in range(len(folds))]
trained_models = []
base_params = {
    'noise': .03,
    'conv_size': 120,
    'layer_scale': 8
}
print('training models for %s folds' % len(folds))
for (train, valid), param_val in zip(folds,param_values):

    ## override base param with the param for this round of KFCV
    # base_params[p_name] = param_val
    # print('training %s with value %s' % (p_name,param_val))

    ## model types
    if model_type == 'movie':
        model = create_larger()
        train_nogroup((x_train[train],y_train[train]),(x_train[valid],y_train[valid]),trial)
    elif model_type == 'season':
        model = create_grouped()
        train_group((x_train[train],y_train[train]),(x_train[valid],y_train[valid]),trial)
    elif model_type == 'merged_season':
        model = create_mergegrouped(base_params)
        train_mergegroup((x_train[train],y_train[train]),(x_train[valid],y_train[valid]),trial)
    else:
        raise ValueError("Unknown model type: %s" % model_type)

    ## for inspecting layer in-out shapes
    # for l in model.layers:
    #     print(l.output_shape)

    trial += 1
# gets list of trained models to pick best from
selection_list = [[f,m]for f,m in zip(fscores,trained_models)]
selection_list.sort(key=lambda x: x[0], reverse=True)
model = selection_list[0][1] # takes model with best f score
print('best model had fscore %s' % selection_list[0][0])
print('testing best model')

## test for model type
if model_type == 'movie':
    test_nogroup((x_test, y_test))
elif model_type == 'season':
    test_group((x_test, y_test))
elif model_type == 'merged_season':
    test_mergegroup((x_test, y_test))
else:
    raise ValueError("Unknown model type: %s" % model_type)

# print report of KFolds
for i, scs in enumerate(zip(accs, precs, recs, fscores)):
    print('Stat report for fold %s: Acc. %.2f%% | Prec. %.2f%% | Rec. %.2f%% | Fscore %.2f%%' % (i + 1, scs[0], scs[1], scs[2], scs[3]))
print('Average accuracy over all folds: %.2f%% (+/- %.2f%%)' % (np.mean(accs), np.std(accs)))
print('Average precision over all folds: %.2f%% (+/- %.2f%%)' % (np.mean(precs), np.std(precs)))
print('Average recall over all folds: %.2f%% (+/- %.2f%%)' % (np.mean(recs), np.std(recs)))
print('Average Fscore over all folds: %.2f%% (+/- %.2f%%)' % (np.mean(fscores), np.std(fscores)))