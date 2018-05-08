'''
Looking at the github repo's rendered python notebook will help with understanding this code.
Some of the code itself was altered to make actually work, and one big difference that will have to
be changed is as you can see at the start, things like dta.head() on its own wont work, must be print(dta.head())
We might, and probably won't need all of this, I'll be playing with it some to try and trim down the stuff we actually need,
so that we can implement the logic within our own code.
'''
import os
import numpy as np
import pandas as pd

#get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("max_rows", 10)
np.set_printoptions(suppress=True)


# Pretty Graphs
from seaborn import set_style
set_style("darkgrid")
import seaborn as sns
import matplotlib.pyplot as plt

# About the Data
with open("../data/adult.names") as fin:
    notes = fin.read()

#print(notes)

# # Pandas for Data Wrangling

# ## Reading Data

dta = pd.read_csv("../data/adult.data.cleaned.csv.gz", compression="gzip")

test = pd.read_csv("../data/adult.test.cleaned.csv.gz", compression="gzip")

# ## Explore the Data

print(dta.head())

print(dta.info())

print(dta.describe())

# ## Pandas Orientation

# ### Indices

# #### Index

dta.index


# #### Columns

dta.columns

dta.columns.difference(test.columns)

# Sanity checks

dta.columns.equals(test.columns)

dta.columns.difference(test.columns)

# #### Indexing

dta.loc[[5, 10, 15]]

# #### Selecting Columns

dta[["workclass", "education"]]

type(dta[["workclass"]])

type(dta["workclass"])
'''
# #### Rows and Columns

dta.loc[[5, 10, 15], ["workclass", "education"]]

# ## GroupBy Operations

dta.groupby("income").education.describe()

grouper = dta.groupby("education")

grouper

education_map = grouper.education_num.unique()
education_map.sort_values(inplace=True)

with pd.option_context("max_rows", 20):
    print(education_map)

grouper.education_num.apply(lambda x : x.unique()[0])
education_map.sort_values(inplace=True)

with pd.option_context("max_rows", 20):
    print(education_map)

# ## Plotting

ax = dta.groupby("education").size().plot(kind="bar", figsize=(8, 8))

ax.set_yticklabels([])  # turn off y tick labels

# resize x label
xlabel = ax.xaxis.get_label()
xlabel.set_fontsize(24)

# resize x tick labels
labels = ax.xaxis.get_ticklabels()
[label.set_fontsize(20) for label in labels];

# ### Seaborn

import seaborn as sns

g = sns.factorplot("education_num", "hours_per_week", hue="sex", col="income", data=dta)

# ### Deleting Columns

del dta["education"]
del dta["fnlwgt"]
del test["education"]
del test["fnlwgt"]

# ### Advanced Indexing

# #### Indexing with Booleans

dta.education_num <= 8

dta.ix[dta.education_num <= 8, "education_num"]

# #### .iloc vs .loc

dta.ix[dta.education_num <= 8, "education_num"].iloc[0]

dta.ix[dta.education_num <= 8, "education_num"].loc[3]

# #### Slicing with labels (!)

dta.groupby("workclass").age.mean()

dta.groupby("workclass").age.mean().loc["Federal-gov":"Private"]

# #### Filtering Columns with Regex

dta.filter(regex="capital")

# ### Working with Categorical Data

# #### Categorical Object

cat = pd.Categorical(dta.workclass)
cat.describe()

cat

cat.categories

cat.codes

# #### Vectorized string operations

dta.workclass.str.contains("\?")

# #### Putting it together: Strings and Boolean Indexing

dta.ix[dta.workclass.str.contains("\?"), "workclass"]

# #### Putting it together: Column Assignment

dta.workclass.unique()

for col in dta:  # iterate through column names
    # only look at object types
    if not dta[col].dtype.kind == "O":
        continue

    # Replace "?" with "Other"
    if dta[col].str.contains("\?").any():
        dta.loc[dta[col].str.contains("\?"), col] = "Other"
        test.loc[test[col].str.contains("\?"), col] = "Other"

dta.workclass.unique()

# #### Replacing values using dictionaries

dta.income

dta.income.replace({"<=50K": 0, ">50K": 1})

# In-place changes

dta.income.replace({"<=50K": 0, ">50K": 1}, inplace=True)

test.income.replace({"<=50K.": 0, ">50K.": 1}, inplace=True)

dta.income.mean()

test.income.mean()

# # Classification with Scikit-Learn

# ## Scikit-Learn API
#
# * Base object is the estimator
# * Any object that learns from data
#   * Classification, regression, clustering, or transformer
#
# * parameters passed to estimator
#
# ```python
#     estimator = Estimator(*args, **kwargs)
# ```
#
# * `fit` method provided
#
# ```python
#     estimator.fit(X, y)
# ```
#
# * Computed parameters have an underscore appended
#
# ```python
#     estimator.coef_
# ```

# ## Preparing the Data

# * scikit-learn works with numerical data

y = dta.pop("income")
y_test = test.pop("income")

dta.info()

# ## Preprocessing

# * Preprocessing for Text, Categorical variables, Standardization etc.

from sklearn.preprocessing import LabelBinarizer

dta.native_country.head(15).values

binarizer = LabelBinarizer()

# `fit_transform` is short hand for calling `fit` then `transform`

binarizer.fit_transform(dta.native_country.head(15))

binarizer.classes_

# Pre-processing with pandas

X_train = pd.get_dummies(dta)

X_test = pd.get_dummies(test)

# Deal with real life

X_train.columns.equals(X_test.columns)

print(X_train.shape)
print(X_test.shape)

X_train.columns.difference(X_test.columns)

X_test[X_train.columns.difference(X_test.columns)[0]] = 0

# Preserve order

X_test = X_test[X_train.columns]

# ## Reported Benchmarks
#
# ```
# |    Algorithm               Error
# | -- ----------------        -----
# | 1  C4.5                    15.54
# | 2  C4.5-auto               14.46
# | 3  C4.5 rules              14.94
# | 4  Voted ID3 (0.6)         15.64
# | 5  Voted ID3 (0.8)         16.47
# | 6  T2                      16.84
# | 7  1R                      19.54
# | 8  NBTree                  14.10
# | 9  CN2                     16.00
# | 10 HOODG                   14.82
# | 11 FSS Naive Bayes         14.05
# | 12 IDTM (Decision table)   14.46
# | 13 Naive-Bayes             16.12
# | 14 Nearest-neighbor (1)    21.42
# | 15 Nearest-neighbor (3)    20.35
# | 16 OC1                     15.04
# ```

# ## Classification and Regression Trees (CART)
#
# * Partition feature space into a set of rectangles via splits that lead to largest information gain
# * Fit simple model in each region (e.g., a constant)
# * Captures non-linearities and feature interactions
# * Note: not strictly necessary to dummy encode variables

from sklearn.tree import DecisionTreeClassifier, export_graphviz

dtree = DecisionTreeClassifier(random_state=0, max_depth=2)

dtree.fit(X_train, y)

export_graphviz(dtree, feature_names=X_train.columns, out_file="tree.dot")

from IPython.display import Image
Image("tree.png", unconfined=True)

# Fit the full tree and look at the error

dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(X_train, y)

# Performs slightly worse than C4.5 with no pruning

from sklearn import metrics

metrics.mean_absolute_error(y_test, dtree.predict(X_test))

# Beware overfitting!

dtree = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=10)
dtree.fit(X_train, y)
metrics.mean_absolute_error(y_test, dtree.predict(X_test))

# ## Aside: Saving Models

# * All of the scikit-learn models are picklable
# * Using joblib directly is often preferable to using pickle

import joblib

# ## Ensemble Methods

# ### Boosting
#
# * Combine many weak classifiers in to one strong one
#   * Weak classifier is slightly better than random
# * Sequentially apply a classifier to repeatedly modified versions of data
# * Each subsequent classifier improves on the mistakes of its predecessor
# * For Boosting Trees, the classifier is a decision tree

# Create a random dataset

import numpy as np
rng = np.random.RandomState(1)
groundX = np.sort(rng.uniform(0, 10, size=250), axis=0)
groundy = np.linspace(1, -1, 250) + np.sin(2*groundX).ravel()
idx = np.random.randint(0, 250, size=30)
idx.sort()
XX = groundX[idx]
yy = groundy[idx]
XX = XX[:, np.newaxis]

from sklearn.tree import DecisionTreeRegressor
tree1 = DecisionTreeRegressor(max_depth=2)

tree1.fit(XX, yy)
y1 = tree1.predict(XX)

resid1 = yy - y1
tree1.fit(XX, resid1)

y2 = tree1.predict(XX)
resid2 = y2 - resid1
tree1.fit(XX, resid2)

y3 = tree1.predict(XX)

fig, ax = plt.subplots(4, 1, figsize=(6, 18), sharey=True, sharex=True)
ax[0].plot(XX, yy, marker='o', ls='', label='observed')
ax[0].plot(groundX, groundy, label='truth')
ax[0].legend(fontsize=16, loc='lower left');
ax[1].plot(XX, yy, marker='o', ls='')
ax[1].plot(XX, y1)
ax[2].plot(XX, resid1, marker='o', ls='')
ax[2].plot(XX, y2)
ax[3].plot(XX, resid2, marker='o', ls='')
ax[3].plot(XX, y3)
fig.suptitle("Residual Fitting", fontsize=24);
fig.tight_layout()

# ### Gradient Boosting
#
# * Generalizes boosting to any differentiable loss function

from sklearn.ensemble import GradientBoostingClassifier

if not os.path.exists("models/gbt1.pkl"):
    gbt = GradientBoostingClassifier(max_depth=5, n_estimators=1000)
    gbt.fit(X_train, y)
    joblib.dump(gbt, "models/gbt1.pkl")
else:
    gbt = joblib.load("models/gbt1.pkl")

metrics.mean_absolute_error(y_test, gbt.predict(X_test))


# In[ ]: THIS TOOK A LONG (DIDN'T LET IT FINISH LONG) TIME

#
#if not os.path.exists("models/gbt2.pkl"):
#    gbt = GradientBoostingClassifier(max_depth=8, n_estimators=1000, subsample=.5, random_state=0,
#                                    learning_rate=.001)
#    gbt.fit(X_train, y)
#    joblib.dump(gbt, "models/gbt2.pkl")
#else:
#    gbt = joblib.load("models/gbt2.pkl")
#
# In[ ]:
#
#
#metrics.mean_absolute_error(y_test, gbt.predict(X_test))


# ### Bagging
#
# * Bootstrap aggregating (bagging)
#   * Simple bootstrapping is sampling with replacement
# * Fit the same learner to many bootstrap samples and average the results
# * Random forests builds on the idea of bagging and uses trees
# * Performance similar to boosting but can be easier to train and tune

# ### Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

if not os.path.exists("models/rf.pkl"):
    rf = RandomForestClassifier(n_estimators=1000, criterion='entropy', n_jobs=4, max_depth=10)
    rf.fit(X_train, y)
    joblib.dump(rf, "models/rf.pkl")
else:
    rf = joblib.load("models/rf.pkl")

metrics.mean_absolute_error(y_test, rf.predict(X_test))

# * Rule of thumb is that you can't really overfit with random forests
#   * This is true in general but only to an extent
# * Usually ok to grow large forests with full depth trees (problem dependent)
#   * Limiting the depth of the trees and the number of trees can be

if not os.path.exists("models/rf_full.pkl"):
    rf_full = RandomForestClassifier(n_estimators=1000, criterion='entropy',
                                     n_jobs=4, max_depth=None)
    rf_full.fit(X_train, y)
    joblib.dump(rf, "models/rf_full.pkl")
else:
    rf_full = joblib.load("models/rf_full.pkl")

metrics.mean_absolute_error(y_test, rf_full.predict(X_test))

# ## Validation Methods

# ### Cross-Validation
#
# * Sampling techniques to ensure low generalization error and avoid overfitting

from sklearn.cross_validation import StratifiedKFold

cv = StratifiedKFold([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0, 0,], n_folds=3)
for idx in cv:
    print("train", idx[0], "test", idx[1])

from sklearn.grid_search import GridSearchCV

cv = StratifiedKFold(y, n_folds=4)

params = {"max_depth": [3, 5, 7]}
gbt = GradientBoostingClassifier(n_estimators=500, learning_rate=.01)

if not os.path.exists("models/grid_search.pkl"):
    estimator = GridSearchCV(gbt, param_grid=params, verbose=2)
    estimator.fit(X_train, y)
    joblib.dump(estimator, "models/grid_search.pkl")
else:
    estimator = joblib.load("models/grid_search.pkl")

# ### Out-of-bag estimates and Early-stopping

gbt = GradientBoostingClassifier(learning_rate=.01, n_estimators=1000, subsample=.5)

gbt.fit(X_train, y)

metrics.mean_absolute_error(y_test, gbt.predict(X_test))

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(gbt.oob_improvement_)

from sklearn.ensemble import GradientBoostingClassifier

# Ad-hoc way to do early-stopping

def monitor(i, self, local_variables):
    start = max(0, i - 4)
    stop = i + 1

    if i > 5 and np.mean(self.oob_improvement_[start:stop]) < 1e-4:
        print("Stopped at {}".format(i))
        return True

gbt.fit(X_train, y, monitor=monitor)

print(len(gbt.oob_improvement_))

# ## Custom Transformers

def get_obj_cols(dta, index=False):
    """
    dta : pd.DataFrame
    index : bool
        Whether to return column names or the numeric index.
        Default False, returns column names.
    """
    columns = dta.columns.tolist()
    obj_col_names = list(filter(lambda x : dta[x].dtype.kind == "O",
                                columns))
    if not index:
        return obj_col_names
    else:
        return list(columns.index(col) for col in obj_col_names)

obj_cols = get_obj_cols(dta)

for col in obj_cols:
    print(col)

# Make a transformer that reliably transforms DataFrames and Arrays

from sklearn.base import TransformerMixin, BaseEstimator


class PandasTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, dataframe):
        self.columns = dataframe.columns
        self.obj_columns = get_obj_cols(dataframe, index=True)
        obj_index = np.zeros(dataframe.shape[1], dtype=bool)
        obj_index[self.obj_columns] = True
        self.obj_index = obj_index


    def fit(self, X, y=None):
        X = np.asarray(X)
        # create the binarizer transforms
        _transformers = {}
        for col in self.obj_columns:
            _transformers.update({col: LabelBinarizer().fit(X[:, col])})

        self._transformers = _transformers
        return self

    def transform(self, X, y=None):
        X = np.asarray(X)

        dummies = None
        for col in self.obj_columns:
            if dummies is None:
                dummies = self._transformers[col].transform(X[:, col])
            else:
                new_dummy = self._transformers[col].transform(X[:, col])
                dummies = np.column_stack((dummies, new_dummy))

        # remove original columns
        X = X[:, ~self.obj_index]

        X = np.column_stack((X, dummies))

        return X

# ## Pipelines

# * Often it makes sense to do the data transformation, feature extraction, etc. as part of a Pipeline
# * Pipelines are flexible and provide the same sklearn API

from sklearn.pipeline import Pipeline

dtree_estimator = Pipeline([('transformer', PandasTransformer(dta)),
                            ('dtree', dtree)])

dtree_estimator.fit(dta, y)

dtree_estimator.named_steps['dtree']

dtree_estimator.predict_proba(test)
'''
