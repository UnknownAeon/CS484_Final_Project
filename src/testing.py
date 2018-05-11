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
#print(notes) # The info on the datasets themselves


##### Reading Data with Pandas #####
dta = pd.read_csv("../data/adult.data.cleaned.csv.gz", compression="gzip")
test = pd.read_csv("../data/adult.test.cleaned.csv.gz", compression="gzip")

##### General Info about Data #####
#print(dta.head()) #Print first 5 rows
#print(dta.info()) # Info about the dataframe object itself
#print(dta.describe()) # Statistical info about attributes in the dataframe
#print(dta.columns) # All the possible Columns (attributes)
#print(dta.columns.difference(test.columns)) #To make sure both datasets have the same attributes as they should
#print(dta.loc[[5, 10, 15]]) # Print out the rows at the specified indexes
#print(dta[["workclass", "education"]]) # Print out the specified columns
#print(dta.loc[[5, 10, 15], ["workclass", "education"]]) # Combining both above to get specified columns and rows


##### GroupBy Operations #####
#print(dta.groupby("income").education.describe()) # Statistical info on categories

### Prints out a map going down education <--> education number ###
'''
grouper = dta.groupby("education") # Grouper = pandas.core.groupby.DataFrameGroupBy object
education_map = grouper.education_num.unique()
education_map.sort_values(inplace=True)
with pd.option_context("max_rows", 20):
    print(education_map)
'''

'''Dunno what this was
grouper.education_num.apply(lambda x : x.unique()[0])
education_map.sort_values(inplace=True)

with pd.option_context("max_rows", 20):
    print(education_map)
'''


##### Plotting #####
##### This block will give us a bar graph of the education spread (# of people per education)
ax = dta.groupby("education").size().plot(kind="bar", figsize=(8, 8))
ax.set_yticklabels([])  # turn off y tick labels
# resize x label
xlabel = ax.xaxis.get_label()
xlabel.set_fontsize(24)
# resize x tick labels
labels = ax.xaxis.get_ticklabels()
[label.set_fontsize(20) for label in labels];


##### Seaborn #####
import seaborn as sns
g = sns.factorplot("education_num", "hours_per_week", hue="sex", col="income", data=dta) # Plot relating education number and hours per week worked


##### Deleting Columns we aren't using #####
del dta["education"]
del dta["fnlwgt"]
del test["education"]
del test["fnlwgt"]

'''This is weird probably won't be useful or used
##### Indexing with Booleans #####
dta.education_num <= 8
dta.ix[dta.education_num <= 8, "education_num"]
### .iloc vs .loc
dta.ix[dta.education_num <= 8, "education_num"].iloc[0]
dta.ix[dta.education_num <= 8, "education_num"].loc[3]
'''


##### Slicing with labels #####
#print(dta.groupby("workclass").age.mean())
'''Prints out:
    workclass
    ?                   40.960240
    Federal-gov         42.590625
    Local-gov           41.751075
    Never-worked        20.571429
    Private             36.797585
    Self-emp-inc        46.017025
    Self-emp-not-inc    44.969697
    State-gov           39.436055
    Without-pay         47.785714
    Name: age, dtype: float64
'''
#print(dta.groupby("workclass").age.mean().loc["Federal-gov":"Private"])
'''Prints out:
    workclass
    Federal-gov     42.590625
    Local-gov       41.751075
    Never-worked    20.571429
    Private         36.797585
    Name: age, dtype: float64
'''


##### Working with Categorical Data #####
cat = pd.Categorical(dta.workclass)
#print(cat.describe()) # Gives counts and frequencies
#print(cat)
''' Prints out:
    [State-gov, Self-emp-not-inc, Private, Private, Private, ..., Private, Private, Private, Private, Self-emp-inc]
    Length: 32561
    Categories (9, object): [?, Federal-gov, Local-gov, Never-worked, ..., Self-emp-inc,
                            Self-emp-not-inc, State-gov, Without-pay]
'''
#print(cat.categories)
''' Prints out:
    Index(['?', 'Federal-gov', 'Local-gov', 'Never-worked', 'Private',
           'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'],
          dtype='object')
'''
#print(cat.codes)
''' Prints out:
    [7 6 4 ... 4 4 5]
'''


##### Vectorized string operations #####
#print(dta.workclass.str.contains("\?")) # Prints out whether the rows contain ? in the workclass
##### Putting it together: Strings and Boolean Indexing #####
#print(dta.ix[dta.workclass.str.contains("\?"), "workclass"]) # Prints out the rows that contain ? in the workclass


##### Putting it together: Column Assignment #####
#print(dta.workclass.unique()) # All the Unique columns/attributes
### This block of code I believe changes all ?'s to 'Other' throughout the data for every column! ###
for col in dta:  # iterate through column names
    # only look at object types
    if not dta[col].dtype.kind == "O":
        continue
    # Replace "?" with "Other"
    if dta[col].str.contains("\?").any():
        dta.loc[dta[col].str.contains("\?"), col] = "Other"
        test.loc[test[col].str.contains("\?"), col] = "Other"
#print(dta.workclass.unique()) # All the Unique columns/attributes after the replacement


##### Replacing values using dictionaries #####
#print(dta.income)
dta.income.replace({"<=50K": 0, ">50K": 1})
# In-place changes
dta.income.replace({"<=50K": 0, ">50K": 1}, inplace=True)
test.income.replace({"<=50K.": 0, ">50K.": 1}, inplace=True)
#print(dta.income.mean()) # 0.2408095574460244
#print(test.income.mean()) # 0.23622627602727106



########## Classification with Scikit-Learn ##########
# ## Scikit-Learn API
#
# * Base object is the estimator
# * Any object that learns from data
#   * Classification, regression, clustering, or transformer
#
# * parameters passed to estimator
#     estimator = Estimator(*args, **kwargs)
#
# * fit method provided
#     estimator.fit(X, y)
#
# * Computed parameters have an underscore appended
#     estimator.coef_


########## Preparing the Data ##########
# * scikit-learn works with numerical data
# THESE Y VALUES ARE USED LATER FOR THE CLASSIFIERS
y = dta.pop("income")
y_test = test.pop("income")

########## Preprocessing ##########
# * Preprocessing for Text, Categorical variables, Standardization etc.
from sklearn.preprocessing import LabelBinarizer
#print(dta.native_country.head(15).values) # Looking at the first 15 rows, print their respective native_country values in an array


### Binary Stuff ###
binarizer = LabelBinarizer()
binarizer.fit_transform(dta.native_country.head(15)) # Makes binary double array
#print(binarizer.classes_) # Prints array for the options above
# Example: If the classes were: ['Cuba', 'India', 'Jamaica', 'Other', 'United-States'], and row 1 was Other, then it'd be [[0, 0, 0, 1, 0],...]


##### Pre-processing with pandas #####
X_train = pd.get_dummies(dta)
X_test = pd.get_dummies(test)


##### Deal with real life #####
#print(X_train.columns.equals(X_test.columns)) # WILL BE FALSE
#print(X_train.shape) #(32561, 91)
#print(X_test.shape) #(16281, 90)
#print(X_train.columns.difference(X_test.columns)) # Index(['native_country_Holand-Netherlands'], dtype='object')
X_test[X_train.columns.difference(X_test.columns)[0]]= 0 # ?
# Preserve order
X_test = X_test[X_train.columns]


# ## Reported Benchmarks
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

##### Classification and Regression Trees (CART) #####
#
# * Partition feature space into a set of rectangles via splits that lead to largest information gain
# * Fit simple model in each region (e.g., a constant)
# * Captures non-linearities and feature interactions
# * Note: not strictly necessary to dummy encode variables
from sklearn.tree import DecisionTreeClassifier, export_graphviz

dtree = DecisionTreeClassifier(random_state=0, max_depth=2)
dtree.fit(X_train, y)
export_graphviz(dtree, feature_names=X_train.columns, out_file="tree.dot")

### Run this if you have graphviz installed ###
#dot -Tpng tree.dot -o tree.png

from IPython.display import Image
#I think the below syntax will still need to be changed to work with python instead of the IPython Notebook
Image("tree.png", unconfined=True) # Displays the tree picture built above

# Fit the full tree and look at the error
dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(X_train, y)
# Performs slightly worse than C4.5 with no pruning
from sklearn import metrics
#print(metrics.mean_absolute_error(y_test, dtree.predict(X_test))) # 0.1768318899330508

# Beware overfitting!
dtree = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=10)
dtree.fit(X_train, y)
#print(metrics.mean_absolute_error(y_test, dtree.predict(X_test))) # 0.13899637614397151



##### Aside: Saving Models #####
# * All of the scikit-learn models are picklable
# * Using joblib directly is often preferable to using pickle
import joblib

##### Ensemble Methods ##### NOT SURE IF WE WANT TO INTO ENSEMBLES FOR THIS
'''
#### Boosting
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
'''
# ### Gradient Boosting
#
# * Generalizes boosting to any differentiable loss function

from sklearn.ensemble import GradientBoostingClassifier
'''
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
'''

##### Validation Methods #####

##### Cross-Validation #####
# * Sampling techniques to ensure low generalization error and avoid overfitting
#****************Needs to be updated
from sklearn.cross_validation import StratifiedKFold
cv = StratifiedKFold([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0, 0,], n_folds=3)
#for idx in cv:
#    print("train", idx[0], "test", idx[1])
''' Prints out:
train [ 4  5  6  7  8  9 10 13 14 15 16] test [ 0  1  2  3 11 12]
train [ 0  1  2  3  8  9 10 11 12 15 16] test [ 4  5  6  7 13 14]
train [ 0  1  2  3  4  5  6  7 11 12 13 14] test [ 8  9 10 15 16]
'''

### Don't really know what this is yet
from sklearn.grid_search import GridSearchCV
#****************Needs to be updated
cv = StratifiedKFold(y, n_folds=4)

#params = {"max_depth": [3, 5, 7]}
#gbt = GradientBoostingClassifier(n_estimators=500, learning_rate=.01)

#if not os.path.exists("models/grid_search.pkl"):
#    estimator = GridSearchCV(gbt, param_grid=params, verbose=2)
#    estimator.fit(X_train, y)
#    joblib.dump(estimator, "models/grid_search.pkl")
#else:
#    estimator = joblib.load("models/grid_search.pkl")
''' Prints out:
Fitting 3 folds for each of 3 candidates, totalling 9 fits
[CV] max_depth=3 .....................................................
[CV] ............................................ max_depth=3 -  34.3s
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:   34.3s remaining:    0.0s
[CV] max_depth=3 .....................................................
[CV] ............................................ max_depth=3 -  33.9s
[CV] max_depth=3 .....................................................
[CV] ............................................ max_depth=3 -  34.4s
[CV] max_depth=5 .....................................................
[CV] ............................................ max_depth=5 - 1.4min
[CV] max_depth=5 .....................................................
[CV] ............................................ max_depth=5 - 1.4min
[CV] max_depth=5 .....................................................
[CV] ............................................ max_depth=5 - 1.4min
[CV] max_depth=7 .....................................................
[CV] ............................................ max_depth=7 - 2.9min
[CV] max_depth=7 .....................................................
[CV] ............................................ max_depth=7 - 2.9min
[CV] max_depth=7 .....................................................
[CV] ............................................ max_depth=7 - 3.1min
[Parallel(n_jobs=1)]: Done   9 out of   9 | elapsed: 14.9min finished
'''


##### Out-of-bag estimates and Early-stopping #####
gbt = GradientBoostingClassifier(learning_rate=.01, n_estimators=1000, subsample=.5)
gbt.fit(X_train, y)
#print(metrics.mean_absolute_error(y_test, gbt.predict(X_test))) # 0.13045881702598117 Mine got: 0.12984460413979484?

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(gbt.oob_improvement_) # Prints big interpolating graph, declines and then plataeus, Y range 0 to .0075, X range 0 to 1000


### Not sure what this is or is for... ###
from sklearn.ensemble import GradientBoostingClassifier
# Ad-hoc way to do early-stopping
def monitor(i, self, local_variables):
    start = max(0, i - 4)
    stop = i + 1

    if i > 5 and np.mean(self.oob_improvement_[start:stop]) < 1e-4:
        #print("Stopped at {}".format(i)) # Stopped at 446
        return True
gbt.fit(X_train, y, monitor=monitor)
#print(len(gbt.oob_improvement_)) # 447


##### Custom Transformers #####
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

#for col in obj_cols:
    #print(col)
''' Prints:
workclass
marital_status
occupation
relationship
race
sex
native_country
'''

# This might be useful if we somehow run into issues about DataFrames vs arrays, otherwise don't know why we'd need it
##### Make a transformer that reliably transforms DataFrames and Arrays #####
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


##### Pipelines #####
# * Often it makes sense to do the data transformation, feature extraction, etc. as part of a Pipeline
# * Pipelines are flexible and provide the same sklearn API
from sklearn.pipeline import Pipeline
dtree_estimator = Pipeline([('transformer', PandasTransformer(dta)),
                            ('dtree', dtree)])
dtree_estimator.fit(dta, y)
#print(dtree_estimator.named_steps['dtree'])
''' Prints:
DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best')
'''
dtree_estimator.predict_proba(test) # Not sure what this prints yet
