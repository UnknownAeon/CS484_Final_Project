'''
Names: Kevin Pfeifle and Nolan Meeks
CS 484-001: Data Mining
Final Project - Census Data Classification
'''
import scipy.sparse as scp
import numpy as np
import imblearn.under_sampling as un
import pandas as pan
from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.preprocessing import CategoricalEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from mlxtend.preprocessing import shuffle_arrays_unison
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics


############## Data Preprocessing ###############
"""
[Age, Workclass, Education (Number), Marital Status, Occupation, Relationship, Race, Sex, Capital Gain, Capital Loss, House per Week, Native Country, *Wage Label*]
Age:
    Continuous
Workclass:
    1. Private
    2. Self-emp-not-inc
    3. Self-emp-inc
    4. Federal-gov
    5. Local-gov
    6. State-gov
    7. Without-pay
    8. Never-worked
Education:
    Continuous
Marital Status:
    1. Married-civ-spouse
    2. Divorced
    3. Never-married
    4. Separated
    5. Widowed
    6. Married-spouse-absent
    7. Married-AF-spouse
Occupation:
    1. Tech-support
    2. Craft-repair
    3. Other-service
    4. Sales, Exec-managerial
    5. Prof-specialty
    6. Handlers-cleaners
    7. Machine-op-inspct
    8. Adm-clerical
    9. Farming-fishing
    10. Transport-moving
    11. Priv-house-serv
    12. Protective-serv
    13. Armed-Forces
Relationship:
    1. Wife
    2. Own-child
    3. Husband
    4. Not-in-family
    5. Other-relative
    6. Unmarried
Race:
    1. White
    2. Asian-Pac-Islander
    3. Amer-Indian-Eskimo
    4. Other
    5. Black
Sex:
    1. Male
    2. Female
Capital Gain:
    Continuous
Capital Loss:
    Continuous
House per Week:
    Continuous
Native Country:
    1. United-States
    2. Cambodia
    3. England
    4. Puerto-Rico
    5. Canada
    6. Germany
    7. Outlying-US(Guam-USVI-etc)
    8. India
    9. Japan
    10. Greece
    11. South
    12. China
    13. Cuba
    14. Iran
    15. Honduras
    16. Philippines
    17. Italy
    18. Poland
    19. Jamaica
    20. Vietnam
    21. Mexico
    22. Portugal
    23. Ireland
    24. France
    25. Dominican-Republic
    26. Laos
    27. Ecuador
    28. Taiwan
    29. Haiti
    30. Columbia
    31. Hungary
    32. Guatemala
    33. Nicaragua
    34. Scotland
    35. Thailand
    36. Yugoslavia
    37. El-Salvador
    38. Trinadad&Tobago
    39. Peru
    40. Hong
    41. Holand-Netherlands

"""

censusData = pan.read_csv("../data/adult.data.cleaned.csv")
testData = pan.read_csv("../data/adult.test.cleaned.csv")
del censusData["education"]
del censusData["fnlwgt"]
del testData["education"]
del testData["fnlwgt"]

censusData.income.replace({"<=50K": 0, ">50K": 1}, inplace=True)
testData.income.replace({"<=50K.": 0, ">50K.": 1}, inplace=True)
#print(censusData.loc[[5, 10, 15]])
#print(testData.loc[[5, 10, 15]])

# Trying to figure out the binarizer still..
#binarizer = LabelBinarizer()
#binarizer.fit_transform(censusData.native_country.head(15)) # Makes binary double array

##### Prepare the data #####
censusLabels = censusData.pop("income")
testLabels = testData.pop("income")

# print(censusData.head())

##### Pre-processing with pandas #####
censusTrain = pan.get_dummies(censusData)
testTrain = pan.get_dummies(testData)

print(censusTrain.head())


##### Deal with real life #####
print(censusTrain.columns.equals(testTrain.columns)) # WILL BE FALSE
# In the train data, someone has their native country as Holand-Netherlands, no one in the test data has that so we need to account for that. (Hence below the 90 and 91)
print(censusTrain.shape) #(32561, 91)
print(testTrain.shape) #(16281, 90)
print(censusTrain.columns.difference(testTrain.columns)) # Index(['native_country_Holand-Netherlands'], dtype='object')
# In the array of differences, get the index 0 object (our only one anyways), then at that spot in testTrain make it = 0
testTrain[censusTrain.columns.difference(testTrain.columns)[0]]= 0
# Preserve order
testTrain = testTrain[censusTrain.columns]

print(censusTrain.columns.equals(testTrain.columns)) # Should be true now...success!

# TODO: c4.5, undersample bayes

### Decision Tree Classifier ###
dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(censusTrain, censusLabels)
print('For Decision Tree Classifier, the mean absolute error is:')
print(metrics.mean_absolute_error(testLabels, dtree.predict(testTrain))) # 0.1761562557582458


### Naive Bayes Classifier ###
nbayes = BernoulliNB()
nbayes.fit(censusTrain, censusLabels)
print('For Naive Bayes Classifier, the mean absolute error is:')
print(metrics.mean_absolute_error(testLabels, nbayes.predict(testTrain))) #0.2588293102389288
# ^^^ This seems too high of an error...

## KNN Classifiers, wrote it out a bunch of times instead of just putting the best - might be good  to include
# the process of trying different values in the report for comparisions.
### K Nearest Neighbors Classifier (KNN = 1)###
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(censusTrain, censusLabels)
print('For K=1 Nearest Neighbors Classifier, the mean absolute error is:')
print(metrics.mean_absolute_error(testLabels, knn.predict(testTrain))) #0.18125422271359254

### K Nearest Neighbors Classifier (KNN = 3)###
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(censusTrain, censusLabels)
print('For K=3 Nearest Neighbors Classifier, the mean absolute error is:')
print(metrics.mean_absolute_error(testLabels, knn.predict(testTrain))) #0.15969535040845156

### K Nearest Neighbors Classifier (KNN = 5)###
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(censusTrain, censusLabels)
print('For K=5 Nearest Neighbors Classifier, the mean absolute error is:')
print(metrics.mean_absolute_error(testLabels, knn.predict(testTrain))) #0.1487623610343345

### Support Vector Classifier (SVM) - Warning: This takes quite a while ###
svm = SVC()
svm.fit(censusTrain, censusLabels)
print('For SVM classifier SVC, the mean absolute error is:')
print(metrics.mean_absolute_error(testLabels, svm.predict(testTrain))) #0.13088876604631164

'''
################ Encoding ##################
encoder = CategoricalEncoder('ordinal')
encodedData = encoder.fit_transform(censusData, labels)
encodedTest = encoder.fit_transform(testData)
print(encodedData)
print(encodedTest)
# print(encoder.inverse_transform(encodedData))
############################################
'''
'''
############## Undersampling ###############
undersampled = un.EditedNearestNeighbours()
#undersampled = un.RepeatedEditedNearestNeighbours()
usTrain, usLabels = undersampled.fit_sample(censusData, labels)
usTrain = scp.csr_matrix(usTrain)
############################################
'''
###HW2 Below###

'''
# Create a Sparse Matrix from a list
def CreateSparseMatrix(drugs):
    ones = []
    drugIndexes = []
    drugFeatures = []

    for index, drug in enumerate(drugs):
        features = map(int, drug.split())
        for featureVal in features:
            ones.append(1)
            drugIndexes.append(index)
            drugFeatures.append(featureVal - 1)
    return scp.coo_matrix((ones, (drugIndexes, drugFeatures)), shape = [len(drugs), 100000]).tocsr()


def main():

	### Modifiable Parameters: ###
	k_features = 316
	kneighbors = 30
	chosenClassifier = 'bnbClassifier'
	#chosenClassifier = 'knnClassifier'
	#chosenClassifier = 'dtcClassifier'


	############# Prepare the data #############
	labelsData = []
	trainDrugs = []
	testDrugs = []
	# Open files
	with open('traindrugs.txt', 'r') as drugs:
		data = drugs.readlines()
		for line in data:
			line=line.split('\t')
			labelsData.append(int(line[0]))
			trainDrugs.append(line[1])
	with open('testdrugs.txt', 'r') as newDoc:
		newData = newDoc.readlines()
		for line in newData:
			testDrugs.append(line)
	# Create Numpy Array of the labels for use as array
	labels = np.array(labelsData)

	# Create Sparse Matrices of the data
	sparseTrain = CreateSparseMatrix(trainDrugs)
	sparseTest = CreateSparseMatrix(testDrugs)
	############################################


	############### Classifiers ################
	# KNN
	#clf = KNeighborsClassifier(n_neighbors=kneighbors)
	# Bernoulli-NB
	clf = BernoulliNB()
	# DTC
	#clf = DecisionTreeClassifier()

	# Properly fit the classified features with the labels for each drug, and predict
	# 	whether the test drugs are active or not
	clf.fit(usTrain, usLabels)
	results = clf.predict(kbestTest)
	############################################


	############# Cross-Validation #############
	randomTrain, randomLabels = shuffle_arrays_unison(arrays=[kbestTrain, labels])
	CVScores = cross_val_score(clf, randomTrain, randomLabels, scoring = 'f1', cv = 10)
	############################################


	################## Write ##################
	print(chosenClassifier + "_" + str(k_features) + "Features")
	print(str(CVScores))
	print(str(np.mean(CVScores)))
	np.savetxt(chosenClassifier + "_" + str(k_features) + "Features" + ".txt", results, fmt='%s')
	############################################

main()
'''
