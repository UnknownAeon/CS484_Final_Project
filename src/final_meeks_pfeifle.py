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
from sklearn.preprocessing import CategoricalEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from mlxtend.preprocessing import shuffle_arrays_unison
from sklearn.model_selection import cross_val_score

############## Data Preprocessing ###############
"""
Test
Reads in the baseline census training data, and the census test data.
These values are stored into 2D lists of the form:
[['v1', 'v2', 'v3', 'v4', ...], [...]],
Where the inner lists represents each of the different census candidates.

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
print(censusData.loc[[5, 10, 15]])
print(testData.loc[[5, 10, 15]])

# Trying to figure out the binarizer still..
# binarizer = LabelBinarizer()
# binarizer.fit_transform(dta.native_country.head(15)) # Makes binary double array

"""
file = open('../data/adult.data', 'r')
censusData = []
labels = []
for line in file:
    data = line.replace(',', '').split()
    data.pop(2)
    data.pop(2)
    if (data[len(data) - 1] == '<=50K'):
        labels.append(0)
    elif (data[len(data) - 1] == '>50K'):
        labels.append(1)
    else:
        raise ValueError('Improper Data Format')
    for i in range(len(data)):
        try:
            data[i] = int(data[i])
        except:
            # We don't want to do anything if it can't cast - silently catch and move on.
            pass
    censusData.append(data)
print(censusData[0:2])
print(censusData[-1])
file.close()
labels = np.array(labels)

file = open('../data/adult.test', 'r')
testData = []
testLabels = []
for line in file:
    data = line.replace(',', '').replace('.', '').split()
    data.pop(2)
    data.pop(2)
    if (data[len(data) - 1] == '<=50K'):
        testLabels.append(0)
    elif (data[len(data) - 1] == '>50K'):
        testLabels.append(1)
    else:
        raise ValueError('Improper Data Format')
    for i in range(len(data)):
        try:
            data[i] = int(data[i])
        except:
            # We don't want to do anything if it can't cast - silently catch and move on.
            pass
    testData.append(data)
file.close()
testLabels = np.array(testLabels)
############################################


################ Encoding ##################
encoder = CategoricalEncoder('ordinal')
encodedData = encoder.fit_transform(censusData, labels)
encodedTest = encoder.fit_transform(testData)
print(encodedData)
print(encodedTest)
# print(encoder.inverse_transform(encodedData))
############################################
"""
'''
############## Undersampling ###############
undersampled = un.EditedNearestNeighbours()
#undersampled = un.RepeatedEditedNearestNeighbours()
usTrain, usLabels = undersampled.fit_sample(censusData, labels)
usTrain = scp.csr_matrix(usTrain)
############################################

###HW2 Below###
'''
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
