import sklearn shit there

"""
Reads in the baseline census training data, and the census test data.
These values are stored into 2D lists of the form:
[['v1', 'v2', 'v3', 'v4', ...], [...]],
Where the inner lists represents each of the different census candidates.
"""
file = open('../data/adult.data', 'r')
censusData = []
for line in file:
    data = line.split()
    data.pop(2)
    data.pop(2)
    censusData.append(data)
file.close()

file = open('../data/adult.test', 'r')
testData = []
for line in file:
    data = line.split()
    data.pop(2)
    data.pop(2)
    censusData.append(data)
file.close()

'''HW2 Below

import scipy.sparse as scp
import numpy as np
import imblearn.under_sampling as un
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from mlxtend.preprocessing import shuffle_arrays_unison
from sklearn.model_selection import cross_val_score


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


	########### Select Best Features ###########
	# *** Similar to setting up & fitting the vectorizer from HW1
	kbest = SelectKBest(score_func=chi2, k=k_features)
	kbestTrain = kbest.fit_transform(sparseTrain.toarray(), labels)
	kbestTest = kbest.transform(sparseTest.toarray())
	############################################


	############## Undersampling ###############
	undersampled = un.EditedNearestNeighbours()
	#undersampled = un.RepeatedEditedNearestNeighbours()
	usTrain, usLabels = undersampled.fit_sample(kbestTrain, labels)
	usTrain = scp.csr_matrix(usTrain)
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
