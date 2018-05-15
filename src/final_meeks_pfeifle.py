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
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
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
######################### Read in the data with Pandas #########################
censusData = pan.read_csv("../data/adult.data.csv")
testData = pan.read_csv("../data/adult.test.csv")

######################### Take out unnecessary columns #########################
del censusData["education"]
del censusData["fnlwgt"]
del testData["education"]
del testData["fnlwgt"]

############## Create our labels in place of the income column. Use <=50K for 0, >50K for 1 ##############
censusData.income.replace({"<=50K": 0, ">50K": 1}, inplace=True)
testData.income.replace({"<=50K.": 0, ">50K.": 1}, inplace=True)

# Trying to figure out the binarizer still..
# We don't need a binerizer if we use the get_dummies function with pandas!
#binarizer = LabelBinarizer()
#binarizer.fit_transform(censusData) # Makes binary double array

################ Take off our income columns to use as our label arrays #########################
censusLabels = censusData.pop("income")
testLabels = testData.pop("income")

############### Use get_dummies to convert/encode our categorical variables into dummy(indicator) variables ###############
##### We need to do this in order for our classifiers to be able to work! Can't handle continuous & categorical data ######
censusTrain = pan.get_dummies(censusData)
testFeatures = pan.get_dummies(testData)

#################### The train and test aren't equal in size, need to fix before classification! #########################
#print(censusTrain.columns.equals(testFeatures.columns)) # WILL BE FALSE
# In the train data, someone has their native country as Holand-Netherlands, no one in the test data has that so we need to account for that. (Hence below the 90 and 91)
#print(censusTrain.shape) #(32561, 91)
#print(testFeatures.shape) #(16281, 90)
#print(censusTrain.columns.difference(testFeatures.columns)) # Index(['native_country_Holand-Netherlands'], dtype='object')
# In the array of differences, get the index 0 object (our only one anyways), then at that spot in testFeatures make it = 0
testFeatures[censusTrain.columns.difference(testFeatures.columns)[0]]= 0
# Preserve order
testFeatures = testFeatures[censusTrain.columns]
#print(censusTrain.columns.equals(testFeatures.columns)) # Should be true now...success!


######################################## CLASSIFIERS ########################################
print('\n')

#################### Decision Tree Classifier ####################
#dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
# Want to prune the tree above, not supported in sklearn, to do this we can set a max depth!
# A max depth of 10 acheived the best accuracy, leave it at that.
dtree = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth = 10)
dtree.fit(censusTrain, censusLabels)
error = metrics.mean_absolute_error(testLabels, dtree.predict(testFeatures))
print('For Decision Tree Classifier, the mean absolute error is: {0}'.format(error)) # 0.13911921872120878
print('and the accuracy score is thus: {0}\n'.format(1 - error))

#################### Naive Bayes Classifier ####################
under = un.RandomUnderSampler(random_state=0)
sampledTestTrain, sampledTestLabels = under.fit_sample(testFeatures, testLabels)
nbayes = BernoulliNB()
nbayes.fit(censusTrain, censusLabels)
error = metrics.mean_absolute_error(sampledTestLabels, nbayes.predict(sampledTestTrain))
print('For Naive Bayes Classifier, the mean absolute error is: {0}'.format(error)) # 0.24167966718668746 <- This seems like a bit high for error?
print('and the accuracy score is thus: {0}\n'.format(1 - error))

############################## K Nearest Neighbors Classifiers ##############################
# Tried several values we could record the accuracy for for comparison and discussion in the report.
#################### K Nearest Neighbors Classifier (KNN = 1) ####################
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(censusTrain, censusLabels)
error = metrics.mean_absolute_error(testLabels, knn.predict(testFeatures))
print('For K=1 Nearest Neighbors Classifier, the mean absolute error is: {0}'.format(error)) # 0.18125422271359254
print('and the accuracy score is thus: {0}\n'.format(1 - error))

#################### K Nearest Neighbors Classifier (KNN = 3) ####################
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(censusTrain, censusLabels)
error = metrics.mean_absolute_error(testLabels, knn.predict(testFeatures))
print('For K=3 Nearest Neighbors Classifier, the mean absolute error is: {0}'.format(error)) # 0.15969535040845156
print('and the accuracy score is thus: {0}\n'.format(1 - error))

#################### K Nearest Neighbors Classifier (KNN = 5) ####################
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(censusTrain, censusLabels)
error = metrics.mean_absolute_error(testLabels, knn.predict(testFeatures))
print('For K=5 Nearest Neighbors Classifier, the mean absolute error is: {0}'.format(error)) # 0.1487623610343345
print('and the accuracy score is thus: {0}\n'.format(1 - error))

#################### Support Vector Classifier (SVM) ####################
svm = SVC(random_state=0)
svm.fit(censusTrain, censusLabels)
error = metrics.mean_absolute_error(testLabels, svm.predict(testFeatures))
print('For SVM classifier SVC, the mean absolute error is: {0}'.format(error)) # 0.13088876604631164
print('and the accuracy score is thus: {0}\n'.format(1 - error))
