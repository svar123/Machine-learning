#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")
from functions import computeFraction,computeSum,countNan,getBestFeatures,getClassifierFit
from feature_format import featureFormat, targetFeatureSplit

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV

from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

features_list = ['poi','salary','bonus','exercised_stock_options','expenses','from_messages','from_poi_to_this_person','from_this_person_to_poi','other','restricted_stock','shared_receipt_with_poi','to_messages','total_payments','total_stock_value','total_bonus_salary','fraction_emails_to_poi','fraction_emails_from_poi']
  
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###Remove outliers
#scatter plot to look for outliers

features_outliers = ["salary", "bonus"]
data = featureFormat(data_dict, features_outliers)
#plot_outlier(data,features_outliers)

#The following three values are outliers which was determined from the
#scatter plot and visualizing the dataset

print "\n"
print "Removing outliers 'TOTAL','THE TRAVEL AGENCY IN PARK'......"
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN PARK',0)
print "Removing user with NaN values for all features......"
data_dict.pop('LOCKHART EUGENE E',0)

#find how many Nans are in each of the features
count = countNan(data_dict)

print "\n"
print "Count of NaNs in each feature"
print "-----------------------------"
for keys,values in count.items():
    print ("{:30s}:{:4d}".format(keys,values))


#Create 3 new features 
#1. 'fraction_emails_from_poi'
#2. 'fraction_emails_to_poi'
#3. 'total_bonus_salary'

print "\n"
print "Creating 3 new features 'fraction_from_poi','fraction_to_poi','total_bonus_salary'....."
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_dict[name]["fraction_emails_from_poi"]=fraction_from_poi
    data_dict[name]["fraction_emails_to_poi"] = fraction_to_poi
    sal = data_point["salary"]
    bon = data_point["bonus"]
    total = computeSum(sal,bon)
    data_dict[name]["total_bonus_salary"] = total



my_dataset = data_dict

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Select K best features, where K=10
features_selected = getBestFeatures(features,labels,features_list)
print '\n'
print "10 Best features and their scores:"
print "----------------------------------"
for keys,values in features_selected:
    print ("{:25s}: {:4f}".format(keys,values)) 
print '\n'

### create a new feature list with the 10 best features
my_feature_list =['poi']
for feature in features_selected:
    my_feature_list.append(feature[0])
print "New Feature list:"
print "----------------"
for f in my_feature_list:
    print f


###extract the new dataset with the new features
data = featureFormat(my_dataset, my_feature_list)
labels, features = targetFeatureSplit(data)

### scale the new dataset features
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
#features = scaler.fit_transform(features)

###split into training and test data
from sklearn import cross_validation

features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(features,labels, test_size=0.3, random_state=42)


# Try a variety of classifiers.

cv_strata = StratifiedShuffleSplit(n_splits=30, test_size=0.3, random_state=42)
classifiers = [GaussianNB(),DecisionTreeClassifier(),SVC(),KNeighborsClassifier()]
classifierNames = ['NaiveBayes','DecisionTree','SupportVector','KNeighbors']

print "\n"
print "Trying out different Classifiers.....:"

i=0
while i < len(classifiers):
    if i == 0:
       print "1. Naive Bayes Classifier:"
       print "   -----------------------------"
       params={}
#       params=dict(pca__n_components=[1,2,3,4,5],
#               pca__random_state=[42])
       nb = getClassifierFit(features,labels,scaler,cv_strata,params,classifiers[i],classifierNames[i])
       print "\n"
    elif i == 1:
       print "2. Decision Tree Classifier:"
       print "   -----------------------------"
       params=dict(DecisionTree__max_features = [1,4],
                  DecisionTree__min_samples_split = [2,10,20,40,60],
                  DecisionTree__max_depth = [ 10,20,50],
                  DecisionTree__criterion = ['gini'],
                  DecisionTree__random_state = [42])
#                  pca__n_components=[1,2,3,4,5],
#                  pca__random_state=[42])
       dt = getClassifierFit(features,labels,scaler,cv_strata,params,classifiers[i],classifierNames[i])
       print "\n"
    elif i == 2:
        print "3. Support Vector Classifier:"
        print "   -----------------------------"
        params=dict(SupportVector__gamma = [0,0.05,0.1,1,5],
                  SupportVector__C = [1,10,50,100,500,700,900],
                  SupportVector__kernel = ['rbf'],
                  SupportVector__random_state=[42])
#                  pca__n_components=[1,2,3,4,5],
#                  pca__random_state=[42])
        sv = getClassifierFit(features,labels,scaler,cv_strata,params,classifiers[i],classifierNames[i])
        print "\n"
    elif i == 3:
        print "4. K-Nearest Neighbors Classifier:"
        print    "-----------------------------"
        params=dict(KNeighbors__n_neighbors = [2,3,5],
                  KNeighbors__leaf_size = [1,2,3])
#                  pca__n_components=[1,2,3,4,5],
#                  pca__random_state=[42])
        kn = getClassifierFit(features,labels,scaler,cv_strata,params,classifiers[i],classifierNames[i])
        print "\n"

    i+=1
    

#used for tester
dump_classifier_and_data(dt,data_dict,my_feature_list)
