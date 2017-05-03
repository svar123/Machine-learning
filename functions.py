#!/usr/bin/python

### file contains helper functions used in poi_id.py

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    fraction = 0.
    if poi_messages == 'NaN' or all_messages == 'NaN':
        fraction = 0.
    else:
        fraction = (float)(poi_messages)/all_messages

    return fraction


def computeSum(salary,bonus):
    """given salary and bonus compute the sum
       and return the total
    """

    if salary == 'NaN':
        salary = 0
    if bonus == 'NaN':
        bonus = 0

    return (salary + bonus)


def countNan(data):
    """ this function returns a count of Nan values
        in each of the features
    """
    count = {}

    for keys in data.keys():
        for val in data[keys].keys():
            count[val] = 0

    for keys in data.keys():
        for val in data[keys].keys():
            if data[keys][val] == 'NaN':
                count[val]+=1
    return count

def getBestFeatures(features,labels,features_list):
    """ given features list return sorted 10 best features 
    """
    from sklearn.feature_selection import SelectKBest,f_classif
    
    selector = SelectKBest(f_classif,10)
    selector.fit(features,labels)
    feature_scores = selector.scores_
    features_selected_tuple=[(features_list[i+1], feature_scores[i]) for i in selector.get_support(indices=True)]
    features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)
    return features_selected_tuple

def getClassifierFit(features,labels,scaler,cv_strata,params,classifier,classifierName):
    """ for the specified classifier and parameters return the
        best estimator model
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    
   
    pipeline = Pipeline(steps=[('scaling',scaler),(classifierName,classifier)])
    gs = GridSearchCV(pipeline,params,scoring='f1', cv=cv_strata)
    gs.fit(features, labels)

    print "Best estimator: \n", gs.best_estimator_
    print 'F1 score: %0.3f' % gs.best_score_
    return gs.best_estimator_
