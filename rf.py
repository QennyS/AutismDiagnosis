import numpy as np
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def accuracy(Y_test,Y_predict):
    count = 0
    for i in range(len(Y_test)):
        if Y_test[i] == Y_predict[i]:
            count += 1
        else:
            pass
    accuracy = count/len(Y_test)
    return accuracy

def ver_1(X,Y):
    x_train, y_train, x_test, y_test = [], [], [], []
    skf = StratifiedKFold(n_splits=5, random_state = 10)
    for train_index, test_index in skf.split(X, Y):
        x_train.append(X[train_index])
        x_test.append(X[test_index])
        y_train.append(Y[train_index])
        y_test.append(Y[test_index])

    result = []
    for estimator in range(10,270,30):
        for depth in range (3,14,1):
            for features in range (3,30,5):
                rf = ensemble.RandomForestClassifier(n_estimators = estimator, max_features = features, max_depth= depth)
                temp = []
                temp2 = [estimator, depth, features]
                for i in range(5):
                    rf.fit(x_train[i],y_train[i])
                    rf_predict = rf.predict(x_test[i])
                    temp.append(accuracy(y_test[i],rf_predict))
                for i in temp:
                    temp2.append(i)
                result.append(temp2)

    result = np.array(result)
    np.savetxt('700_result_5.csv',result,fmt='%s',delimiter=',')
    return

def ver_2(X,Y):
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
    result = []
    for estimator in range(10,270,20):
        for feature in range (1,50,5):
            for depth in range (1,11,1):
                rf = ensemble.RandomForestClassifier(n_estimators = estimator, max_features = feature, max_depth= depth)
                temp = cross_val_score(rf, X, Y, scoring='roc_auc', n_jobs = -1, cv = 5).tolist()
                temp2 = [estimator, feature, depth]
                for i in temp:
                    temp2.append(i)
                result.append(temp2)

    result = np.array(result)
    np.savetxt('700_result_new_4.csv',result,fmt='%s',delimiter=',')
    return

def grid_Search():
    para_test2 = {'min_samples_leaf':range(1,50,5)}
    gsearch = GridSearchCV(estimator = ensemble.RandomForestClassifier(n_estimators = 200, max_depth = 11, min_samples_split = 60, random_state=10), param_grid = para_test2,
                                      scoring='roc_auc', n_jobs = -1, cv = 10)
    gsearch.fit(X_train,Y_train)
    print (gsearch.best_params_)
    print (gsearch.best_score_)
    return

def main():
    data = np.loadtxt('700.csv', delimiter=',')
    m, n = data.shape
    Y = data[:,n-1]
    X = np.delete(data, n-1, axis=1)
    ver_1(X,Y)
    #ver_2(X,Y)
    return

if __name__ == '__main__':
    main()
