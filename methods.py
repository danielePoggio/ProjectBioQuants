from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Levenshtein import distance
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from calcolatePerformance import calcolateModel
from sequenceAlignment import sequenceAlignment
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def statistics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    return accuracy, precision, recall


def randomForest(dataframe, classes, flagplt):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataframe['SequenceTransform'], dataframe['Function'],
                                                        test_size=0.3)
    # per funzionare gli alberi hanno bisogno di array non series
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    # Fit a model to the training data
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    # Make predictions on the testing data
    y_pred = model.predict(X_test)
    # staticisRFC = calcolateModel(y_test, y_pred, classes, flagplt, strName='Random Forest')
    statisticisRFC = statistics(y_test, y_pred)
    return statisticisRFC


def knn(dataframe, k, classes, flagplt):
    X_train, X_test, y_train, y_test = train_test_split(dataframe['Sequence'], dataframe['Function'],
                                                        test_size=0.3)
    # per funzionare gli alberi hanno bisogno di array non series
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    X_train = X_train.reshape(-1, 1)
    # y_train = y_train.reshape(-1,1)
    X_test = X_test.reshape(-1, 1)
    distanceTrainTrain = np.zeros((X_train.size, X_train.size))
    for i in range(X_train.size):
        for j in range(i, X_train.size):
            xi = X_train[i]
            xj = X_train[j]
            xi = xi[0]
            xj = xj[0]
            distanceTrainTrain[i, j] = distance(xi, xj)
            distanceTrainTrain[j, i] = distanceTrainTrain[i, j]

    distanceTestTrain = np.zeros((X_test.size, X_train.size))
    for i in range(X_test.size):
        for j in range(X_train.size):
            xi = X_test[i]
            xj = X_train[j]
            xi = xi[0]
            xj = xj[0]
            distanceTestTrain[i, j] = distance(xi, xj)

    neigh = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
    neigh.fit(distanceTrainTrain, y_train)
    y_predKNN = neigh.predict(distanceTestTrain)

    staticisKNN = calcolateModel(y_test, y_predKNN, classes, flagplt, strName='KNN with k = ' + str(k))

    cm = confusion_matrix(y_test, y_predKNN)
    global_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    return global_accuracy, staticisKNN


def knnSA(dataframe, k, classes):
    X_train, X_test, y_train, y_test = train_test_split(dataframe['Sequence'], dataframe['Function'], test_size=0.3)
    # per funzionare gli alberi hanno bisogno di array non series
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    X_train = X_train.reshape(-1, 1)
    # y_train = y_train.reshape(-1,1)
    X_test = X_test.reshape(-1, 1)
    distanceTrainTrain = np.zeros((X_train.size, X_train.size))
    for i in range(X_train.size):
        for j in range(i, X_train.size):
            xi = X_train[i]
            xj = X_train[j]
            xi = xi[0]
            xj = xj[0]
            distanceTrainTrain[i, j] = sequenceAlignment(xi, xj)
            distanceTrainTrain[j, i] = distanceTrainTrain[i, j]

    distanceTestTrain = np.zeros((X_test.size, X_train.size))
    for i in range(X_test.size):
        for j in range(X_train.size):
            xi = X_test[i]
            xj = X_train[j]
            xi = xi[0]
            xj = xj[0]
            distanceTestTrain[i, j] = sequenceAlignment(xi, xj)

    neigh = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
    neigh.fit(distanceTrainTrain, y_train)
    y_predKNN = neigh.predict(distanceTestTrain)

    staticisKNN = calcolateModel(y_test, y_predKNN, classes)
    return staticisKNN
