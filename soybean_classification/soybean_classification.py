from __future__ import print_function
import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import datahandler


def run():
    # Split-out validation dataset
    dataset = load_data_set()
    array = dataset.values
    x = array[:, 1:35]
    y = array[:, 0]
    validation_size = 0.10
    seed = 7
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=validation_size,
                                                                                    random_state=seed)

    best_k = choose_best_k_to_knn(x_train, y_train, x_validation, y_validation)
    #print("\nThe best k to KNN: %s" % best_k)

    # Algorithms
    knn = KNeighborsClassifier(n_neighbors=best_k)
    dtc = DecisionTreeClassifier()
    mlp = MLPClassifier(random_state=seed, solver='lbfgs')
    nb = GaussianNB()
    svc = SVC(kernel='poly', gamma="auto")

    # Make predictions on validation dataset
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    algorithm = knn
    algorithm_chosen = "KNN (n_neighbors= %i)" % best_k

    dtc.fit(x_train, y_train)
    predictions = dtc.predict(x_validation)
    accuracy_aux = accuracy_score(y_validation, predictions)
    if accuracy_aux > accuracy:
        accuracy = accuracy_aux
        algorithm = dtc
        algorithm_chosen = "Decision Tree"

    mlp.fit(x_train, y_train)
    predictions = mlp.predict(x_validation)
    accuracy_aux = accuracy_score(y_validation, predictions)
    if accuracy_aux > accuracy:
        accuracy = accuracy_aux
        algorithm = mlp
        algorithm_chosen = "MultiLayer Perceptron"

    nb.fit(x_train, y_train)
    predictions = nb.predict(x_validation)
    accuracy_aux = accuracy_score(y_validation, predictions)
    if accuracy_aux > accuracy:
        accuracy = accuracy_aux
        algorithm = nb
        algorithm_chosen = "Naive Bayes"

    svc.fit(x_train, y_train)
    predictions = svc.predict(x_validation)
    accuracy_aux = accuracy_score(y_validation, predictions)
    if accuracy_aux > accuracy:
        algorithm = svc
        algorithm_chosen = "SVM"

    print("\nAlgorithm Chosen: " + algorithm_chosen)
    print("Accuracy: %f" % accuracy)
    #print(accuracy_score(y_validation, predictions))
    #print(confusion_matrix(y_validation, predictions))
    #print(classification_report(y_validation, predictions))

def choose_best_k_to_knn(x_train, y_train, x_validation, y_validation):
    accuracy = 0
    k = 1
    for i in range(3, 30):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        predictions = knn.predict(x_validation)
        accuracy_aux = accuracy_score(y_validation, predictions)
        if accuracy_aux > accuracy:
            accuracy = accuracy_aux
            k = i
    return k