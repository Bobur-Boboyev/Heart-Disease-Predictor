from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from preprocessing import preprocessing_data

x_train, x_test, y_train, y_test, scaler = preprocessing_data('data/heart.csv')

def train_KNN():
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(x_train, y_train)
    acc = round(accuracy_score(y_test, model.predict(x_test)), 2)*100
    return model, acc

def train_SVM():
    model = SVC(C=1, gamma='scale')
    model.fit(x_train, y_train)
    acc = round(accuracy_score(y_test, model.predict(x_test)), 2)*100
    return model, acc

def train_NN():
    model = MLPClassifier(hidden_layer_sizes=(50,), alpha=0.001, early_stopping=True, max_iter=1000)
    model.fit(x_train, y_train)
    acc = round(accuracy_score(y_test, model.predict(x_test)), 2)*100
    return model, acc
