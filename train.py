from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocessing import preprocessing_data

train_x, test_x, train_y, test_y = preprocessing_data('data/heart.csv')

def train_KNN():
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    
    acc = round(accuracy_score(test_y, pred), 2)*100
    cm = confusion_matrix(test_y, pred)

    print("KNN Confusion Matrix:\n", cm)

    return f"KNN Accuracy Score: %{acc}\n"

def train_SVM():
    model = SVC(max_iter=1000)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    
    acc = round(accuracy_score(test_y, pred), 2)*100
    cm = confusion_matrix(test_y, pred)

    print("SVM Confusion Matrix:\n", cm)

    return f"SVM Accuracy Score: %{acc}\n"

def train_NN():
    model = MLPClassifier(max_iter=1000)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    
    acc = round(accuracy_score(test_y, pred), 2)*100
    cm = confusion_matrix(test_y, pred)

    print("Neural Network Confusion Matrix:\n", cm)

    return f"NN Accuracy Score: %{acc}\n"
