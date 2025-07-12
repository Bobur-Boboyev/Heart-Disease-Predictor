from train import train_KNN, train_SVM, train_NN
from preprocessing import preprocessing_data
import matplotlib.pyplot as plt
import pandas as pd

x_train, x_test, y_train, y_test, scaler = preprocessing_data('data/heart.csv')
model_KNN, acc_KNN = train_KNN()
model_SVM, acc_SVM = train_SVM()
model_NN, acc_NN = train_NN()

def new_patient():
    print("\nPlease enter the following patient details:")

    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    input_data = []
    for feature in features:
        value = float(input(f"{feature}: "))
        input_data.append(value)
    patient_df = pd.DataFrame([input_data], columns=features)
    patient_scaled = scaler.transform(patient_df)
    return patient_scaled

while True:
    print("\n====== Heart Disease Predictor ======")
    print("1 - Accuracy of KNN")
    print("2 - Accuracy of SVM")
    print("3 - Accuracy of Neural Network")
    print("4 - Predict for new patient")
    print("5 - Visualize Accuracy")
    print("0 - Exit")
    choice = input("Enter your choice: ")

    if choice == "1":
        print(f"KNN Accuracy: %{acc_KNN}")
    elif choice == "2":
        print(f"SVM Accuracy: %{acc_SVM}")
    elif choice == "3":
        print(f"Neural Network Accuracy: %{acc_NN}")
    elif choice == "4":
        user_data = new_patient()
        pred_knn = model_KNN.predict(user_data)[0]
        pred_svm = model_SVM.predict(user_data)[0]
        pred_nn = model_NN.predict(user_data)[0]

        print("\n---- Prediction Results ----")
        print(f"KNN predicts: {'Heart Disease' if pred_knn == 1 else 'No Heart Disease'}")
        print(f"SVM predicts: {'Heart Disease' if pred_svm == 1 else 'No Heart Disease'}")
        print(f"Neural Network predicts: {'Heart Disease' if pred_nn == 1 else 'No Heart Disease'}")


    elif choice == "5":
        models = ["KNN", "SVM", "NN"]
        accs = [acc_KNN, acc_SVM, acc_NN]
        plt.bar(models, accs)
        plt.title("Model Accuracy Comparison")
        plt.ylim(0, 100)
        for i, acc in enumerate(accs):
            plt.text(i, acc - 5, f"%{acc}", ha="center", color="white")
        plt.ylabel("Accuracy (%)")
        plt.show()
    elif choice == "0":
        break
    else:
        print("Invalid input.")
