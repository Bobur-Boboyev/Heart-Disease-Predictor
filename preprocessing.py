import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocessing_data(path):
    data = pd.read_csv(path)

    x = data.drop("target", axis=1)
    y = data["target"]
 
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test, scaler
