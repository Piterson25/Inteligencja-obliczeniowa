import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('diabetes.csv')
pd.set_option('display.max_columns', None)

target_column = ['class']
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()

X = df[predictors].values
y = df[target_column].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

mlp = MLPClassifier(hidden_layer_sizes=(6, 3), activation='relu', max_iter=500)
mlp.fit(X_train, y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print("Zbior testowy:")

print("Macierz bledu:")
print(confusion_matrix(y_test, predict_test))

print("Dokladnosc:")
print(classification_report(y_test, predict_test))
