from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
iris = load_iris()

datasets = train_test_split(iris.data, iris.target,
                            test_size=0.7)

# train_labels - zbior treningowy
# test_labels - zbior testowy
# wartosci dla kazdego z irysow
# Iris Setosa: 0
# Iris Versicolor: 1
# Iris Virginica: 2
train_data, test_data, train_labels, test_labels = datasets
scaler = StandardScaler()

scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

model1 = MLPClassifier(hidden_layer_sizes=(2,), max_iter=2000)

model1.fit(train_data, train_labels)
print("Model 1 (2 ukryte neurony)")
predictions_test = model1.predict(test_data)
print(accuracy_score(predictions_test, test_labels))

model2 = MLPClassifier(hidden_layer_sizes=(3,), max_iter=2000)

model2.fit(train_data, train_labels)
print("Model 2 (3 ukryte neurony)")
predictions_test = model2.predict(test_data)
print(accuracy_score(predictions_test, test_labels))

model3 = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=2000)

model3.fit(train_data, train_labels)
print("Model 3 (2+2 ukryte neurony)")
predictions_test = model3.predict(test_data)
print(accuracy_score(predictions_test, test_labels))
