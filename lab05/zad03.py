import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# DOKOŃCZYĆ ZADANIE

df = pd.read_csv("iris.csv") 
df = df.sort_values(by=['species'])

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=278833)

def classify_iris(sl, sw, pl, pw): 
    if pl <= 2.5:
        return("setosa")
    elif pw <= 1.6:
        if pl <= 4.8:
            return("versicolor")
        else:
            return("virginica")
    else: 
        return("virginica")

good_predictions = 0 
len = test_set.shape[0] 

for i in range(len): 
    if classify_iris(test_set[i,0], test_set[i,1], test_set[i,2], test_set[i,3]) == test_set[i,4]: 
        good_predictions = good_predictions + 1 

print(good_predictions) 
print(good_predictions/len*100, "%")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

print("Zbiór treningowy:")
print("Cechy:")
print(X_train)
print("Etykiety:")
print(y_train)

print("Zbiór testowy:")
print("Cechy:")
print(X_test)
print("Etykiety:")
print(y_test)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

text_representation = classifier.tree_.pretty_text()
print(text_representation)

plt.figure(figsize=(20,10))
plot_tree(classifier, filled=True, rounded=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

accuracy = classifier.score(X_test, y_test)
print("Dokładność klasyfikatora: {:.2f}%".format(accuracy*100))

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Macierz błędów:")
print(cm)