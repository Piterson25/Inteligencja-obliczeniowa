import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('iris.csv')
df = df.sort_values(by=['species'])

(train_set, test_set) = train_test_split(df, train_size=0.7, random_state=278833)

train_set_X = train_set.iloc[:, :-1].values # Input 
train_set_y = train_set.iloc[:, -1].values # Class
test_set_X = test_set.iloc[:, :-1].values # Input 
test_set_y = test_set.iloc[:, -1].values # Class

knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_11 = KNeighborsClassifier(n_neighbors=11)
nb = GaussianNB()

knn_3.fit(train_set_X , train_set_y)
knn_5.fit(train_set_X , train_set_y)
knn_11.fit(train_set_X , train_set_y)
nb.fit(train_set_X , train_set_y)

knn_3_acc = accuracy_score(test_set_y, knn_3.predict(test_set_X))
knn_5_acc = accuracy_score(test_set_y, knn_5.predict(test_set_X))
knn_11_acc = accuracy_score(test_set_y, knn_11.predict(test_set_X))
nb_acc = accuracy_score(test_set_y, nb.predict(test_set_X))

knn_3_cm = confusion_matrix(test_set_y, knn_3.predict(test_set_X))
knn_5_cm = confusion_matrix(test_set_y, knn_5.predict(test_set_X))
knn_11_cm = confusion_matrix(test_set_y, knn_11.predict(test_set_X))
nb_cm = confusion_matrix(test_set_y, nb.predict(test_set_X))

print(f"Dokładność klasyfikatora 3-NN: {knn_3_acc * 100}%")
print("Macierz błędów dla klasyfikatora 3-NN:")
print(knn_3_cm)

print(f"Dokładność klasyfikatora 5-NN: {knn_5_acc * 100}%")
print("Macierz błędów dla klasyfikatora 5-NN:")
print(knn_5_cm)

print(f"Dokładność klasyfikatora 11-NN: {knn_11_acc * 100}%")
print("Macierz błędów dla klasyfikatora 11-NN:")
print(knn_11_cm)

print(f"Dokładność klasyfikatora Naive Bayes: {nb_acc * 100}%")
print("Macierz błędów dla klasyfikatora Naive Bayes:")
print(nb_cm)

# DD 97.77777777777777%
# 3NN 97.77777777777777%
# 5NN 97.77777777777777%
# 11NN 97.77777777777777%
# NB 95.55555555555556%
