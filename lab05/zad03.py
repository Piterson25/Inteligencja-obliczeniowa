import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix

df = pd.read_csv("iris.csv")
df = df.sort_values(by=['species'])

(train_set, test_set) = train_test_split(df, test_size=0.3, train_size=0.7, random_state=278833)

# Wyciąganie z odpowiednich kolumn (bez ostatniej)
train_set_X = train_set.iloc[:, :-1].values  # Input
train_set_y = train_set.iloc[:, -1].values  # Class
test_set_X = test_set.iloc[:, :-1].values  # Input
test_set_y = test_set.iloc[:, -1].values  # Class

dtc = DecisionTreeClassifier()

dtc.fit(train_set_X, train_set_y)

text_representation = export_text(dtc)
print(text_representation)

accuracy = dtc.score(test_set_X, test_set_y)
print("Dokładność klasyfikatora:", accuracy)

y_pred = dtc.predict(test_set_X)
cm = confusion_matrix(test_set_y, y_pred)
print("Macierz błędów:")
print(cm)

# Dokładność wyszła lepsza niż w poprzednim zadaniu - ~97%
