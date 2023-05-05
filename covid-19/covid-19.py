import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Covid Data.csv')
df = df.head(20000)
df.head()

df.isna().sum()
df.describe()

wyniki_testu = df['CLASIFFICATION_FINAL'].unique()
plt.bar(wyniki_testu, df.CLASIFFICATION_FINAL.value_counts())

plt.xlabel("Wyniki testu")
plt.ylabel("Liczba wystąpień")
plt.title("Rozkład wyników testu")

plt.show()

plt.show()

num_positive = len(df[df.CLASIFFICATION_FINAL <= 3])
num_negative = len(df[df.CLASIFFICATION_FINAL > 3])

percent_positive = num_positive / len(df) * 100
percent_negative = num_negative / len(df) * 100

print(f"Test pozytywny: {percent_positive:.2f}%")
print(f"Test negatywny: {percent_negative:.2f}%")

labels = ['Test pozytywny', 'Test negatywny']
sizes = [percent_positive, percent_negative]
colors = ['green', 'red']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

plt.axis('equal')
plt.title('Wyniki testu COVID-19')

plt.show()

from sklearn.model_selection import train_test_split

X = df.drop(['CLASIFFICATION_FINAL'], axis=1)
y = df['CLASIFFICATION_FINAL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

dtc1 = DecisionTreeClassifier(max_depth=5)
dtc1.fit(X_train, y_train)
y_pred = dtc1.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Decision Tree (max_depth=5): Accuracy={acc:.4f}")
print(f"Confusion Matrix:\n{cm}")

dtc2 = DecisionTreeClassifier(max_depth=None)
dtc2.fit(X_train, y_train)
y_pred = dtc2.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Decision Tree (max_depth=None): Accuracy={acc:.4f}")
print(f"Confusion Matrix:\n{cm}")

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Naive Bayes: Accuracy={acc:.4f}")
print(f"Confusion Matrix:\n{cm}")

from sklearn.neighbors import KNeighborsClassifier

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)
y_pred = knn3.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"k-NN (k=3): Accuracy={acc:.4f}")
print(f"Confusion Matrix:\n{cm}")

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)
y_pred = knn5.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"k-NN (k=5): Accuracy={acc:.4f}")
print(f"Confusion Matrix:\n{cm}")

knn7 = KNeighborsClassifier(n_neighbors=7)
knn7.fit(X_train, y_train)
y_pred = knn7.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"k-NN (k=7): Accuracy={acc:.4f}")
print(f"Confusion Matrix:\n{cm}")

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

target_column = ['CLASIFFICATION_FINAL']
predictors = list(set(list(df.columns)) - set(target_column))
df[predictors] = df[predictors] / df[predictors].max()

X = df[predictors].values
y = df[target_column].values.ravel()

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def train_and_visualize_model(model, optimizer_name, activation_name):
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test), verbose=0)

    # Dokładnosc i Macierz bledu
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=-1)
    y_test_classes = y_test.argmax(axis=-1)
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
    print("Zbior testowy:")
    print("Dokladnosc:", accuracy)
    print("Macierz bledu:\n", conf_matrix)

    # Wykres krzywej uczenia się
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Test')
    plt.title(f'Model loss ({optimizer_name}, {activation_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


model = Sequential()
model.add(Dense(10, input_dim=20, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
train_and_visualize_model(model, 'adam', 'sigmoid')

model = Sequential()
model.add(Dense(10, input_dim=20, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(2, activation='softmax'))
train_and_visualize_model(model, 'sgd', 'relu')

model = Sequential()
model.add(Dense(12, input_dim=20, activation='relu'))
model.add(Dense(2, activation='softmax'))
train_and_visualize_model(model, 'adam', 'relu')
