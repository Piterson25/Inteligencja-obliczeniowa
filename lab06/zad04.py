import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

df = pd.read_csv('diabetes.csv')
pd.set_option('display.max_columns', None)

target_column = ['class']
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()

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

    plot_model(model, show_shapes=True, show_layer_names=True, to_file='models.png')


# Rozne optimizery i funkcje aktywacji
for optimizer_name in ['adam', 'sgd']:
    for activation_name in ['relu', 'sigmoid']:
        model = Sequential()
        model.add(Dense(6, input_dim=8, activation=activation_name))
        model.add(Dense(3, activation=activation_name))
        model.add(Dense(2, activation='softmax'))
        train_and_visualize_model(model, optimizer_name, activation_name)

# Może dojść do przeuczenia
# Błąd na zbiorze treningowym zmniejsza się, a na testowym przestaje się
# zwiększać i może nawet urosnąć
# Dwie krzywe na wykresie oddalone od siebie mogą wskazywać na przeuczenie
