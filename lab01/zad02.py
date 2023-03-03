import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = 'C:\Studia\Inteligencja obliczeniowa\Inteligencja-obliczeniowa\lab01\miasta.csv'

miasta = pd.read_csv(path)
print(miasta.values)


row = pd.DataFrame({'Rok': [2010], 'Gdansk': [460], 'Poznan': [555], 'Szczecin': [405]})
miasta = miasta.append(row, ignore_index=True)
print(miasta)

miasta.plot(kind='line', x='Rok', xticks=np.arange(miasta.Rok.min(), miasta.Rok.max() + 1, 10), y='Gdansk', color='red',
            xlabel="Lata", ylabel='Liczba ludnosci [w tys.]', legend=False,
            title='Ludnosc w miastach Polski', style='.-')
plt.show()


miasta.plot(kind='line', x='Rok', xticks=np.arange(miasta.Rok.min(), miasta.Rok.max() + 1, 10), style='.-')
plt.show()

print("\nStandaryzacja")

def standaryzacja(col):
    return (col - col.mean()) / col.std()

miasta_standaryzowane = miasta[['Gdansk', 'Poznan', 'Szczecin']].apply(standaryzacja, axis=0)

print(miasta_standaryzowane)

srednia = miasta_standaryzowane.mean()
odchylenie = miasta_standaryzowane.std()
print("\nŚrednia:")
print(srednia)
print("\nOdchylenie standardowe:")
print(odchylenie)



print("\nNormalizacja")

def normalizacja(col):
    return (col - col.min()) / (col.max() - col.min())

miasta_normalizowane = miasta[['Gdansk', 'Poznan', 'Szczecin']].apply(normalizacja, axis=0)

print(miasta_normalizowane)

minimal = miasta_normalizowane.min()
maximum = miasta_normalizowane.max()
print("\nMin:")
print(minimal)
print("\nMax:")
print(maximum)
