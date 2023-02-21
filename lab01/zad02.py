import pandas as pd
import matplotlib.pyplot as plt

miasta = pd.read_csv('miasta.csv')
print(miasta.values)


row = pd.DataFrame({'Rok': [2010], 'Gdansk': [460], 'Poznan': [555], 'Szczecin': [405]})
miasta = miasta.append(row, ignore_index=True)
print(miasta)

miasta.plot(kind='line', x='Rok', xticks=miasta.Rok, y='Gdansk', color='red',
            xlabel="Lata", ylabel='Liczba ludnosci (w tys.)', legend=False,
            title='Ludnosc w miastach Polski', style='.-')
plt.show()


miasta.plot(kind='line', x='Rok', style='.-')
plt.show()

print("\nStandaryzacja")

miasta_std = (miasta - miasta.mean()) / miasta.std()
print(miasta_std.mean())
print(miasta_std.std())

print("\nNormalizacja")

miasta_norm = (miasta - miasta.min()) / (miasta.max() - miasta.min())

print(miasta_norm.min())
print(miasta_norm.max())
