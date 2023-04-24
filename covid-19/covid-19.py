import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Covid Data.csv')
df.head()

df.isna().sum()
df.describe()

wyniki_testu = df['CLASIFFICATION_FINAL'].unique()
plt.bar(wyniki_testu, df.CLASIFFICATION_FINAL.value_counts())

plt.xlabel("Wyniki testu")
plt.ylabel("Liczba wystąpień")
plt.title("Rozkład wyników testu")

plt.show()
