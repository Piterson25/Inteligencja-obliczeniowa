from collections import Counter

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

with open('artykul.txt', 'r') as file:
    article = file.read()

tokeny = word_tokenize(article)
print("Liczba słów po tokenizacji:", len(tokeny))

stop_words = set(stopwords.words('english'))
przefiltrowane_tokeny = [word for word in tokeny if word.lower() not in stop_words]
print("Liczba słów po usunięciu stop-words:", len(przefiltrowane_tokeny))

wlasne_stop_words = ['\"\"', ',', '.', '\'\'', '``', '\'s']
stop_words.update(wlasne_stop_words)
przefiltrowane_tokeny = [word for word in przefiltrowane_tokeny if word.lower() not in stop_words]
print("Liczba słów po usunięciu dodatkowych stop-words:", len(przefiltrowane_tokeny))

lematyzer = WordNetLemmatizer()
lematyzowane_tokeny = [lematyzer.lemmatize(word) for word in przefiltrowane_tokeny]
print("Liczba słów po lematyzacji:", len(lematyzowane_tokeny))

word_counts = Counter(lematyzowane_tokeny)
top_10_words = word_counts.most_common(10)
words = [word for word, _ in top_10_words]
counts = [count for _, count in top_10_words]

plt.bar(words, counts)
plt.xlabel('Słowa')
plt.ylabel('Liczba wystąpień')
plt.title('Słowa')
plt.xticks(rotation=45)
plt.show()

# Chmurka tagów
wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_counts)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
