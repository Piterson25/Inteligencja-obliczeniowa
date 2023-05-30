import re
import nltk
from langdetect import detect, LangDetectException
from translate import Translator


def preprocess_tweet(tweet):
    # Usunięcie użytkowników z znakiem @
    tweet = re.sub(r'@\w+\s?', '', tweet)

    # Usunięcie znaków specjalnych
    tweet = re.sub(r'[^\w\s]', '', tweet)

    # Tokenizacja
    tokens = nltk.word_tokenize(tweet)

    # Usunięcie stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Normalizacja - sprowadzenie do małych liter
    tokens = [token.lower() for token in tokens]

    # Połączenie tokenów z powrotem w tekst
    preprocessed_tweet = ' '.join(tokens)

    return preprocessed_tweet


def preprocess_and_translate_tweets(filename):
    preprocessed_tweets = []

    with open(filename, 'r', encoding='utf-8') as file:
        tweets = file.readlines()

    for tweet in tweets:
        # Usunięcie znaku nowej linii
        tweet = tweet.strip()

        # Wykrywanie języka tweeta
        try:
            detected_language = detect(tweet)
        except LangDetectException:
            detected_language = 'unknown'

        # Tłumaczenie na angielski, jeśli wykryty język jest inny niż angielski
        if detected_language != 'en':
            translator = Translator(to_lang='en')
            translation = translator.translate(tweet)
            translated_tweet = translation
        else:
            translated_tweet = tweet

        # Preprocessowanie tweeta
        preprocessed_tweet = preprocess_tweet(translated_tweet)
        preprocessed_tweets.append(preprocessed_tweet)

    return preprocessed_tweets


# Przetwarzanie i tłumaczenie tweetów
preprocessed_tweets = preprocess_and_translate_tweets('tweets.txt')

# Wyświetlenie wyników
for tweet in preprocessed_tweets:
    print(tweet)