import snscrape.modules.twitter as sntwitter
import csv

def get_tweets(hashtag, num):
    tweets = []

    query = f'#{hashtag}'
    for tweet in sntwitter.TwitterHashtagScraper(query).get_items():

        tweets.append({
            'rawContent': str(tweet.rawContent).replace('\n', ''),
            'date': tweet.date,
            'lang': tweet.lang
        })

        if len(tweets) >= num:
            break

    with open('tweets.csv', 'w', encoding='utf-8', newline='') as file:
        fieldnames = ['rawContent', 'date', 'lang']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for tweet in tweets:
            writer.writerow(tweet)

    print("Pomyślnie zapisano do pliku tweets.csv")

hashtag = 'premierleague'
num_tweets = 10000

get_tweets(hashtag, num_tweets)
