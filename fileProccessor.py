import csv
from nltk.corpus import stopwords


def remove_noise(line):
    set_stopwords = set(stopwords.words('english'))
    return [word.lower().strip() for word in line.split() if len(word) >= 3\
            and word[0] != '@' and word.find('http') == -1 and not(word in set_stopwords)]


def get_sentiment(key):
    sentiments = {'0': 'negative', '2': 'neutral', '4': 'positive'}
    sentiment = sentiments[key]
    return sentiment


def get_processed_tweets(csv_file_name, max_tweets_per_sentiment):
    processed_tweets = []
    with open(csv_file_name, 'rb') as csv_file:
        raw_tweet_data = csv.reader(csv_file, delimiter=',', quotechar='"')
        count = {'0': 0, '2': 0, '4': 0}
        print "Tweets  = ", max_tweets_per_sentiment
        for tweet in raw_tweet_data:
            count[tweet[0]] += 1
            if count[tweet[0]] <= max_tweets_per_sentiment:
                words = remove_noise(tweet[5])
                processed_tweets.append((words, get_sentiment(tweet[0])))
    return processed_tweets




