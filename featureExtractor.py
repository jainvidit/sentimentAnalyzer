import nltk


def get_words_in_tweet(tweet):
    word, sentiment = tweet
    return word


def get_words_in_tweets(tweets):
    all_words = []
    for tweet in tweets:
        all_words.extend(get_words_in_tweet(tweet))
    return all_words


def get_frequency_distribution(word_list):
    frequency_distribution = nltk.FreqDist(word_list)
    return frequency_distribution


def get_keys(frequency_distribution):
    keys = frequency_distribution.keys()
    return keys


def get_word_features(word_list):
    frequency_distribution = get_frequency_distribution(word_list)
    word_features = get_keys(frequency_distribution)
    return word_features


def get_word_features_from_tweets(tweets):
    words = get_words_in_tweets(tweets)
    word_features = get_word_features(words)
    return word_features

