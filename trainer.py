from fileProccessor import get_processed_tweets
from featureExtractor import get_word_features_from_tweets
import nltk
import pickle


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in all_word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

csv_file = 'training.1600000.processed.noemoticon.csv'
max_tweets_per_sentiment = 1000
processed_tweets = get_processed_tweets(csv_file,max_tweets_per_sentiment)
all_word_features = get_word_features_from_tweets(processed_tweets)
training_set = nltk.classify.apply_features(extract_features, processed_tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier_dump = open('classifier.dump', 'wb')
pickle.dump(classifier, classifier_dump)
classifier_dump.close()

