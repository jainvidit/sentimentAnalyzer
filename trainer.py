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

csv_file = 'trainingData.csv'
max_tweets_per_sentiment = 1000
processed_tweets = get_processed_tweets(csv_file, max_tweets_per_sentiment)
all_word_features = get_word_features_from_tweets(processed_tweets)

all_word_features_dump = open('all_word_features.dump', 'wb')
pickle.dump(all_word_features, all_word_features_dump)
all_word_features_dump.close()

training_set = nltk.classify.apply_features(extract_features, processed_tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier_dump = open('classifier.dump', 'wb')
pickle.dump(classifier, classifier_dump)
classifier_dump.close()

