from fileProccessor import get_processed_tweets
from featureExtractor import get_word_features_from_tweets
import nltk
import pickle
import time


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in all_word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

start = time.time()
csv_file = 'trainingData.csv'
max_tweets_per_sentiment = 80000
processed_tweets = get_processed_tweets(csv_file, max_tweets_per_sentiment)
all_word_features = get_word_features_from_tweets(processed_tweets)

print "Processing time ", time.time() - start
start = time.time()
all_word_features_dump = open('all_word_features.dump', 'wb')
pickle.dump(all_word_features, all_word_features_dump)
all_word_features_dump.close()
print "all_word_features dumped: time ", time.time() - start
start = time.time()

training_set = nltk.classify.apply_features(extract_features, processed_tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

print "Feature Extraction and train time ", time.time() - start
start = time.time()
classifier_dump = open('classifier.dump', 'wb')
pickle.dump(classifier, classifier_dump)
classifier_dump.close()
print "classifier dumped: time ", time.time() - start
start = time.time()