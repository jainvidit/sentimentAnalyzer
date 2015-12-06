import pickle
import time


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in all_word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

start = time.time()

all_word_features_dump = open('all_word_features.dump')
all_word_features = pickle.load(all_word_features_dump)
all_word_features_dump.close()
print "Feature read time ", time.time() - start
start = time.time()


classifier_dump = open('classifier.dump')
classifier = pickle.load(classifier_dump)
classifier_dump.close()
print "Classifier read time ", time.time() - start

tweet = raw_input('Enter your tweet')
sentiment = classifier.classify(extract_features(tweet.split()))
print sentiment
