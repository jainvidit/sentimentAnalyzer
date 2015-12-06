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

def evaluate_features(feature_select):
	posFeatures = []
	negFeatures = []
	#http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
	#breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
	with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
		for i in posSentences:
			posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			posWords = [feature_select(posWords), 'pos']
			posFeatures.append(posWords)
	with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
		for i in negSentences:
			negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			negWords = [feature_select(negWords), 'neg']
			negFeatures.append(negWords)

	
	#selects 3/4 of the features to be used for training and 1/4 to be used for testing
	posCutoff = int(math.floor(len(posFeatures)*3/4))
	negCutoff = int(math.floor(len(negFeatures)*3/4))
	trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
	testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

	#trains a Naive Bayes Classifier
	classifier = NaiveBayesClassifier.train(trainFeatures)	

	#initiates referenceSets and testSets
	referenceSets = collections.defaultdict(set)
	testSets = collections.defaultdict(set)	

	#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
	for i, (features, label) in enumerate(testFeatures):
		referenceSets[label].add(i)
		predicted = classifier.classify(features)
		testSets[predicted].add(i)	

	#prints metrics to show how well the feature selection did
	print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
	print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
	print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
	print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
	print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
	print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
	classifier.show_most_informative_features(10)

#creates a feature selection mechanism that uses all words
def make_full_dict(words):
	return dict([(word, True) for word in words])

#scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
def create_word_scores():
	#creates lists of all positive and negative words
	posWords = []
	negWords = []
	with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
		for i in posSentences:
			posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			posWords.append(posWord)
	with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
		for i in negSentences:
			negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			negWords.append(negWord)
	posWords = list(itertools.chain(*posWords))
	negWords = list(itertools.chain(*negWords))

	#build frequency distibution of all words and then frequency distributions of words within positive and negative labels
	word_fd = FreqDist()
	cond_word_fd = ConditionalFreqDist()
	for word in posWords:
		word_fd[word.lower()] += 1
		cond_word_fd['pos'][word.lower()] += 1
	for word in negWords:
		word_fd[word.lower()] += 1
		cond_word_fd['neg'][word.lower()] += 1

	#finds the number of positive and negative words, as well as the total number of words
	pos_word_count = cond_word_fd['pos'].N()
	neg_word_count = cond_word_fd['neg'].N()
	total_word_count = pos_word_count + neg_word_count

	#builds dictionary of word scores based on chi-squared test
	word_scores = {}
	for word, freq in word_fd.iteritems():
		pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
		neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
		word_scores[word] = pos_score + neg_score

	return word_scores

#finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
	best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
	best_words = set([w for w, s in best_vals])
	return best_words

#creates feature selection mechanism that only uses best words
def best_word_features(words):
	return dict([(word, True) for word in words if word in best_words])

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