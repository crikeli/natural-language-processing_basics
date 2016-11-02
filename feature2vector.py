# Opinion mining and Converting words to Features using the Natural Language Tool-kit
# Exercise number 5.

import nltk
import random
# Movie Reviews are 1000 positive & 1000 negative reviews.
from nltk.corpus import movie_reviews
# Wrapper to include scikit learn algos within nltk
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

# Importing various SK Algorithms
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)

# print documents[1]

# Declaring an empty list
all_words = []

# Converting every word to lowercase.
for w in movie_reviews.words():
    all_words.append(w.lower())

# We find the most common words.
# FreqDist most common to least common {word: numberofWords}
all_words = nltk.FreqDist(all_words)

# print "15 most common words", all_words.most_common(15)
# Output : 15 most common words [(u',', 77717), (u'the', 76529), (u'.', 65876), (u'a', 38106), (u'and', 35576), (u'of', 34123), (u'to', 31937), (u"'", 30585), (u'is', 25195), (u'in', 21822), (u's', 18513), (u'"', 17612), (u'it', 16107), (u'that', 15924), (u'-', 15595)]
# print "Fantastic occurs this many times: ",all_words["fantastic"]
# Fantastic occurs this many times:  77

# # This feature contains the top 3000 most common words in the pos and neg docs.
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
# print featuresets

# Naive Bayes Algorithm Implementation
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# posterior = prioroccurences * likelihood / evidence
# classifier = nltk.NaiveBayesClassifier.train(training_set)

# rb is read as bytes
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print "Original Accuracy Percentage for NB: ", nltk.classify.accuracy(classifier, testing_set) * 100
classifier.show_most_informative_features(15)

# Output
# Accuracy Percentage for NaiveBayesClassifier :  70.0
# Most Informative Features
#                insulting = True              neg : pos    =     10.6 : 1.0
#             refreshingly = True              pos : neg    =      8.4 : 1.0
#                  wasting = True              neg : pos    =      8.3 : 1.0
#                     sans = True              neg : pos    =      7.7 : 1.0
#               mediocrity = True              neg : pos    =      7.7 : 1.0
#                dismissed = True              pos : neg    =      7.0 : 1.0
#                   fabric = True              pos : neg    =      6.3 : 1.0
#                uplifting = True              pos : neg    =      6.2 : 1.0
#                   stinks = True              neg : pos    =      5.8 : 1.0
#               cronenberg = True              pos : neg    =      5.7 : 1.0
#                     lang = True              pos : neg    =      5.7 : 1.0
#                  topping = True              pos : neg    =      5.7 : 1.0
#              bruckheimer = True              neg : pos    =      5.7 : 1.0
#                    wires = True              neg : pos    =      5.7 : 1.0
#             effortlessly = True              pos : neg    =      5.6 : 1.0


# Using Pickle to save the trained portion of the algorithm to avoid re-training.
# wb is write as bytes
# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print "Accuracy Percentage for MNB: ", nltk.classify.accuracy(MNB_classifier, testing_set) * 100
# Output : Accuracy Percentage for MNB:  69.0

# BernoulliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print "Accuracy Percentage for BernoulliNB: ", nltk.classify.accuracy(BernoulliNB_classifier, testing_set) * 100
# Output : Accuracy Percentage for BernoulliNB:  70.0

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
# Output : ('LogisticRegression_classifier accuracy percent:', 61.0)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
# Output : ('SGDClassifier_classifier accuracy percent:', 69.0)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
# Output : ('SVC_classifier accuracy percent:', 44.0)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
# Output : ('LinearSVC_classifier accuracy percent:', 62.0)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
# Output : ('NuSVC_classifier accuracy percent:', 67.0)
