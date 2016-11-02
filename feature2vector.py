# Opinion mining and Converting words to Features using the Natural Language Tool-kit
# Exercise number 5.

import nltk
import random
# Movie Reviews are 1000 positive & 1000 negative reviews.
from nltk.corpus import movie_reviews

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)

# print documents[1]

# Declaring an empty list
all_words = []

# Converting everything to lowercase.
for w in movie_reviews.words():
    all_words.append(w.lower())

# We find the most common words.
all_words = nltk.FreqDist(all_words)

# print "15 most common words", all_words.most_common(15)
# Output : 15 most common words [(u',', 77717), (u'the', 76529), (u'.', 65876), (u'a', 38106), (u'and', 35576), (u'of', 34123), (u'to', 31937), (u"'", 30585), (u'is', 25195), (u'in', 21822), (u's', 18513), (u'"', 17612), (u'it', 16107), (u'that', 15924), (u'-', 15595)]
print "Fantastic occurs this many times: ",all_words["fantastic"]
# Stupid occurs this many times:  253

# # This feature contains the top 3000 most common words
# word_features = list(all_words.keys())[:3000]
#
# def find_features(document):
#     words = set(document)
#     features = {}
#     for w in word_features:
#         features[w] = (w in words)
#
#     return features
#
# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
#
# featuresets = [(find_features(rev), category) for (rev, category) in documents]
