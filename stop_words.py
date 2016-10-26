# Opinion mining and Sentiment analysis using the Natural Language Tool-kit
# Exercise number 2.

# In working with making sense of natural language processing, the most important part of
# the process is filtering/pre-processing data. The words that are not of use in NLP are called stopwords.

# Accessing stopwords(useless words) in the nltk corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

EXAMPLE_SENTENCE = "Hi Kelin! Is jam sort of like butter or is it just a thing on its own?"

STOP_WORDS = set(stopwords.words('english'))

WORD_TOKENS = word_tokenize(EXAMPLE_SENTENCE)

FILTERED_SENTENCE = []

for w in WORD_TOKENS:
    if w not in STOP_WORDS:
        FILTERED_SENTENCE.append(w)

print "Word Tokens: ",WORD_TOKENS
print "Filtered Sentence: ",FILTERED_SENTENCE

# Output
# python stop_words.py
# Word Tokens:  ['Hi', 'Kelin', '!', 'Is', 'jam', 'sort', 'of', 'like', 'butter', 'or', 'is', 'it', 'just', 'a', 'thing', 'on', 'its', 'own', '?']
# Filtered Sentence:  ['Hi', 'Kelin', '!', 'Is', 'jam', 'sort', 'like', 'butter', 'thing', '?']
