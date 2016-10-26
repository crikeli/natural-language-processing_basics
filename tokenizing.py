# Opinion mining and Sentiment analysis using the Natural Language Tool-kit
# Exercise number 1.

# The code below is a short example demonstrating tokenizing.

import nltk
# Uncomment the following line if this is your first time working with NLTK and you don't already have all the nltk resources downloaded.
# nltk.download()

# Some vocabulary -
# Corpus: A body of text.
# Lexicon: Words and their meanings in certain contexts.
# Token: Any entity that is part of whatever was divided dependent on certain rules.

from nltk.tokenize import sent_tokenize, word_tokenize

TEXT_TO_BE_TOKENIZED = "Wow, what is life even? Like for real!"

tokSent = sent_tokenize(TEXT_TO_BE_TOKENIZED)
tokWord = word_tokenize(TEXT_TO_BE_TOKENIZED)

print "Tokeized Sentences: ",(tokSent)
print "Tokenized Words: ",(tokWord)

# Output in terminal:
# python tokenizing.py
# Tokeized Sentences:  ['Wow, what is life even?', 'Like for real!']
# Tokenized Words:  ['Wow', ',', 'what', 'is', 'life', 'even', '?', 'Like', 'for', 'real', '!']

# Main Takeaways -
# Punctuations are treated as seperate tokens
# We learn that there are certain words that add value to the sentence such as Wow, life, even
# We also learn that there are a few words that don't matter so much such as is & for.
