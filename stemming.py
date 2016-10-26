# Opinion mining and Sentiment analysis using the Natural Language Tool-kit
# Exercise number 3.

# Stemming shortens the lookup time of a word and normalizes a sentence.
# Working with the porterstemmer algorithm:

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Assigning porter_stemmer to instantiating the PorterStemmer algorithm.
porter_stemmer = PorterStemmer()

EXAMPLE_STEMMING_WORDS = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in EXAMPLE_STEMMING_WORDS:
    print "Stemmed Word Results: ",(porter_stemmer.stem(w))

# output -
# python stemming.py
# Stemmed Word Results:  python
# Stemmed Word Results:  python
# Stemmed Word Results:  python
# Stemmed Word Results:  python
# Stemmed Word Results:  pythonli

# Discrepency in the last result*** NO IDEA WHY! :(

NEW_EXAMPLE = "It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly atleast once."

WORDS = word_tokenize(NEW_EXAMPLE)

for w in WORDS:
    print "Stemmed Results 2: ",(porter_stemmer.stem(w))

# Output -
# Stemmed Results 2:  It
# Stemmed Results 2:  is
# Stemmed Results 2:  veri
# Stemmed Results 2:  import
# Stemmed Results 2:  to
# Stemmed Results 2:  be
# Stemmed Results 2:  pythonli
# Stemmed Results 2:  while
# Stemmed Results 2:  you
# Stemmed Results 2:  are
# Stemmed Results 2:  python
# Stemmed Results 2:  with
# Stemmed Results 2:  python
# Stemmed Results 2:  .
# Stemmed Results 2:  All
# Stemmed Results 2:  python
# Stemmed Results 2:  have
# Stemmed Results 2:  python
# Stemmed Results 2:  poorli
# Stemmed Results 2:  atleast
# Stemmed Results 2:  onc
# Stemmed Results 2:  .

# Interesting results however there are cetain results such as veri and onc that does not quiet make sense!
