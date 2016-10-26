# Opinion mining and Sentiment analysis using the Natural Language Tool-kit
# Exercise number 4.

import nltk
from nltk.corpus import state_union
# An unsupervised machine learning tokenizer(comes pre-trained.)
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

# We train the PunktSentenceTokenizer to a clinton speech in 1993
TRAIN_TEXT = state_union.raw("1993-Clinton.txt")
# We train the PunktSentenceTokenizer to a clinton speech in 1994
SAMPLE_TEXT = state_union.raw("1994-Clinton.txt")

# Actual training of the PunktSentenceTokenizer
SENTENCE_TOKENIZER = PunktSentenceTokenizer(TRAIN_TEXT)

# Tokenizing using the trained model.
TOKENIZED = SENTENCE_TOKENIZER.tokenize(SAMPLE_TEXT)

# Processing function.
def process_content():
    try:
        for i in TOKENIZED:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print"POS of tagged words", (tagged)
    except Exception as e:
        print(str(e))

process_content()

# Output
# [(u'People', 'NNS'), (u'have', 'VBP'), (u'to', 'TO'), (u'take', 'VB'), (u'their', 'PRP$'), (u'kids', 'NNS'), (u'to', 'TO'), (u'get', 'VB'), (u'immunized', 'VBN'), (u'.', '.')]
# [(u'We', 'PRP'), (u'should', 'MD'), (u'all', 'DT'), (u'take', 'VB'), (u'advantage', 'NN'), (u'of', 'IN'), (u'preventive', 'JJ'), (u'care', 'NN'), (u'.', '.')]

# POS cheat sheet for above output reference.

# CC Coordinating conjunction
# CD Cardinal number
# DT Determiner
# EX Existential there
# FW Foreign word
# IN Preposition or subordinating conjunction
# JJ Adjective
# JJR Adjective, comparative
# JJS Adjective, superlative
# LS List item marker
# MD Modal
# NN Noun, singular or mass
# NNS Noun, plural
# NNP Proper noun, singular
# NNPS Proper noun, plural
# PDT Predeterminer
# POS Possessive ending
# PRP Personal pronoun
# PRP$ Possessive pronoun
# RB Adverb
# RBR Adverb, comparative
# RBS Adverb, superlative
# RP Particle
# SYM Symbol
# TO to
# UH Interjection
# VB Verb, base form
# VBD Verb, past tense
# VBG Verb, gerund or present participle
# VBN Verb, past participle
# VBP Verb, non­3rd person singular present
# VBZ Verb, 3rd person singular present
# WDT Wh­determiner
# WP Wh­pronoun
# WP$ Possessive wh­pronoun
# WRB Wh­adverb
