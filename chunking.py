# Opinion mining and Sentiment analysis using the Natural Language Tool-kit
# Exercise number 5.

# Chunking groups similar words into meaningful "chunks"
# Chunking combines POS tagging with regular expressions.

import nltk
from nltk.corpus import state_union
# An unsupervised machine learning tokenizer(comes pre-trained.)
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

# Regex reference
# + = match 1 or more
# ? = match 0 or 1 repetitions.
# * = match 0 or MORE repetitions
# . = Any character except a new line

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
            # A chunkGram is used to link similar chunks of words together.
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            # chunkParser uses the above defined chunkGram to specify the behaviour of the parser.
            chunkParser = nltk.RegexpParser(chunkGram)
            # parses all the taged words according to the definition of chunkParser
            chunked = chunkParser.parse(tagged)
            # chunked.draw()

            for subtree in chunked.subtrees(filter=lambda t: t.label() == "Chunk"):
                print subtree
            # Sample Output
            # (Chunk President/NNP Roosevelt/NNP)
            # (Chunk President/NNP Truman/NNP)
            # (Chunk President/NNP Nixon/NNP)
            # (Chunk President/NNP Carter/NNP)
            # (Chunk PRESIDENT/NNP BILL/NNP CLINTON/NNP)
            # (Chunk A/NNP JOINT/NNP SESSION/NNP)
            # (Chunk THE/NNP CONGRESS/NNP ON/NNP THE/NNP STATE/NNP)

    except Exception as e:
        print(str(e))

process_content()
