# Utils to pre-process dataset

import torch

# For file reading and writing
import os
from io import open

# For processing the encode of chars
import unicodedata
import re

# For padding
import itertools


#####################################################################
# Deal with original Dataset

def printLines(filename, n=5):
    """
    Print some lines of this file
    :param filename:
    :param n: the num of lines you want to print
    :return: none
    """
    # Use model:'rb' to read file in binary
    with open(filename, 'rb') as file:
        lines = file.readlines()
    for line in lines[:n]:
        print(line)

# Get pairs of query/answer from original dataset
# 1. Split movie_lines.txt into (id_line, id_char, id_mv, char, text)
def loadLines(filename, fields):
    """
    Get lines from  original dataset, and split by the fields
    :param filename: original dataset of lines
    :param fields: classes of different terms in this line
    :return: [dict] { id_line:{fields:value} }
    """
    lines = {}
    with open(filename, 'r', encoding='iso-8859-1') as file:
        # split by fields:
        for line in file:
            values = line.split(' +++$+++ ')
            # dict_lines: to save values of each line as a dict
            dict_line = {}
            for i, field in enumerate(fields):
                dict_line[field] = values[i]
            # append it into lines
            lines[dict_line['id_line']] = dict_line
    return lines

# 2. Gather lines by the list of lines in conv.txt
def loadConv(filename, fields, lines):
    """
    Get the list of lines in one conversation, then gather them from lines
    :param filename: conv.txt
    :param fields: fields in conv.txt
    :param lines: lines from 1st step
    :return: [list]: [ {fields of conv} ]
    """
    conv = []
    with open(filename, 'r', encoding='iso-8859-1') as file:
        for line in file:
            values = line.split(' +++$+++ ')
            # inner dict
            dict_line = {}
            for i, field in enumerate(fields):
                dict_line[field] = values[i]
                # dict_line: {xxx, ... , id_utterance:["'Lxxx', 'Lxxx'"]}
                # Use eval() to change id_utterance into a list of string
            ids_lines = eval(dict_line['ids_utterance'])
            dict_line['lines'] = []
            for id_line in ids_lines:
                dict_line['lines'].append(lines[id_line])
            # dict_line['lines']: [{L1} {L2} ... ]
            conv.append(dict_line)
    # What is the most important in the conv is the dict_line['lines']
    # which contains a dict of id_lines and id_lines['text'] as sentence
    return conv

# 3. Get query/answer pairs from conv
def extractPairs(conv):
    """
    Extract query/answer pairs from conv[i]['lines']['text']
    ith is query, i+1th is answer, and abandon the last sentence
    which has no answer
    :param conv: conv from step 2
    :return: [list]: ['query, answer', ...]
    """
    pairs = []
    for term in conv:
        # trem: dict_lien in step2
        # -1 to abandon the last sentence
        for i in range(len(term['lines']) - 1):
            query = term['lines'][i]['text'].strip()
            answer = term['lines'][i+1]['text'].strip()
            pair = [query, answer]
            ## dont forget to drop the empty one
            if query and answer:
                pairs.append(pair)
    return pairs

### Remained issues:
''' 
    1. eval()
    2.codecs.decode(char, "unicode-escape")
    3. csv.writer()
'''

##################################################################
# Create Vocabulay for our dataset
# The use of Vocabulary is to change every word in the sentence into index
# which could be processed by our DNN model.

# Pre-defined token:
PAD_token = 0
SOS_token = 1
EOS_token = 2

# Define the class: Vocabulary
class Vocabulary():

    def __init__(self, name):
        self.name = name # the name of our vocabulary
        self.trimmed = False # is this vocabulary trimmed?
        self.num_words = 3 # Num of words in our vocabulary, which contains init three pre-defined tokens
        self.word2index = {} # from word to index
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"} # pre-defined tokens
        self.word_counts = {} # count each word in our vocabulary

    def add_word(self, word):
        """
        Add a word into vocabulary
        :param word: str
        :return: none
        """
        # if it's a new word
        if word not in self.word2index:
            # add it into word2index
            self.index2word[self.num_words] = word
            self.word2index[word] = self.num_words
            self.word_counts[word] = 1
            self.num_words += 1
        else: # it has been in the vocabulary
            self.word_counts[word] += 1

    def add_sentence(self, sentence):
        """
        Add each word into vocabulary by add_word
        :param sentence: str
        :return: none
        """
        for word in sentence.split(' '):
            self.add_word(word)

    def trim(self, min_count):
        """
        Remove words with count < min_count
        :param min_count: int, the threshould of trimming
        :return: none, but update the vocabulary
        """
        # judge if this vocabulary has been trimmed
        if self.trimmed:
            return
        self.trimmed = True
        # list of word kept
        keep_words = []
        # judge each word in the vocabulary
        for word, counts in self.word_counts.items():
            if counts >= min_count:
                keep_words.append(word)
        # Print the ratio of kept words
        print("keep_words {} / {} = {:.4f}".format(
            len(keep_words), len(self.word2index), len(keep_words)/len(self.word2index)
        ))
        # Update our vocabulary
        self.word2index = {}
        self.word_counts = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.add_word(word)

#####################################################################
# Funcs to normalize our input words and chars

# Unicode->ASCII
def unicode2Ascii(str):
    """
    :param str: str in unicode
    :return: str in ascii
    """
    return "".join(
        char for char in unicodedata.normalize('NFD', str)
        if unicodedata.category(char) != 'Mn'
        )

# Normalization
def normalize(str):
    """
    normalize str
    :param str: str
    :return: normalized str
    """
    # lowercase, strip and unicode2ascii
    str = unicode2Ascii(str.lower().strip())
    # Make every punctuation into a word by adding a \s in front of them and behind them
    str = re.sub(r"([.!?])", r" \1", str)
    # Replace other char into \s
    str = re.sub(r"[^a-zA-Z.!?]+", r" ", str)
    # Remove useless spaces
    str = re.sub(r"\s+", r" ", str).strip()
    return str

# Read query/answer pairs and return vocabulary
def get_vocabulary_pairs(filename, corpus_name):
    """
    Read the formatted file and get Q/A pairs and an empty vocabulary
    :param filename: formatted file
    :param corpus_name: the name of your corpus
    :return: vocabulary(Vocabulary), pais(list)
    """
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize(sentence) for sentence in line.split('\t')] for line in lines]
    # Create a new vocabulary
    vocabulary = Vocabulary(corpus_name)
    return vocabulary, pairs

# Filter pairs with max_length
def filter_pair(pair, max_length):
    """
    Remove one pair including sentences above max_length
    :param pair: list, a sub list of pairs
    :param max_length: int
    :return: [list]pairs
    """
    return len(pair[0].split(" ")) < max_length and len(pair[1].split(' ')) < max_length

# Filter all paris
def filter_pairs(pairs, max_length):
    """
    Filter all pairs
    :param pairs: list
    :return: list filtered
    """
    return [pair for pair in pairs if filter_pair(pair, max_length)]

# Process vocabulary and pairs:
def load_data(corpus_name, formatted_file, max_length):
    """
    load vocabulary and pairs
    :param corpus_name: str
    :param formatted_file: file_path
    :param max_length: int
    :return: vocabulary, [list]pairs
    """
    vocabulary, pairs = get_vocabulary_pairs(formatted_file, corpus_name)
    pairs = filter_pairs(pairs, max_length)
    # Add words into vocabulary
    for pair in pairs:
        vocabulary.add_sentence(pair[0])
        vocabulary.add_sentence(pair[1])
    return vocabulary, pairs

# Trim words with low frequency
def trim_words(vocabulary, pairs, min_count):
    """
    Remove pairs containing words, whose count is below min_count, and update vocabulary and pairs
    :param vocabulary:
    :param pairs:
    :param min_count:
    :return: pairs trimmed
    """
    vocabulary.trim(min_count)
    # pairs kept
    keep_pairs = []
    for pair in pairs:
        query = pair[0]
        answer = pair[1]
        keep_query = True
        keep_answer = True
        # if query contains trimmed words or not:
        for word in query.split(' '):
            if word not in vocabulary.word2index:
                keep_query = False
                # dont forget break
                break
        # it's turn to answer
        for word in answer.split(' '):
            if word not in vocabulary.word2index:
                keep_answer = False
                break
        if keep_query and keep_answer:
            keep_pairs.append(pair)
    return keep_pairs

##################################################################################################
# Prepare input data for our model
# Which means change each word in our sentence into index of words with vocabulary.word2index
''' 
    To accelerate training, torch uses 'batch' to handle data.
    However, sentences are not in the same length, it is necessary to pad them.
    At last, we could get input data in the size of (batch, max_length)
    
    But in order to make use of our GPU, we need to transpose input tensor 
    into the size of (max_length, batch), since data of each sentence is continuous in memory.
'''

# change word into index
def word2index(vocabulary, sentence):
    """
    Replace words in sentence with their indecies and add EOS_token
    :param vocabulary:
    :param sentence:
    :return: [list]sentence
    """
    return [vocabulary.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# Padding a sentence with PAD_token:0
def padding(l, fillvalue=PAD_token):
    """
    Padding a set of sentences with the max_length of them
    :param list: a list of sentences
    :param fillvalue: PAD_token:0
    :return: [list]: sentences in the same length
    """
    # use itertools.zip_longest
    # *list will unpack the list into a list of elements
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

# Get the BiMatrix of padded list for mask
def biMatrix(list, value=PAD_token):
    """
    Get the BiMatrix of padded list, which is useful to define maskNLLLoss
    :param list: a batch of sentences
    :param value: PAD_token:0
    :return: [matrix] biMatrix
    """
    m = []
    for i, seq in enumerate(list):
        # Add rows for each sentence
        m.append([])
        # Add elements for each row
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Process every query in pairs
def process_query(list, vocabulary):
    """
    Process query by: 1.word2index 2.get lengths 3.padding 4.change into Longtensor
    :param list: a batch of sentences
    :param vocabulary:
    :return: [Longtensor]:pad_query  [list]:sentence_length
    """
    # word->index
    index_batch = [word2index(vocabulary, sentence) for sentence in list]
    # get the length of each sentence, which is used for packing
    sentence_length = torch.tensor([len(sentence) for sentence in index_batch])
    # Padding
    pad_list = padding(index_batch)
    # Change into tensor
    pad_query = torch.LongTensor(pad_list)
    return pad_query, sentence_length

# Process every answer in pairs
def process_answer(list, vocabulary):
    """
    Besides processing above, we need to get the max_answer_length and masknon-singleton dimension 0 tensor for maskLoss
    :param list: a batch of answers
    :param vocabulary:
    :return: [LongTensor]:pad_answer, [LongTensor]:mask, [int]:max_answer_length
    """
    # word->index
    index_batch = [word2index(vocabulary, sentence) for sentence in list]
    # get the max_length in answers
    max_answer_length = max([len(sentence) for sentence in index_batch])
    # Padding
    pad_list = padding(index_batch)
    # Get mask
    mask = biMatrix(pad_list)
    # change into tensor
    mask = torch.ByteTensor(mask)
    # get tensor of answer
    pad_answer = torch.LongTensor(pad_list)
    return pad_answer, mask, max_answer_length

# Call funcs above to handle a batch of pairs
def process_pair(vocabulary, pairs_batch):
    # Sort by the length of each sentence
    # lamda function
    pairs_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    # init the query and answer batch
    query_batch, answer_batch = [], []
    for pair in pairs_batch:
        query_batch.append(pair[0])
        answer_batch.append(pair[1])
    query, lengths = process_query(query_batch, vocabulary)
    answer, mask, max_answer_length = process_answer(answer_batch, vocabulary)
    return query, lengths, answer, mask, max_answer_length
