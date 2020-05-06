'''
    main:
        1. Initialization
        2. Use utils and models from other files to create our chatbot
'''

import os
from src import utils_data_preprocessing as dp
from src import model
from src import utils_training_funcs as tf

import torch
import torch.nn as nn
from torch import optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import codecs
import csv
import random

# Define the path of our data
corpus_name = 'cornell movie-dialogs corpus'
corpus_path = os.path.join('../dataset/', corpus_name)

# show some lines of original files
# dp.printLines(os.path.join(corpus_path, 'movie_lines.txt'))

# Define the file to save the formatted pairs
formatted_file = os.path.join('../data/', 'formatted_lines.txt')

# Define the delimiter '\t' and decode it
delimiter = '\t'
    # unicode-escape: escape char '\t'
delimiter = str(codecs.decode(delimiter, 'unicode-escape'))

# Initialize dict:lines, list:conv and files
fields_lines = ['id_line', 'id_char', 'id_mv', 'char', 'text']
fields_conv = ['id_char1', 'id_char2', 'id_mv', 'ids_utterance']
lines = {}
conv = []
file_lines = os.path.join(corpus_path, 'movie_lines.txt')
file_conv = os.path.join(corpus_path, 'movie_conversations.txt')

# process each file step by step
lines = dp.loadLines(file_lines, fields_lines)
conv = dp.loadConv(file_conv, fields_conv, lines)

# Write pairs into a new .txt file by csv
with open(formatted_file, 'w', encoding='utf-8') as output:
    # use csv.writer
    writer = csv.writer(output, delimiter=delimiter, lineterminator='\n')
    # Get pairs from step 3
    for pair in dp.extractPairs(conv):
        writer.writerow(pair)

# dp.printLines(formatted_file)


# Define the max length of each sentence, which includes pre-defined tokens
MAX_LENGTH = 10

# Load vocabulary and pairs
vocabulary, pairs = dp.load_data(corpus_name, formatted_file, MAX_LENGTH)

# for pair in pairs[:5]:
#     print(pair)

# Define the min count
MIN_COUNT = 3

# Trimming
pairs = dp.trim_words(vocabulary, pairs, MIN_COUNT)

# # Create batches
# small_batch_size = 5
# # choose pairs randomly
# batches = dp.process_pair(vocabulary, [random.choice(pairs) for _ in range(small_batch_size)])
# query_variable, lengths, answer_variable, mask, max_answer_length = batches
#
# print("{}, {}, {}, {}, {}".format(query_variable, lengths, answer_variable, mask, max_answer_length) )

# max_length: The length of longest sentence in a batch NOT MAX_LENGTH
# The input of encoder is (max_length, batch)
# The input of decoder is (max_length, batch)


# Configuration of model
model_name = 'chatbot'
attn_model = 'general'
hidden_size = 512
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

loadFilename = None
checkpoint_iter = 4000
save_dir = os.path.join('../data/')

if loadFilename:
    checkpoint = torch.load(loadFilename)
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    vocabulary.__dict__ = checkpoint['voc_dict']

print("Building encoder and decoder ... ")
embedding = nn.Embedding(vocabulary.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
encoder = model.Encoder(hidden_size, embedding, encoder_n_layers, dropout=dropout)
decoder = model.Decoder(embedding, attn_model, hidden_size, vocabulary.num_words,dropout, decoder_n_layers)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


# Configuration of hyparameter and optimizer
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 1e-4
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 100
save_every = 1000

encoder.train()
decoder.train()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate*decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

tf.train_iters(model_name, vocabulary, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, corpus_name, loadFilename, teacher_forcing_ratio,
               MAX_LENGTH, checkpoint_iter)

encoder.eval()
decoder.eval()

searcher = tf.GreedySearchDecoder(encoder, decoder)
tf.evaluateInput(encoder, decoder, searcher, vocabulary, MAX_LENGTH)