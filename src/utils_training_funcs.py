# Utils for training and evalution

import torch
import torch.nn as nn

import random
import os

from src import model, utils_data_preprocessing as dp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAD_token = 0
SOS_token = 1
EOS_token = 2
################################################################
# One iteration
'''
    For one iteration, we need to implement two tircks
    1. teacher forcing: give the decoder the correct answer rather than
    one predicted by decoder in the time step t-1 as the input of t.
    2. Gradient clip: In order to avoid Explosion Gradient.
'''

# def train for one time step
def train(query_variable, lengths, answer_variable, mask, max_answer_length,
          encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
          batch_size, clip, max_length, teacher_forcing_ratio):
    # Clear gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device
    query_variable= query_variable.to(device)
    lengths = lengths.to(device)
    answer_variable = answer_variable.to(device)
    mask = mask.to(device)

    # Initialize values
    loss = 0
    print_losses = [] # for plot
    n_totals = 0

    # Encoder forward
    encoder_outputs, encoder_hidden = encoder(query_variable, lengths)

    # Create SOS_token:(1, batch) for Decoder forward
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    # fetch half of encoder_hidden as h_0 for decoder, since Encoder is bi-directional, Decoder is,
    # however, uni-direction.
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # If teacher forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Train for one time step
    if use_teacher_forcing:
        for t in range(max_answer_length):
            decoder_output, decoder_hidden = decoder(decoder_input, encoder_outputs, decoder_hidden)
            # Teacer Forcing
            decoder_input = answer_variable[t].view(1, -1)
            # Loss
            mask_loss, nTotal = model.maskNLLLoss(decoder_output, answer_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_answer_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Non Teacher Forcing
            # Use topk to get argmax
            _, topi = decoder_output.topl(dim=1)
            # Get each argmax in this batch
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Loss
            mask_loss, nTotal = model.maskNLLLoss(decoder_output, answer_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Backward
    loss.backward()

    # gradient clip
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # update
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

# Train
def train_iters(model_name, vocabulary, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                print_every, save_every, clip, corpus_name, loadFilename, teacher_forcing_ratio,
                max_length, checkpoint=0, ):

    # get n_iteration * batch
    training_batches = [dp.process_pair(vocabulary, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initialization
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Train:
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]

        query_variable, lengths, answer_variable, mask, max_answer_length = training_batch

        # Loss for one time step
        loss = train(query_variable, lengths, answer_variable, mask, max_answer_length,
                     encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size,
                     clip, max_length, teacher_forcing_ratio)
        print_loss += loss

        # rate of process
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("IterationL {}, Percent compelete: {:.1f}%; Average loss: {:.4f}"
                  .format(iteration, iteration/n_iteration*100, print_loss_avg))
            print_loss = 0

        # Save CheckPoint
        if (iteration % save_every) == 0:
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}-{}'
                                     .format(encoder_n_layers, decoder_n_layers, decoder.hidden_size))
            if not os.path.exists(directory):
                os.mkdir(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': vocabulary.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}-{}.tar'.format(iteration, 'checkpoint')))


# Greedy decoding
class GreedySearchDecoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Encoder forward
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Use the h_n as the h_0 of Decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Create init input of SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Tensor for decoding results
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Recurrent
        for _ in range(max_length):
            # Decoder forward for one time step
            decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_outputs, decoder_hidden)
            # Decoder_output: batch * vocabulary_size
            # Get the argmax word and score
            decoder_score, decoder_input = torch.max(decoder_output, dim=1)
            # Save
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_score), dim=0)
            # Use unsqueeze to add 'batch' dim
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores

# Test our chatbot
def evaluate(encoder, decoder, searcher, vocabulary, sentence, max_length):
    # input_batch -> id
    index_batch = [dp.word2index(vocabulary, sentence)]
    # Get lengths Tensor
    lengths = torch.tensor([len(index) for index in index_batch]).to(device)
    # T
    input_batch = torch.LongTensor(index_batch).transpose(0, 1).to(device)
    # Greedy search decoding
    tokens, scores = searcher(input_batch, lengths, max_length)
    # ID->word
    decoded_words = [vocabulary.index2word[token.item()] for token in tokens]
    return decoded_words

# Input
def evaluateInput(encoder, decoder, searcher, vocabulary, max_length):
    input_sentence = ''
    while(1):
        try:
            input_sentence = input('>')
            if input_sentence == 'quit' or input_sentence == 'q':
                break
            input_sentence = dp.normalize(input_sentence)
            output_words = evaluate(encoder, decoder, searcher, vocabulary, input_sentence, max_length)
            # Remove content after EOS_token
            words = []
            for word in output_words:
                if word == 'EOS':
                    break
                elif word != "PAD":
                    words.append(word)
            print("Bot: ", ' '.join(words))

        except KeyError:
            print("Error: Encountered unknown word.")
