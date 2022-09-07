import sys
import numpy as np
from typing import Tuple
from keras.models import Model
from pre_processing import vocab_creator
import numpy as np

MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def model_load(filepath: str, HIDDEN_DIM=300, MAX_LEN=30):
    """
    This function loads the trained model located at the file path specified by the input variable filepath.
    """
    model = model.load(filepath)

    #the encoder model 
    encoder_model = Model(inputs = encoder_inputs, outputs = [encoder_outputs, state_h, state_c])

    #tensors to hold previous time steps
    decoder_state_input_h = Input(shape=(HIDDEN_DIM, ))
    decoder_state_input_c = Input(shape=(HIDDEN_DIM, ))
    decoder_hidden_state_input = Input(shape=(MAX_LEN, HIDDEN_DIM))

    #decoder sequence embeddings
    dec_emb2 = dec_emb_layer(decoder_inputs)

    #set the initial states for the next time step
    (decoder_outputs2, state_h2, state_c2) = decoder_lstm(dec_emb2,
        initial_state=[decoder_state_input_h, decoder_state_input_c])

    #generate a probability distribution over the target vocab to predict next characters
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    #the decoder model
    decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input,
                      decoder_state_input_h, decoder_state_input_c],
                      [decoder_outputs2] + [state_h2, state_c2])


    return model, encoder_model, decoder_model


def predict(input_seq: str, model, encoder_model, decoder_model):
    '''
    This is the function which predicts the expanded version of the input factored polynomial.

    This is where I run into issues: Having a difficult time figuring out how to write the code
    here (as was drafted up in seq2seq_play.ipynb) in an object-oriented/modular fashion. 
    
    '''
 
    #start by encoding the input sequence
    (e_out, e_h, e_c) = encoder_model.predict(input_seq)

    #generate an empty target sequence
    target_seq = np.zeros((1,1))

    #populate the first character w the BOS tag
    target_seq[0,0] = word2idx['<bos>']

    #adding characters until reach EOS tag or MAX_LEN
    stop_condition = False
    decoded_sequence = ''
    while not stop_condition:
        (output_tokens, h, c) = decoder_model.predict([target_seq]+[e_out, e_h, e_c])
        
        #sample a token
        sampled_token_index = np.argmax(output_tokens[0,-1,:])
        sampled_token = idx2word[sampled_token_index]

        if sampled_token_index!= '<eos>':
            decoded_sequence +=',' + sampled_token
            
        #the exit condition reached
        if sampled_token == '<eos>' or len(decoded_sequence.split(","))>=MAX_LEN-1:
            stop_condition = True

        #now update the target sequencee
        target_seq = np.zeros((1,1))
        target_seq[0,0] = sampled_token_index

        #update the internal states
        (e_h, e_c) = (h,c)

        #now go from a decoded sequence to an expansion and return
        prediction = ''
        for i in input_seq:
            if i != 0 and i != word2idx['<eos>'] and i != word2idx['<bos>']:
                prediction = prediction + idx2word[i]

    return prediction


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    factors, expansions = load_file(filepath)
    pred = [predict(f) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))


if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")