from keras.layers import LSTM, Embedding, Input, Dense, TimeDistributed
from keras.models import Model

# def seq2seq_model(MAX_LEN, VOCAB_SIZE, HIDDEN_DIM=300, EMBEDDING_DIM=10):
    
#     #first the encoder layers
#     encoder_inputs = Input(shape = (MAX_LEN, ), 
#                            dtype = 'int32')
#     embed_layer = Embedding(input_dim = VOCAB_SIZE,
#                            output_dim = EMBEDDING_DIM,
#                            input_length = MAX_LEN)
#     encoder_embedding = embed_layer(encoder_inputs) #sofia what is embed_layer() doing?
#     encoder_LSTM = LSTM(HIDDEN_DIM,
#                        return_state = True)
#     encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    
#     #now the decoder layers
#     decoder_inputs = Input(shape = (MAX_LEN, ), 
#                           dtype = 'int32')
#     decoder_embedding = embed_layer(decoder_inputs)
#     decoder_LSTM = LSTM(HIDDEN_DIM, 
#                        return_state = True, 
#                        return_sequences = True)
#     #setting the initial state of the LSTM layer as the final state of the encoder LSTM layer
#     decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
    
#     #the final Dense layer ( applying a layer to every temporal input slice )
#     outputs = TimeDistributed(Dense(VOCAB_SIZE, activation = 'softmax'))(decoder_outputs)
#     model = Model([encoder_inputs, decoder_inputs], outputs)
    
#     return model


def seq2seq_model(MAX_LEN, VOCAB_SIZE, HIDDEN_DIM=300, EMBEDDING_DIM=10):
    #Encoder
    encoder_inputs = Input(shape = (MAX_LEN,))

    #Embedding layer
    enc_emb = Embedding(input_dim = VOCAB_SIZE,
                        output_dim = EMBEDDING_DIM,
                        input_length = MAX_LEN,
                    trainable = True)(encoder_inputs)

    #Encoder LSTM 1
    encoder_lstm1 = LSTM(HIDDEN_DIM, 
                        return_sequences = True,
                        return_state = True, 
                        dropout = 0.4, 
                        recurrent_dropout = 0.4)
    (encoder_output1, state_h1, state_c1) = encoder_lstm1(enc_emb)

    #ENcoder LSTM 2
    encoder_lstm2 = LSTM(HIDDEN_DIM, 
                        return_sequences = True,
                        return_state = True, 
                        dropout = 0.4, 
                        recurrent_dropout = 0.4)
    (encoder_output2, state_h2, state_c2) = encoder_lstm2(encoder_output1)

    #Encoder LSTM 3
    encoder_lstm3 = LSTM(HIDDEN_DIM, 
                        return_sequences = True,
                        return_state = True, 
                        dropout = 0.4, 
                        recurrent_dropout = 0.4)
    (encoder_outputs, state_h, state_c) = encoder_lstm3(encoder_output1)

    #now set up the decoder using the encoder_states as the initial decoder state
    decoder_inputs = Input(shape=(None, ))

    #Embedding layer
    dec_emb_layer = Embedding(input_dim = VOCAB_SIZE,
                        output_dim = EMBEDDING_DIM,
                        trainable = True)
    dec_emb = dec_emb_layer(decoder_inputs)

    #Decoder LSTM
    decoder_lstm = LSTM(HIDDEN_DIM, 
                    return_sequences = True, 
                    return_state = True, 
                    dropout = 0.4, 
                    recurrent_dropout = 0.2)
    (decoder_outputs, decoder_fwd_state, decoder_back_state) = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

    #Dense layer
    decoder_dense = TimeDistributed(Dense(VOCAB_SIZE, activation = 'softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
