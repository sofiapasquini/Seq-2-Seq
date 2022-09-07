from tkinter import HIDDEN
from build_model import seq2seq_model
from pre_processing import *

#the name of the file holding the data set
file_name = "data.txt"

VOCAB_SIZE = 31 # 29+2 for BOS and EOS tags
MAX_LEN = 30 # with the tags

#read in the training data
data = load_data(file_name)
data.columns=["factored", "expanded"]

#split the sequences into lists of characters
data.factored = data.factored.apply(lambda x: seq_splitter(x))
data.expanded = data.expanded.apply(lambda x: seq_splitter(x))

#define the factors
factors = data.factored

#tag the expansions
expansions = tagger(data.expanded)

#define the vocabulary
# word2idx, idx2word = vocab_creator(seq_lists = factors + expansions, VOCAB_SIZE=VOCAB_SIZE)

#transform the sequences of characters to ID sequences
factor_sequences, expansion_sequences = text2seq(factors, expansions, VOCAB_SIZE=VOCAB_SIZE)

#pad the factor/expansion input sequences to the same length
factors_padded, expansions_padded = padding(factor_sequences, expansion_sequences, MAX_LEN=MAX_LEN)

#define the model architecture
model = seq2seq_model(MAX_LEN, VOCAB_SIZE)

# #write the model summary to the appropriate file
# with open('network.txt', 'w') as file:
#     model.summary(print_fn = lambda x: file.write(x + '\n'))

#compile the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#train the model
BATCH_SIZE = 1000
EPOCHS = 5

model.fit([factors_padded, expansions_padded[:,:-1]], expansions_padded.reshape(expansions_padded.shape[0],expansions_padded.shape[1],1)[:,1:], epochs = EPOCHS, batch_size = BATCH_SIZE)

#save the model to file
model.save('seq2seq_trained.h5')
