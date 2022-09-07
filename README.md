# Seq-2-Seq
I explore sequence-to-sequence NLP with a Keras Encoder-Decoder model.

I am exploring building a sequence-to-sequence encoder that will accept as input a factored polynomial and predict the expanded version based on input characters (not a mathematical scheme). I use the Keras framework and utilities to build my solution, along with other utilities form packages such as scikit-learn, numpy, pandas, etc; all requirements for running this code can be found in requirements.txt.

My pre-processing steps include splitting sequences to individual characters, tokenization,  identifier vectorization and implementing padding to target sequences.

The model architecture used is an Encoder-Decoder network with three LSTM layers in the Encoder and 1 LSTM layer in the Decoder. I use a sparse catagorical cross-entropy loss function and Adam optimizer. Network architecture is written to network.txt.

I am currently working on implementation of the inference model which successfully translates the decoder output to the expanded polynomial string, but have saved the trained model (model trained on data in file entitled "data.txt" which was populated with data from a coding challenge- please reach out should you wish to have access to this URL) as "seq2seq_2_trained.h5". This was the second model architecture explored, the first model architecture, trained on the same data, was saved as "seq2seq_trained.h5".

My exploration and drafting of this model can be viewed in the notebook seq2seq.ipynb, however, as I work through building the inference model, the goal is to translate this work into a more modular format in a series of Python files which have a single main() function used for testing/scoring the model on any test data.
