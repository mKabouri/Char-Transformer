import os
import torch

import utils

text_data = utils.get_data("shakespeare.txt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NB_ITERATIONS=500
SEQ_LEN = 50
VOCAB_SIZE=utils.get_vocab_size(text_data)
EMBED_DIM=96
ATTENTION_DIM=EMBED_DIM
NB_HEADS=6

LEARNING_RATE=1e-3

itoc = utils.get_itoc(text_data)
ctoi = utils.get_ctoi(text_data)
training_data = utils.text_to_tensor(text_data, ctoi).unfold(0, SEQ_LEN, 1).to(device)

path_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../weights/")
