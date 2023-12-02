import os
import torch

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/")

def get_data(text_data_fname):
    return open(path + text_data_fname, 'r', encoding="utf8").read()

def get_vocabulary(text_data):
    return set(text_data)

def get_vocab_size(text_data):
    return len(get_vocabulary(text_data))

def get_ctoi(text_data):
    return {x: i for i, x in enumerate(get_vocabulary(text_data))}

def get_itoc(text_data):
    return {i: x for i, x in enumerate(get_vocabulary(text_data))}

def text_to_tensor(text_data, ctoi):
    return torch.tensor([ctoi[x] for x in text_data])

def tensor_to_text(tensor, itoc):
    return "".join([itoc[x.item()] for x in tensor])
