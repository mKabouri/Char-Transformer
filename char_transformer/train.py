import os
import torch
import time

import utils
import config
from main import CharTransformer

def train(model, n_iters, device, optimizer, loss_fn, training_data):
    model.train()
    
    len_train = len(training_data)
    print(f"Training with {device}")
    n_seqs = 3000

    start_training_time = time.time()
    for iteration in range(n_iters):
        start_time = time.time()
        iteration_loss = 0
        perm = torch.randperm(len_train-1)
        idx = perm[:n_seqs]
        selected_seqs = training_data[idx]
        for i, input_seq in enumerate(selected_seqs):
            optimizer.zero_grad()
            
            output_logits = model(input_seq)

            target_seq = training_data[idx[i]+1]

            loss = loss_fn(output_logits.view(-1, output_logits.size(-1)),\
                           target_seq.view(-1))
            
            loss.backward()
            
            optimizer.step()
            
            iteration_loss += loss.item()
        end_time = time.time() - start_time
        if iteration%3 == 0:
            print(f"Iteration: {iteration}")
            print(f"Loss: {iteration_loss/n_seqs:.3}, iteration duration: {time.strftime('%M min %S s', time.gmtime(end_time)):.3}")
            print(f"-------------------------------------------")
            print()

    training_time = time.time() - start_training_time
    print(f'Training duration: {time.strftime("%H h %M min %S s", time.gmtime(training_time))}')

if __name__ == '__main__':
    text_data = config.text_data
    device = config.device
    
    SEQ_LEN = config.SEQ_LEN 
    VOCAB_SIZE = config.VOCAB_SIZE
    EMBED_DIM = config.EMBED_DIM
    ATTENTION_DIM = config.ATTENTION_DIM
    NB_HEADS = config.NB_HEADS

    training_data = config.training_data
    char_transformer = CharTransformer(VOCAB_SIZE,
                                       EMBED_DIM,
                                       SEQ_LEN,
                                       ATTENTION_DIM,
                                       NB_HEADS).to(device)

    LEARNING_RATE = config.LEARNING_RATE
    optimizer = torch.optim.Adam(char_transformer.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    NB_ITERATIONS = config.NB_ITERATIONS
    train(char_transformer,
          NB_ITERATIONS,
          device,
          optimizer,
          loss_fn,
          training_data)
    
    torch.save(char_transformer.state_dict(), "shakespeare_transformer.pt")
