"""
FINAL SEQUENCE AFTER PREDICTION DON'T HAVE TO EXCEED SEQ_LEN
SEPECIFIED IN `./char_transformer/config.py` FILE.
"""

import torch
import char_transformer.config as config
import char_transformer.utils as utils
from char_transformer.main import CharTransformer

def predict_proba(model, input_context, n_predictions):
    model.eval()
    pred_characters = ""
    input_context = input_context.to(config.device)
    with torch.no_grad():
        for _ in range(n_predictions):
            logits_tensor = model(input_context)
            categories = torch.distributions.categorical.Categorical(logits=logits_tensor[-1])
            pred_index = torch.tensor([categories.sample()]).to(config.device)
            pred_characters += utils.tensor_to_text(pred_index, config.itoc)
            input_context = torch.cat((input_context, pred_index))
        return pred_characters
    
if __name__ == '__main__':
    device = config.device

    SEQ_LEN = config.SEQ_LEN 
    VOCAB_SIZE = config.VOCAB_SIZE
    EMBED_DIM = config.EMBED_DIM
    ATTENTION_DIM = config.ATTENTION_DIM
    NB_HEADS = config.NB_HEADS

    char_transformer = CharTransformer(VOCAB_SIZE,
                                       EMBED_DIM,
                                       SEQ_LEN,
                                       ATTENTION_DIM,
                                       NB_HEADS).to(device)

    # load weights
    char_transformer.load_state_dict(torch.load(config.path_weights))

    text_data = config.text_data

    context_text = text_data[1000:1020]
    tensor_context = utils.text_to_tensor(context_text, config.ctoi)
    print(f"{context_text + predict_proba(char_transformer, tensor_context, 10)} ")