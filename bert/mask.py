import sys
import torch

from transformers import BertTokenizer, BertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = BertTokenizer.from_pretrained(MODEL)
    encoded_input = tokenizer(text, return_tensors="pt")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, encoded_input)

    if mask_token_index is None:
        sys.exit(f"Input must inclde mask token {tokenizer.mask_token}.")

    # Model to process input
    model = BertForMaskedLM.from_pretrained(MODEL)
    output = model(**encoded_input, output_attentions=True)

    # Generate predictions
    mask_token_logits = output.logits[0][mask_token_index]
    
    top_tokens = torch.topk(mask_token_logits, K).indices

    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or `None` if not present in the `inputs`.
    """
    encoded_sequence = inputs['input_ids'][0]
    
    for i in range(len(encoded_sequence)):
        if encoded_sequence[i] == mask_token_id:
            return i
    return None



if __name__ == "__main__":
    main()
