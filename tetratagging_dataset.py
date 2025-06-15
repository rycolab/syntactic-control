import os
import torch
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# The code is based on the Tetra-Tagging implementation from:
# https://github.com/nikitakit/tetra-tagging

# Please define your Hugging face token as environment variable.
HF_HUB_OFFLINE = 1
HF_TOKEN = os.environ["HF_TOKEN"]

READER = BracketParseCorpusReader(".", ["English.train", "English.dev", "English.test"])

# Mapping for normalizing special tokens.
TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‹": "'",
    "›": "'",
    "\u2013": "--",  
    "\u2014": "--",  
}

def ptb_unescape(sent):
    """
    Mapping special tokens to readable tokens.
    - sent: List of words in a sent.
    - return: List of cleaned words.
    """
    cleaned_words = []
    for word in sent:
        word = TOKEN_MAPPING.get(word, word)
        word = word.replace("\\/", "/").replace("\\*", "*")
        word = word.replace("-LSB-", "[").replace("-RSB-", "]")
        word = word.replace("-LRB-", "(").replace("-RRB-", ")")
        if word == "n't" and cleaned_words:
            cleaned_words[-1] = cleaned_words[-1] + "n"
            word = "'t"
        cleaned_words.append(word)
    return cleaned_words


class TetraTaggingDataset(torch.utils.data.Dataset):
    """
    # Dataset class for Tetra Tagging.
    """
    def __init__(self, split, tokenizer, tag_system, pad_to_len=None, max_train_len=60):
        assert split in ("English.train", "English.dev", "English.test")
        self.trees = READER.parsed_sents(split)
        self.tokenizer = tokenizer
        self.tag_system = tag_system
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_to_len = pad_to_len

        # Optionally filter long training examples.
        if split == "English.train" and max_train_len is not None:
            self.trees = [
                tree for tree in self.trees if len(tree.leaves()) <= max_train_len
            ]

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        tree = self.trees[index]
        words = ptb_unescape(tree.leaves())
        encoded = self.tokenizer.encode_plus(" ".join(words))
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)

        #  Map character positions to token positions.
        mapping = [self.tokenizer.encode((" " if i != 0 else "") + word, add_special_tokens=False) for i, word in enumerate(words)]
        word_end_positions = []
        temp = 0
        for word in mapping:
            i = temp + len(word)
            word_end_positions.append(i-1)
            temp = i

        tag_ids = self.tag_system.ids_from_tree(tree)
        tag_ids = [tag_id + 1 for tag_id in tag_ids] + [0]
        tag_ids = torch.tensor(tag_ids, dtype=torch.long)

        labels = torch.zeros_like(input_ids)
        leaf_labels = tag_ids[::2] - self.tag_system.internal_tag_vocab_size
        internal_labels = tag_ids[1::2]
        labels[word_end_positions] = (
            internal_labels * (self.tag_system.leaf_tag_vocab_size + 1) + leaf_labels
        )
        # Pad input and labels if needed.
        if self.pad_to_len is not None:
            pad_amount = self.pad_to_len - input_ids.shape[0]
            assert pad_amount >= 0
            if pad_amount != 0:
                input_ids = F.pad(input_ids, [0, pad_amount], value=self.pad_token_id)
                labels = F.pad(labels, [0, pad_amount], value=0)
        return {"input_ids": input_ids, "labels": labels}

    def collate(self, batch):
        # Collate function to prepare batches with padding.
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            [item["labels"] for item in batch], batch_first=True, padding_value=0.0
        )
        attention_mask = input_ids != self.pad_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
