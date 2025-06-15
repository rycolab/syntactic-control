import os
import numpy as np
import torch
import transformers
from path import Path
from torch.nn.utils.rnn import pad_sequence
import spacy
from safetensors.torch import load_file
from tqdm import tqdm
from peft import PeftModel
import tetra_tag
from models_train import ModelForTetraTaggingLlama,ModelForTetraTagging

# The code is based on the Tetra-Tagging implementation from:
# https://github.com/nikitakit/tetra-tagging


class Tetratagger:
    """
    General Tetratagger class.
    """
    def __init__(self):
        self.config = transformers.AutoConfig.from_pretrained("kitaev/tetra-tag-en")
        self.tag_vocab = [self.config.id2label[i] for i in sorted(self.config.id2label.keys())]
        self.tag_system = tetra_tag.TetraTagSystem(tag_vocab=self.tag_vocab)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.leaf_labels = list(
            range(self.tag_system.internal_tag_vocab_size, len(self.tag_vocab))
        )
        self.internal_labels = list(
            range(
                self.tag_system.internal_tag_vocab_size,
            )
        )
        self.id2label = self.config.id2label
        self.model = 0
        self.tokenizer = 0 
    
    @staticmethod
    def ptb_unescape(sent):
        """
        Mapping special tokens to readable tokens.
        - sent: List of words in a sent.
        - return: List of cleaned words.
        """
        BERT_TOKEN_MAPPING = {
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

        cleaned_words = []
        for word in sent:
            word = BERT_TOKEN_MAPPING.get(word, word)
            word = word.replace("\\/", "/").replace("\\*", "*")
            # Mid-token punctuation occurs in biomedical text
            word = word.replace("-LSB-", "[").replace("-RSB-", "]")
            word = word.replace("-LRB-", "(").replace("-RRB-", ")")
            if word == "n't" and cleaned_words:
                cleaned_words[-1] = cleaned_words[-1] + "n"
                word = "'t"
            cleaned_words.append(word)
        return cleaned_words

    def parse_batch(self, batched_pos): 
        """
        Parses a batch of POS-tagged sentences into predicted trees.
        - batched_pos: Batch of listed tuples (words, POS tags).
        - return: Predicted trees (one for each input sentence).
        """
        (input_ids, label_mask, _logits) = self.logits(batched_pos)
        logits = _logits.numpy()

        predicted_trees = []
        for i in range(len(batched_pos)):
            tree = self.tag_system.tree_from_logits(
                logits[i], label_mask[i], pos=batched_pos[i]
            )
            show(
                self,
                input_ids[i][label_mask[i]],
                _logits[i][label_mask[i]],
                self.tag_system.ids_from_tree(tree),
            )

            predicted_trees.append(tree)

        return predicted_trees

    def parse(self, dataset_pos, batch_size=32):
        """
        Parses a list of POS-tagged sentences into trees using batching. Batches by length to speed up inference.
        - dataset_pos: Batch of listed tuples (words, POS tags).
        - return: Predicted trees.
        """
        sort_keys = sorted(range(len(dataset_pos)), key=lambda i: -len(dataset_pos[i]))
        sorted_dataset_pos = [dataset_pos[i] for i in sort_keys]

        predicted_trees = []
        for start_index in tqdm.notebook.trange(
            0, len(sorted_dataset_pos), batch_size, unit="batch"
        ):
            batch_pos = sorted_dataset_pos[start_index : start_index + batch_size]
            predicted_trees.extend(self.parse_batch(batch_pos))

        # Undo sort by length
        predicted_trees = [
            tree
            for (_, tree) in sorted(
                enumerate(predicted_trees), key=lambda x: sort_keys[x[0]]
            )
        ]
        return predicted_trees

    def logp(self, pos, tree): # returns logprobs of true tree
        """
        Returns log-probability of a sentence under a given ground-truth tree.
        - pos: List of (words, POS tags) of a sentence.
        - tree: Ground-truth nltk tree.
        - return: Float with the log-probability.
        """
        try:
            (input_ids, label_mask, _logits) = self.logits([pos])
        except: 
            return -np.inf
        logits = _logits.numpy()
        ys_ = input_ids[0][label_mask[0]]
        condition_logits = _logits[0][label_mask[0]]
        target_tree_encoding = self.tag_system.ids_from_tree(tree)

        tokenizer = self.tokenizer
        internal_labels = self.internal_labels
        leaf_labels = self.leaf_labels

        tag_sequence_leaf_nodes = target_tree_encoding[::2]
        tag_sequence_internal_nodes = target_tree_encoding[1::2]

        logprob = 0
        for t in range(ys_.size(0)):
            internal_logprobs = 0
            if t < len(tag_sequence_internal_nodes):  
                internal_tag = tag_sequence_internal_nodes[t]
                logZ_internal = torch.logsumexp(
                    condition_logits[t, internal_labels], dim=0
                ).item()
                internal_logprobs = (
                    condition_logits[t, internal_tag].item() - logZ_internal
                )

            leaf_tag = tag_sequence_leaf_nodes[t]
            logZ_leaf = torch.logsumexp(condition_logits[t, leaf_labels], dim=0).item()
            leaf_logprobs = condition_logits[t, leaf_tag].item() - logZ_leaf

            logprob += internal_logprobs + leaf_logprobs
        return logprob

    def logits(self, batched_pos):
        """
        Computes model logits for a batch of POS-tagged words.
        - batched_pos: Batch of listed tuples (words, POS tags).
        - return: (token_ids, mask of word_end_positions, logits).
        """
        tokenizer = self.tokenizer
        all_input_ids = []
        all_label_mask = []
        for pos in batched_pos:
            words = self.ptb_unescape([word for word, _ in pos])
            encoded = tokenizer.encode_plus(" ".join(words))
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            word_end_positions = [
                encoded.char_to_token(i)
                for i in np.cumsum([len(word) + 1 for word in words]) - 2
            ]
            label_mask = torch.zeros_like(input_ids, dtype=bool)
            label_mask[word_end_positions] = True
            all_input_ids.append(input_ids)
            all_label_mask.append(label_mask)
        input_ids = pad_sequence(
            all_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        all_label_mask = pad_sequence(
            all_label_mask, batch_first=True, padding_value=False
        ).numpy()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = input_ids != tokenizer.pad_token_id
            logits = self.model(input_ids, attention_mask=attention_mask).logits
        _logits = logits.cpu()

        return (all_input_ids, all_label_mask, _logits)

class Potential(Tetratagger):
    """
    Tetratagger used as Potential in SMC algorithm.
    This is the original Tetratagger trained. 
    """
    def __init__(self):
        super().__init__()
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            "kitaev/tetra-tag-en", config=self.config, ignore_mismatched_sizes=True
        )
        model.eval().to(self.device)  
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "kitaev/tetra-tag-en",
            use_fast=True,
            return_tensors="pt",
            padding=False,
        )
        self.model = model
        self.tokenizer = tokenizer    

class Shaping(Tetratagger):
    """
    Autoregressive tetratagger used as the shaping function in SMC algorithm.
    """
    def __init__(self, conditioning_model_path,model_string):
        super().__init__()
        
        HF_TOKEN = os.getenv("HF_TOKEN")
        nlp = spacy.blank("en")

        conditioning_model_path = Path(conditioning_model_path)
        assert conditioning_model_path.exists()  

        if 'llama-3' in model_string.lower():    
            tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(model_string, token=HF_TOKEN)
            adapter_path = conditioning_model_path
            config = transformers.AutoConfig.from_pretrained(
                model_string,
                num_labels=len(self.tag_system.tag_vocab),
                id2label={i: label for i, label in enumerate(self.tag_system.tag_vocab)},
                label2id={label: i for i, label in enumerate(self.tag_system.tag_vocab)},
                task_specific_params={
                    "num_leaf_labels": self.tag_system.leaf_tag_vocab_size,
                    "num_internal_labels": self.tag_system.internal_tag_vocab_size,
                },
                output_hidden_states=True,
                token=HF_TOKEN, local_files_only=True)
            base_model = ModelForTetraTaggingLlama.from_pretrained(
                model_string, config=config, torch_dtype=torch.bfloat16, token=HF_TOKEN)
            condition_model = PeftModel.from_pretrained(base_model, adapter_path).to(self.device)
            condition_model.eval()
        elif 'gpt2' in model_string:
            config = transformers.AutoConfig.from_pretrained(conditioning_model_path)
            condition_model = ModelForTetraTagging(config)
            weights_path = f"{conditioning_model_path}/model.safetensors"  
            state_dict = load_file(weights_path)
            condition_model.load_state_dict(state_dict)
            condition_model.to(self.device)
            condition_model.eval()
            tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_string)
        else:
            raise ValueError(f"Unsupported model string: {model_string}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.condition_model = condition_model
        self.tokenizer = tokenizer
        self.nlp = nlp
        self.model_string = model_string

    def logits(self, batched_pos): 
        """
        Computes model logits for a batch of POS-tagged words.
        - batched_pos: Batch of listed tuples (words, POS tags).
        - return: (token_ids, mask of word_end_positions, logits).
        """
        tokenizer = self.tokenizer
        all_input_ids = []
        all_label_mask = []
        for pos in batched_pos:
            words = [word for word, _ in pos]
            if 'llama-3' in model_string.lower():    
                encoded = tokenizer.encode_plus(" ".join(words), add_special_tokens=False)
            else:
                 encoded = tokenizer.encode_plus(" ".join(words))
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long) 
            if 'llama-3' in model_string.lower():    
                mapping = [tokenizer.encode((" " if i != 0 else "") + word, add_special_tokens=False) for i, word in enumerate(words)]
                word_end_positions = []
                temp = 0
                for word in mapping:
                    i = temp + len(word)-1
                    word_end_positions.append(i)
                    temp = i+1
            else:
                word_end_positions = [
                    encoded.char_to_token(i)
                    for i in np.cumsum([len(word) + 1 for word in words]) - 2]
            label_mask = torch.zeros_like(input_ids, dtype=bool)
            label_mask[word_end_positions] = True
            all_input_ids.append(input_ids)
            all_label_mask.append(label_mask)
        input_ids = pad_sequence(
            all_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        all_label_mask = pad_sequence(
            all_label_mask, batch_first=True, padding_value=False
        ).numpy()

        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = input_ids != tokenizer.pad_token_id
            logits = self.condition_model(input_ids, attention_mask=attention_mask).logits
        if 'llama-3' in model_string.lower():
            _logits = logits.to(dtype=torch.float32).cpu()
        else:
            _logits = logits.cpu()

        return (all_input_ids, all_label_mask, _logits)
    
    def tetratagger_logits(self, input_ids):
        """
        Returns full logits of Tetratagger of an already encoded input.
        - input_ids: Tensor of encoded input.
        - return: Logits.
        """
        with torch.no_grad():
            attention_mask = input_ids != self.tokenizer.pad_token_id            
            logits = self.condition_model(input_ids =input_ids, attention_mask=attention_mask).logits.to(dtype=torch.float32).to(self.device)
        return logits