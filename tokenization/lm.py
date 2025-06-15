"""
Language models go here
"""

import numpy as np
import torch
import transformers
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

from collections import OrderedDict
from arsenal.maths import sample_dict

from tokenization.util import  flatten, unflatten, Chart, decode_hf_tokenizer
from tokenization.backend import vocab_hack
import os

HF_TOKEN = os.getenv("HF_TOKEN")

def concat(xs, ys):
    return unflatten(flatten(xs) + flatten(ys))


class LM:
    r"""We say that $p\colon V^* \to [0,1]$ is a language model if $p$ is a probability
    distribution over strings from some alphabet $V$ of tokens.

    Every language model admits a left-to-right factorization:

    $$
    p(x_1 x_2 \cdots x_T) = p(x_1 \mid \varepsilon) p(x_2 \mid x_1) \cdots p(x_T \mid x_1 \cdots x_{T-1}) p(\mathrm{EOS} \mid x_1 \cdots x_T)
    $$

    Arguments:

      - `V`: a vocabulary of symbols

      - `eos`: a distinguished end of sequence symbol

      - `p_next(xs)`: $p(\cdot \mid x_1 \cdots x_T)$ is provided by subclasses.

    """

    def __init__(self, V, eos):
        self.eos = eos
        self.V = V
        self.concat = concat
        self.empty = ()

    def batch_p_next(self, contexts, keep_on_gpu=True):
        return torch.Tensor([self.p_next(context)._p for context in contexts])

    def logp_next(self, context):
        "Compute the log conditional distribution over the next token given the `prefix`."
        raise NotImplementedError()

    def logprefix(self, context):
        assert isinstance(context, tuple) and len(context) == 0 or len(context) == 2, context
        if len(context) == 0:
            return 0.0
        else:
            context, y = context
            return self.logprefix(context) + self.logp_next(context)[y]

    def logp(self, context):
        "Compute the log-probability of a complete string."
        return self.logprefix(context) + self.logp_next(context)[self.eos]

    def logp_next_seq(self, context, extension):
        """
        Compute `p(extension | context)` where `extension` is a sequence with |extension| > 1.
        """
        return self.logprefix(self.concat(context, extension)) - self.logprefix(context)

    def clear_cache(self):  
        pass

    def sample(
        self,
        ys=(),
        draw=sample_dict,
        prob=True,
        verbose=0,
        max_tokens=np.inf,
    ):
        "Draw a sample from this distribution."
        assert isinstance(ys, tuple) and len(ys) in {0, 2}, ys
        logP = 0
        t = 0
        while True:
            logp = self.logp_next(ys)
            p = logp.apply(np.exp)
            y = draw(p) if t < max_tokens else self.eos
            logP += logp[y]
            t += 1
            if verbose:
                if y == self.eos:
                    print()
                else:
                    print(y, end='')
            if y == self.eos:
                return [ys, logP] if prob else ys
            ys = (ys, y)

    def greedy(self, ys, **kwargs):
        return self.sample(ys=ys, draw=lambda p: p.materialize(top=1).argmax(), **kwargs)


class TokenizedLLM(LM):
    """
    This is a simple class which wraps a token LLM with a tokenizer.
    """

    def __init__(self, tokenizer, model, temperature=1.0, cache_size=64, byte_level=False):
        self.tokenizer = tokenizer
        self.temperature = temperature

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)    # send the model to gpu, if one is available
        model.eval()             # Set the model in evaluation mode; avoids gradient overhead

        self.byte_level = byte_level
        if byte_level:
            (_, self._encode, self._decode, _) = decode_hf_tokenizer(tokenizer)
            eos = self.tokenizer.eos_token.encode()
        else:
            self._decode = vocab_hack.decode_tokenizer_vocab(self.tokenizer)
            self._encode = {x: i for i, x in enumerate(self._decode)}
            eos = self.tokenizer.eos_token

        self._cache = OrderedDict()
        self._cache_size = cache_size
        self._eos_token_id = self._encode[eos]
        self.prompt = torch.LongTensor([[self.tokenizer.bos_token_id]]).to(self.device)

        super().__init__(V=set(self._decode), eos=eos)

    def encode_prompt(self, prompt):
        "Encode `prompt` as a tuple of tokens (each a string)."
        return unflatten(tuple(self._decode[i] for i in self.tokenizer.encode(prompt,add_special_tokens=False)))


    def clear_cache(self):
        self._cache.clear()

    def get_state(self, context):
        
        assert isinstance(context, tuple) and (len(context) == 0 or len(context) == 2), context

        value = self._cache.get(context, None)
        if value is not None:
            self._cache.move_to_end(context)   # Move the key to the end to show it was recently used
            return value

        if len(context) == 0:
            input_ids = self.prompt
            value = self.model(
                input_ids=input_ids,
                labels=input_ids,
                past_key_values=None,
                use_cache=True,
            )

        else:
            # Note: you should disable kv_cache for meta-llama/Meta-Llama-3-8B to achieve better results, if it is loaded with bfloat16
            (xs, x) = context
            x = self._encode[x]
            prev_state = self.get_state(xs)
            input_ids = torch.LongTensor([[x]]).to(self.device)
            value = self.model(
                input_ids=input_ids,
                labels=input_ids,
                past_key_values=prev_state.past_key_values,
                use_cache=True,
            )

        self._cache[context] = value
        if len(self._cache) > self._cache_size:
            # Pop the first item, as it is least recently used
            self._cache.popitem(last=False)

        return value

    def logp_next(self, context):
        with torch.no_grad():
            outputs = self.get_state(context)
            logits = outputs.logits.squeeze(0)
            if self.temperature!=1.0:
                logits = logits/self.temperature
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        _logp = lprobs[-1, :]
        if hasattr(_logp, 'cpu'):
            _logp = _logp.float().cpu().numpy()
        return LazyProb(_logp, self._encode, self._decode)
    
    def set_prompt(self, prompt):
        self.prompt = self.tokenizer(prompt, return_tensors="pt",add_special_tokens=False).input_ids.to(self.device)
        self.clear_cache()


class LazyProb:
    """
    This class is used to efficiently associate string with the indices of LLM's
    tokens distribution over next tokens.
    """

    def __init__(self, _p, encode, decode):
        self._p = _p
        self._encode = encode
        self._decode = decode

    def keys(self):
        return self._decode

    def values(self):
        return self._p

    def items(self):
        return zip(self._decode, self._p)

    def __getitem__(self, token):
        i = self._encode.get(token)
        return self._p[i] if i is not None else 0

    def materialize(self, top=None, sort=None):
        _p = self._p
        _decode = self._decode
        if top is None and sort is None:
            top_p = _p
        elif top is None:
            _p.argsort()
        else:
            top_p =  _p.argsort()[-int(top) :]
            #top_p = _p.argsort() if top is None else _p.argsort()[-int(top) :]
        pp = Chart(None)   # unsafe to guess a default value
        for i in reversed(top_p):
            pp[_decode[i]] = _p[i]
        return pp

    def top(self, K):
        return self.materialize(top=K)

    def __repr__(self):
        return repr(self.materialize())

    def apply(self, f):
        return LazyProb(
            _p = f(self._p),
            encode = self._encode,
            decode = self._decode,
        )

    def copy(self):
        return self.apply(lambda x: x.copy())


def load_model_by_name(model_name, **kwargs):
    """
    Load an LLM from 🤗 into a `TokenizedLLM`.
    """
    return TokenizedLLM(
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False),
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name),
        **kwargs,
    )

def load_model_by_name_llama(model_name, **kwargs):
    """
    Load an LLM from 🤗 into a `TokenizedLLM`.
    """
    return TokenizedLLM(
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(model_name, token=HF_TOKEN),
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN,torch_dtype=torch.bfloat16),
        **kwargs,
    )

