import numpy as np
import pylab as pl
import nltk
import torch
import pandas as pd
import plotly.express as px
from arsenal import colors, iterview
from arsenal.maths import sample_dict, sample, softmax
from IPython.display import display
from tokenization.util import (
    Chart,
    flatten,
    unflatten,
    escape,
    logsumexp,
    logmeanexp,
)
from collections import defaultdict, Counter
from functools import lru_cache
from IPython.display import update_display, HTML
import spacy
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from tetratagger import Potential, Shaping

def append(xs, ys):
    """
    Recursively appends a nested tuple ys onto tuple xs.
    - xs: Nested tuple.
    - ys: Nested tuple.
    - return: Tuple.
    """
    if ys == ():
        return xs
    else:
        ys, y = ys
        return (append(xs, ys), y)

def copy_tree(tree):
    """
    Creates a new tree with the same labels.
    - tree: Input tree.
    - return: New tree.
    """
    if isinstance(tree, nltk.Tree):
        return nltk.Tree(tree.label(), [copy_tree(child) for child in tree])
    else:
        return tree


def replace_leaves(target_tree, value="?"):
    """
    Replaces leaves of a tree with "?".
    - target_tree: Input tree.
    - return: New tree.
    """
    t = copy_tree(target_tree)
    for leaf_pos in t.treepositions("leaves"):
        t[leaf_pos] = value
    return t

REPLACEMENTS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
}

class Proposal():
    """
    Superclass Proposal for SMC algorithm.
    """
    def __init__(self, lm, parser, Shaping, K=0):
        self.Shaping = Shaping
        self.parser = parser
        self.lm = lm
        self.nlp = spacy.blank("en")
        self.K = K
        self.positions=[]

    @classmethod
    def load_corpus(cls, lm, lines):
        """
        Loads and tokenizes a corpus of trees.
        - lm: Language model with encode_prompt() method defined.
        - lines: Dataset with trees.
        - return: (tree with replaced leaves, words of the tree-sentence, POS tags of the tree, list of tokenized words)
        """
        @lru_cache(None)
        def tokenize_word(word):
            return flatten(lm.encode_prompt(word))

        def make_tokenized_words(words):
            N = len(words)
            tws = []
            for n in range(N):
                w = words[n]
                w = REPLACEMENTS.get(w, w)
                if n == 0:
                    word = w
                # Special case for likely English contractions, possessives, and punctuation.
                elif w[0] in {"'", ".", "?"}:
                    word = w
                else:
                    word = " " + w
                tws.append(tokenize_word(word))
            return tws

        for line in iterview(list(lines)):
            tree = nltk.Tree.fromstring(line.strip())
            words, tags = zip(*tree.pos())
            yield (
                replace_leaves(tree),
                words,
                tags,
                make_tokenized_words(words),
            )
    
    def decoding_and_word_count(self, context):
        """
        Flattens a nested tuple of tokenized words into a sentence string and returns the number of words using spaCy. 
        - context: Nested tuple of tokenized words.
        - return: Word count using spaCy.
        """
        sentence = b"".join(flatten(context)).decode(errors="replace")
        nlp_decoding = self.nlp(sentence.lstrip())
        word_count = len(nlp_decoding)
        return word_count


    def logparse(self, s):
        """
        Computes the log-probability of a sentence for the given target tree.
        - s: state containing:
            - s.words: nested tuple of words of the sentence,
            - s.target_tree: target tree,
            - s.tags: POS-tags.
        - return: Log probability.
        """
        sentence = tuple(b"".join(word).decode() for word in flatten(s.words)) 

        try:
            return self.parser.logp(list(zip(sentence, s.tags)), s.target_tree)

        except (AssertionError, IndexError, ValueError):
            import sys
            import traceback

            (etype, evalue, tb) = sys.exc_info()
            error_message = "\n".join(traceback.format_exception(etype, evalue, tb))
            print(colors.light.red % error_message)

            self.failed.append((message, s))
            return -np.inf
    
    def word_count_list(self, sentence):
        """
        Returns word count and list of the words in the decoded sentence by spaCy. 
        - sentence: Input sentence.
        - return: (word count, list of decoded words of the sentence in spaCy)
        """
        # returns the word count of the decoding sequence
        nlp_decoding = self.nlp(sentence)
        word_count = len(nlp_decoding)
        return word_count, [token.text for token in nlp_decoding]

    def logp_potential(self, s, words):
        """
        Returns log-likelihood of a tree given a sentence under potential.
        - s: State containing information of the particle.
            - s.target_tree: Target tree.
            - s.tags: POS-tags.
        - words: words of the sentence.
        - return: Log-potential.
        """
        pos = [(word, s.tags[i]) for i, word in enumerate(words)]
        llh = self.parser.logp(pos,s.target_tree)
        return llh
    
    def logp_shaping(self,s, words):
        """
        Returns log-likelihood of a tree given a sentence under the shaping function.
        - s: State containing information of the particle.
            - s.target_tree: Target tree.
            - s.tags: POS-tags.
        - words: Words of the sentence.
        - return: Log-shaping.
        """
        pos = [(word, s.tags[i]) for i, word in enumerate(words)]
        llh = self.Shaping.logp(pos,s.target_tree)
        return llh
    
    def tetra_logprobs(self, sentence, s, word_count):
        """
        Returns log-probability of current leaf and internal tag given the last word of the sentence (shaping).
        - sentence: Generated sentence of the particle.
        - s: State containing information of the particle.
        - word_count: Count of words in the sentence.
        - return: Log-probability for the last word generated of the shaping function (for both leaf and internal node).
        """
        encoded_input = self.Shaping.tokenizer.encode_plus(sentence, add_special_tokens=False).input_ids
        encoded_input = torch.tensor([encoded_input]).to(self.Shaping.device)
        condition_logits = self.Shaping.tetratagger_logits(encoded_input) 
        weights = self.weights_last_token_tetra(condition_logits, s, word_count)
        return weights
    
    def weights_last_token_tetra(self, logits, s, word_count):
        """
        Computes log-probability of current leaf and internal tag given the last word of the sentence (shaping).
        - logits: Output logits of shaping function for the given tree and sentence.
        - s: State containing information of the particle.
            - s.tag_sequence: Tag sequence for the target tree.
            - s.N: Total Number of words.
        - word_count: Count of words in the sentence.
        - return: Log-probability for the last word generated of the shaping function (for both leaf and internal node).
        """
        tag_sequence_leaf_nodes = s.tag_sequence[::2]
        tag_sequence_internal_nodes = s.tag_sequence[1::2]
        if word_count<s.N:
            internal_tag = tag_sequence_internal_nodes[word_count-1]
            internal_logprobs = logits[0,-1,internal_tag].item() - torch.logsumexp(logits[0,-1,self.Shaping.internal_labels], dim=0).item()
        elif word_count==s.N:
            internal_logprobs = 0.0
        else:
            return -1e9
        leaf_tag = tag_sequence_leaf_nodes[word_count-1]
        leaf_logprobs = logits[0,-1,leaf_tag].item() - torch.logsumexp(logits[0,-1,self.Shaping.leaf_labels], dim=0).item()
        new_weights = leaf_logprobs+internal_logprobs
        return new_weights
    
    
    def initial_state(self, target_tree):
        """
        Initialize particle.
        - target tree: Target tree.
        - return: State containing information of the particle.
        """
        _, _tags = zip(*target_tree.pos())
        tag_sequence= self.parser.tag_system.ids_from_tree(target_tree)
        return State(
            context=(),
            words=(),
            n=0,
            N=len(_tags),
            tag_sequence = tag_sequence,
            tags=_tags + (None,) * self.K,
            target_tree=target_tree,
            weight=0,
            logp=0,
            logq=0,
            llh=None,
            parent=None,
            model=self,
        )

    def transition(self, s, word, choice):
        """
        Transition to next particle.
        - s: State containing information of the particle.
        - word: Word sampled.
        - choice: Dictionary {word sampled: log-probability under the proposal}
        """
        if word[0] == self.lm.eos:
            logp = self.lm.logp_next(s.context)[self.lm.eos]
            logq = np.log(1)

            # Finalized the particle's weight by including the parser's log
            # probability (potential)
            llh = self.logparse(s)
            weight_update = (logp - logq) + llh

            return State(
                context=s.context,
                words=s.words,
                tags=s.tags,
                target_tree=s.target_tree,
                tag_sequence=s.tag_sequence,
                n=s.n + 1,
                N=s.N,
                weight=s.weight + weight_update,
                logp=s.logp + logp,
                logq=s.logq + logq,
                llh=llh,
                parent=s,
                model=s.model,
            )

        else:
            word = tuple(word)
            # complete score of the sampled word under the LM
            logp = self.lm.logp_next_seq(
                s.context, unflatten(word)
            )
            # proposal's log-probability
            logq = choice[word]

            weight_update = logp - logq 

            return State(
                context=append(s.context, unflatten(word)),
                words=(s.words, word),
                tags=s.tags,
                tag_sequence=s.tag_sequence,
                target_tree=s.target_tree,
                n=s.n + 1,
                N=s.N,
                weight=s.weight + weight_update,
                logp=s.logp + logp,
                logq=s.logq + logq,
                llh=None,
                parent=s,
                model=s.model,
            )

    def transition_tetra(self, s, word, choice):
        """
        Transition to next particle when shaping function is used.
        - s: State containing information of the particle.
        - word: Word sampled.
        - choice: Dictionary {word sampled: log-probability under the proposal}
        """
        if word[0] == self.lm.eos:
            logp = self.lm.logp_next(s.context)[self.lm.eos]
            logq = np.log(1)

            sent = flatten(s.context)
            sentence=b"".join(sent).decode(errors="replace").lstrip()
            word_count, words = self.word_count_list(sentence)

            cleaned_words = self.parser.ptb_unescape(words)
            sentence_cleaned = " ".join(cleaned_words).lstrip()
            word_count, cleaned_words = self.word_count_list(sentence_cleaned)
            
            if len(cleaned_words)>s.N:
                llh_tagger = -np.inf 
                llh_tagger_shaping = 0.0
            else:
                llh_tagger = self.logp_potential(s, cleaned_words)
                llh_tagger_shaping = self.logp_shaping(s, cleaned_words)
              
            weight_update = (logp - logq) +(llh_tagger-llh_tagger_shaping) 

            return State(
                context=s.context,
                words=s.words,
                tags=s.tags,
                target_tree=s.target_tree,
                tag_sequence=s.tag_sequence,
                n=s.n + 1,
                N=s.N,
                weight=s.weight + weight_update,
                logp=s.logp + logp,
                logq=s.logq + logq,
                llh=llh_tagger,
                parent=s,
                model=s.model,
            )

        else:
            word = tuple(word)

            logp = self.lm.logp_next_seq(
                s.context, unflatten(word)
            )  # the complete score of the word
            logq = choice[word]  # guess at the score of the word
            
            new_sentence = append(s.context, unflatten(word))
            sentence=b"".join(flatten(new_sentence)).decode(errors="replace").lstrip()
            word_count, words = self.word_count_list(sentence)
            
            cleaned_words = self.parser.ptb_unescape(words)
            sentence_cleaned = " ".join(cleaned_words).lstrip()
            word_count, _ = self.word_count_list(sentence_cleaned)

            weights_tetra = self.tetra_logprobs(sentence_cleaned, s, word_count)
            weight_update = logp - logq + weights_tetra 

            if word_count==s.n:
                return State(
                    context=append(s.context, unflatten(word)),
                    words=(s.words, word),
                    tags=s.tags,
                    target_tree=s.target_tree,
                    tag_sequence=s.tag_sequence,
                    n=word_count,
                    N=s.N,
                    weight=s.weight + weight_update,
                    logp=s.logp + logp,
                    logq=s.logq + logq,
                    llh=500,
                    parent=s,
                    model=s.model,
                )

            return State(
                context=append(s.context, unflatten(word)),
                words=(s.words, word),
                tags=s.tags,
                target_tree=s.target_tree,
                tag_sequence=s.tag_sequence,
                n=word_count,
                N=s.N,
                weight=s.weight + weight_update,
                logp=s.logp + logp,
                logq=s.logq + logq,
                llh=None,
                parent=s,
                model=s.model,
            )


class NGramProposal(Proposal):
    """
    N-Gram Proposal for SMC algorithm.
    """
    def __init__(self, lm, C, K, parser, V, Shaping, heads_tokens_dict):
        super().__init__(lm, parser, Shaping, K)
        self.C = C # Conditional frequency counts for POS n-grams.
        self.V = V # Vocabulary mapping of tokenized words to indices.
        self.heads_tokens_dict = heads_tokens_dict # Maps first-token of tokenized word to list of matching word indices.
        self.positions = []

    @classmethod
    def fit(cls, lm, lines, K, parser, Shaping):
        """
        Fits an N-gram model from a parsed corpus.
        - lm: A language model (used for tokenizing words).
        - lines: Iterable of tree strings (Penn Treebank style).
        - K: The size of the n-gram window (e.g., 2 for bigram).
        - parser: Potential.
        - Shaping: Shaping function.
        - return: Instance of the class with learned counts and vocabulary mappings.
        """
        C = defaultdict(Counter) 
        tokenized_words_dict = {}
        heads_tokens_dict ={}
        corpus = cls.load_corpus(lm, lines)
        index_words=0
        max_words = 0
        for tree, words, pos_tags, tokenized_words in corpus:
            N = len(pos_tags) 
            pos_tags = (
                pos_tags + (None,) * K
            )  
            for n in range(N):
                pos_gram = pos_tags[n : n + K]
                C[pos_gram][tokenized_words[n]] += 1
                if len(tokenized_words[n])>max_words:
                    max_words =  len(tokenized_words[n])
                if tokenized_words[n] not in tokenized_words_dict:
                    tokenized_words_dict[tokenized_words[n]] = index_words
                    index_words+=1
                    if tokenized_words[n][0] not in heads_tokens_dict:
                        heads_tokens_dict[tokenized_words[n][0]] = [index_words-1]
                    else:
                        heads_tokens_dict[tokenized_words[n][0]].append(index_words-1)
        V = tokenized_words_dict
        return cls(lm=lm, C=C, K=K, parser=parser, V=V, Shaping=Shaping, heads_tokens_dict=heads_tokens_dict)
    
    def cheap_next_word(self, context, gram, _lambda=0.00001, flag_first=0):
        """
        Estimates the log-probability of the next word by combining:
            - the language model's estimate for the first token of the word
            - the count-based frequency of future tokens from the n-gram model.
        - context: Current decoding context as a nested tuple (prefix tokens so far).
        - gram: POS n-gram context used for conditional frequency lookup.
        - _lambda: Smoothing factor for interpolating with uniform distribution.
        - flag_first: If 1, initializes token positions mapping for fast lookup.
        - return: Numpy array with log-probabilities for each word in the vocabulary.
        """
        weights = Chart(-np.inf)
        c_ht = defaultdict(Counter)
        c_h = defaultdict(int)
        c = 0
        for word_tokens, count in self.C[gram].items():
            head = word_tokens[0]
            tail = word_tokens[1:]
            c_ht[head][tail] += count
            c_h[head] += count
            c += count

        logp_next = self.lm.logp_next(context)
        logps = logp_next.values()
        
        if flag_first==1:
            self.positions = np.zeros(len(self.V), dtype=int)
            index_logp = 0
            for key in logp_next.keys():
                if key in self.heads_tokens_dict:
                    self.positions[self.heads_tokens_dict[key]] = index_logp 
                index_logp+=1

        logps = np.array(list(logps))
        logps_mapped = np.array(logps[self.positions])        
        uniform = _lambda / len(self.V)
        smoothed_logps = np.full(len(self.V), uniform, dtype=np.float32)
        for word_tokens, count in self.C[gram].items():
            idx = self.V[word_tokens]
            smoothed_logps[idx]+=(1-_lambda)*count/c 
        smoothed_logps = np.log(smoothed_logps)  
        weights = smoothed_logps+logps_mapped

        return weights
    
    def logp_next(self, s, flag_first, _lambda):
        """
        Samples the next word.
        - s: State containing information of the particle.
            - s.context: Prefix.
            - s.tags: POS-tags of current and future words.
        - return: Sampled word and {sampled_word: log-probability}.
        """
        if s.n >= s.N:
            return (self.lm.eos,), {(self.lm.eos,):0}

        else:
            logp_next = self.cheap_next_word(s.context, s.tags[s.n : s.n + self.K],_lambda, flag_first)
            logZ = logsumexp(list(logp_next))
            logp_next_norm = logp_next - logZ
            probs = np.exp(logp_next_norm)
            word_tokens_list = list(self.V.keys())
            sampled_word = np.random.choice(np.array(word_tokens_list, dtype=object), p=probs)
            sampled_index = word_tokens_list.index(sampled_word)
            logp = logp_next_norm[sampled_index]
            return sampled_word, {sampled_word:logp}
    
    def smc(self, target_tree, n_particles, threshold, _lambda=0.00001, tetra=0):
        """
        SMC algorithm.
        - target_tree: Target tree.
        - n_particles: Number of particles.
        - threshold: Threshold for resampling.
        - _lambda: Lambda value for n-gram smoothing.
        - tetra: Enable shaping function.
        - return: ApproximatePosterior bject containing the final set of particles and statistics.
        """
        particles = []
        for _ in range(n_particles):
            particles.append(self.initial_state(target_tree))
        flag_first = 1

        while not all(
            s.is_complete() for s in particles
        ):  # still at some unfinished particles
            new_particles = []
            for s in particles:
                if s.is_complete():  # particle has completed; extension is a no-op
                    new_particles.append(s)
                else:
                    word, choice = self.logp_next(s, flag_first, _lambda)
                    flag_first = 0
                    if tetra:
                        new_particles.append(self.transition_tetra(s, word, choice))
                    else:
                        new_particles.append(self.transition(s, word, choice))
            ps = softmax([s.weight for s in new_particles])
            avg_weight = logmeanexp([s.weight for s in new_particles])
            print("next...", 1 / (ps @ ps), avg_weight)
            if threshold * n_particles * (ps @ ps) >= 1:
                print("resample")
                bootstrap = sample(ps, size=n_particles)
                particles = [new_particles[i] for i in bootstrap]
                for s in particles:
                    s.weight = avg_weight
            else:
                particles = new_particles

        return ApproximatePosterior(particles)


class LMProposal(Proposal):
    """
    LM Proposal for SMC algorithm.
    """
    def initial_state(self, target_tree):
        """
        Initialize particle.
        - target tree: Target tree.
        - return: StateLM containing information of the particle.
        """
        _, _tags = zip(*target_tree.pos())
        tag_sequence= self.parser.tag_system.ids_from_tree(target_tree)
        return StateLM(
            context=(),
            words=(),
            n=0,
            N=len(_tags),
            tag_sequence = tag_sequence,
            tags=_tags,
            target_tree=target_tree,
            weight=0,
            logp=0,
            logq=0,
            llh=None,
            parent=None,
            model=self,
            next_logp = None,
            next_context = None,
            next_token = None,
            count_tokens = 0
        )
        
    def transition_tetra(self, s, logp, new_context, next_context, next_logp, next_token, word, count_tokens):
        """
        Transition to next particle when shaping function is used.
        - s: State containing information of the particle.
        - logp: Language model log-probability of the tokens of the last word sampled.
        - new_context: New context of particle (prefix + new tokens sampled).
        - next_context: Next context of particle (new_context + next token sampled).
        - next_logp: Language model log-probability of the next token sampled.
        - next_token: Next token sampled.
        - word: Tokens of word sampled.
        - count tokens: Number of tokens already sampled (-1 refers to failure case).
        - return: State (particle).
        """    
        if word[0] == self.lm.eos:
            # last step, EOS token sampled
            logq = self.lm.logp_next(s.context)[self.lm.eos]
           
            sent = flatten(s.context)
            sentence=b"".join(sent).decode(errors="replace").lstrip()
            word_count, words = self.word_count_list(sentence)
            cleaned_words = self.parser.ptb_unescape(words)

            if count_tokens == -1:
                weight_update = -np.inf
                llh_tagger_potential = 500 # failure case
            else:
                if len(cleaned_words)!=s.N:
                    llh_tagger_potential = -np.inf 
                    llh_tagger_shaping = 0.0
                else:
                    llh_tagger_potential = self.logp_potential(s, cleaned_words)
                    llh_tagger_shaping = self.logp_shaping(s, cleaned_words)
                    if llh_tagger_shaping==-np.inf:
                        llh_tagger_shaping = 0
                        llh_tagger_potential = -np.inf
                weight_update = llh_tagger_potential-llh_tagger_shaping

            return StateLM(
                context=s.context,
                words=s.words,
                tags=s.tags,
                target_tree=s.target_tree,
                tag_sequence=s.tag_sequence,
                n=s.n + 1,
                N=s.N,
                weight=s.weight+weight_update,
                logp=s.logp+logq,
                logq=s.logq+logq,
                llh=llh_tagger_potential,
                parent=s,
                model=s.model,
                next_logp = None,
                next_context = next_context,
                next_token = next_token,
                count_tokens = count_tokens
            )

        else:
            word = tuple(word)

            logq = logp
            
            sentence=b"".join(flatten(new_context)).decode(errors="replace").lstrip()
            word_count, words = self.word_count_list(sentence)
            cleaned_words = self.parser.ptb_unescape(words)
            sentence_cleaned = " ".join(cleaned_words).lstrip()
            word_count, _ = self.word_count_list(sentence_cleaned)

            if word_count==0:
                weight_update = -np.inf
            else:
                weights_tetra = self.tetra_logprobs(sentence_cleaned,s, word_count)
                weight_update = weights_tetra 

            if count_tokens == 0:
                weight_update = -np.inf

            return StateLM(
                context=new_context,
                words=(s.words, word),
                tags=s.tags,
                tag_sequence=s.tag_sequence,
                target_tree=s.target_tree,
                n=len(cleaned_words),
                N=s.N,
                weight=s.weight + weight_update,
                logp=s.logp + logq,
                logq=s.logq + logq,
                llh=None,
                parent=s,
                model=s.model,
                next_logp = next_logp,
                next_context = next_context,
                next_token = next_token,
                count_tokens = count_tokens
            )

    def transition(self, s, logp, new_context, next_context, next_logp, next_token, word, count_tokens):
        """
        Transition to next particle when shaping function is not used.
        - s: State containing information of the particle.
        - logp: Language model log-probability of the tokens of the last word sampled.
        - new_context: New context of particle (prefix + new tokens sampled).
        - next_context: Next context of particle (new_context + next token sampled).
        - next_logp: Language model log-probability of the next token sampled.
        - next_token: Next token sampled.
        - word: Tokens of word sampled.
        - count tokens: Number of tokens already sampled (-1 refers to failure case).
        - return: State (particle).
        """ 
        if word[0] == self.lm.eos:
            logq = self.lm.logp_next(s.context)[self.lm.eos]
            sent = flatten(s.context)
            sentence=b"".join(sent).decode(errors="replace").lstrip()
            word_count, words = self.word_count_list(sentence)
            cleaned_words = self.parser.ptb_unescape(words)

            if count_tokens == -1:
                llh = 500 # failure cases
                weight_update = -np.inf
            else:
                if len(cleaned_words)!=s.N:
                    llh = -np.inf
                else:
                    llh = self.logp_potential(s, cleaned_words)

                weight_update = (logq - logq) + llh # proposal = prior

            return StateLM(
                context=s.context,
                words=s.words,
                tags=s.tags,
                target_tree=s.target_tree,
                tag_sequence=s.tag_sequence,
                n=s.n + 1,
                N=s.N,
                weight=s.weight+weight_update,
                logp=s.logp+logq,
                logq=s.logq+logq,
                llh=llh,
                parent=s,
                model=s.model,
                next_logp = None,
                next_context = next_context,
                next_token = next_token,
                count_tokens = count_tokens
            )

        else:
            word = tuple(word)
            
            logp = logp
            logq = logp

            weight_update = logp - logq 
            
            sent = flatten(new_context)
            sentence=b"".join(sent).decode(errors="replace").lstrip()
            word_count, words = self.word_count_list(sentence)
            cleaned_words = self.parser.ptb_unescape(words)
           
            if count_tokens == -1:
                # failure case
                weight_update = -np.inf

            return StateLM(
                context=new_context,
                words=(s.words, word),
                tags=s.tags,
                tag_sequence=s.tag_sequence,
                target_tree=s.target_tree,
                n=len(cleaned_words),
                N=s.N,
                weight=s.weight + weight_update,
                logp=s.logp + logp,
                logq=s.logq + logq,
                llh=None,
                parent=s,
                model=s.model,
                next_logp = next_logp,
                next_context = next_context,
                next_token = next_token,
                count_tokens = count_tokens
            )
    
    def smc(self, target_tree, n_particles, threshold, tetra=0):
        """
        SMC algorithm for LM as proposal.
        - target_tree: Target tree.
        - n_particles: Number of particles.
        - threshold: Threshold for resampling.
        - tetra: Enable shaping function.
        - return: ApproximatePosterior object containing the final set of particles and statistics.
        """
        particles = []
        for _ in range(n_particles):
            particles.append(self.initial_state(target_tree))
        init = 1
        while not all(
            s.is_complete() for s in particles
        ):  # still at some unfinished particles
            new_particles = []
            for s in particles:
                if s.is_complete():  # particle has completed; extension is a no-op
                    new_particles.append(s)
                elif init==1:
                    # sample first token
                    a, choices = self.logp_next(s.context, s.n, s.N)
                    word_a = tuple([a])
                    logp_a = choices[a]    
                    next_context, next_token, next_logp, logp, new_context, word, count_tokens  = self.generate_word(s, word_a, logp_a) # generate whole word
                    if tetra==1:
                        new_particles.append(self.transition_tetra(s, logp, new_context, next_context, next_logp, next_token,word, count_tokens))
                    else:
                        new_particles.append(self.transition(s, logp, new_context, next_context, next_logp, next_token, word, count_tokens))
                else:
                        new_particles.append(self.go_to_new_state(s, tetra))      
            init = 0
            ps = softmax([s.weight for s in new_particles])
            avg_weight = logmeanexp([s.weight for s in new_particles])
            print("next...", 1 / (ps @ ps), avg_weight)
            if threshold * n_particles * (ps @ ps) >= 1:
                print("resample")
                bootstrap = sample(ps, size=n_particles)
                particles = [new_particles[i] for i in bootstrap]
                for s in particles:
                    s.weight = avg_weight
            else:
                particles = new_particles

        return ApproximatePosterior(particles)

    def eos_case(self, state):
        """
        EOS case if sampled token is EOS or if the number of words have already generated (defined by the target tree)
        - state: State containing information of the particle.
        """
        logp_next = Chart(-np.inf)
        logp_next[self.lm.eos] = 0
        choices = Choice(state, logp_next)
        a = choices.sample()
        return a, choices

    def go_to_new_state(self, state, tetra):
        """
        Advances the given state (particle) to a new state by generating the next word or ending the sequence.
        - If the end of the sequence is reached or an EOS token is sampled, transition is triggered.
        - Otherwise, it attempts to generate the next word/token using the model.
        - If word generation fails (e.g. generating tokens without ending word), it defaults to an EOS transition.

        - state: State containing information of the particle.
        - tetra: Whether or not to use shaping function.
        - return: New state (particle).
        """
        if state.n>=state.N or state.next_token[0]==self.lm.eos or state.next_token[0]==b'<|eom_id|>':
            a, choices = self.eos_case(state)
            if tetra==1:
                return self.transition_tetra(state, choices[a], '', '', 0, tuple([a]), tuple([a]), state.count_tokens)
            else:
                return self.transition(state, choices[a], '', '', 0, tuple([a]), tuple([a]), state.count_tokens)
        else:
            next_context, next_token, next_logp, logp, new_context, word, count_tokens = self.generate_word(state, state.next_token, state.next_logp)
            
            if count_tokens == -1: 
                # failure cases 
                a, choices = self.eos_case(state)
                if tetra==1:
                    return self.transition_tetra(state, choices[a], '', '', 0, tuple([a]), tuple([a]), -1)
                else:
                    return self.transition(state, choices[a], '', '', 0, tuple([a]), tuple([a]), -1)
            else:
                if tetra==1:
                    return self.transition_tetra(state, logp, new_context, next_context, next_logp, next_token, word, count_tokens)
                else:
                    return self.transition(state, logp, new_context, next_context, next_logp, next_token, word, count_tokens)


    def generate_word(self, state, word_a, logp,early_stop=100, early_stop_ind=8):
        """
        Generates a word by sampling successive tokens until a complete word is formed or stopping criteria are met.
        It begins with an initial token word_a (this has been sampled from the previous step) 
        and repeatedly samples the next token until the number of 
        decoded words increases, indicating a complete word has been generated. It supports early stopping if the number 
        of total tokens (early_stop) or per-word tokens (early_stop_ind) exceeds a limit, or if an EOS 
        token is produced.
        - state: State containing information of the particle.
        - word_a:
        - logp:
        - early_stop: Early stop token generation if the total number of tokens generated for the string exceeds early_stop.
        - early_stop_ind: Early stop token generation if the number of generated tokens exceeds early_stop_ind.
        - return: (next context, next token generated, log-probability of next token, new context, new word generated (tokens), log-probability of tokens of current word)
        """
        words = []
        count_tokens = state.count_tokens
        context = state.context
        count_tokens+=1
        count_tokens_indiv=1
        words.append(word_a[0])

        # append sampled token from previous step to the current context
        new_context_a = append(context, unflatten(word_a))
        # count decoded words 
        word_count_a = self.decoding_and_word_count(new_context_a)
        # sample new token
        b, choices = self.logp_next(new_context_a, state.n, state.N)
        word_b = tuple([b])
       
        new_context_b = append(new_context_a, unflatten(word_b))

        # special cases of early stop or EOS token
        if count_tokens >= early_stop:
            return new_context_b, word_b, choices[b], logp, new_context_a, words, -1
        if count_tokens_indiv >= early_stop_ind:
            return new_context_b, word_b, choices[b], logp, new_context_a, words, -1
        if word_b[0]==self.lm.eos or word_b[0]==b'<|eom_id|>':
            return new_context_b, word_b, choices[b], logp, new_context_a, words, count_tokens
        
        word_count_b = self.decoding_and_word_count(new_context_b)

        # generate new tokens until the word count is changed, meaning we have a new word
        while word_count_b == word_count_a:
            count_tokens+=1
            count_tokens_indiv+=1
            words.append(word_b[0])
            logp += choices[b]
            new_context_a = new_context_b                
            if count_tokens_indiv >= early_stop_ind:
                return new_context_b, word_b, choices[b], logp, new_context_a, words, -1

            b, choices = self.logp_next(new_context_a, state.n, state.N) 
            word_b = tuple([b])
            new_context_b = append(new_context_a, unflatten(word_b))
            word_count_b = self.decoding_and_word_count(new_context_b)
        return new_context_b, word_b, choices[b], logp, new_context_a, words, count_tokens

    def logp_next(self, context, n, N):
        """
        - context: Prefix (nested tuple).
        - N: Total length of the sentence.
        - n: Current word-position.
        - return: 
        """
        if n == N:
            return (self.lm.eos,), {(self.lm.eos,):0}
        else:
            logp_next = self.token_generate(context)
            logZ = logsumexp(list(logp_next.values()))
            logp_next_norm = logp_next.values() - logZ
            probs = np.exp(logp_next_norm)
            keys = list(logp_next.keys())
            values = list(logp_next.values())
            sampled_index = np.random.choice(len(keys), p=probs)
            sampled_word = keys[sampled_index]
            logp = values[sampled_index]

            return sampled_word, {sampled_word:logp}

    def token_generate(self, context):
        """
        Generates next token log-probabilities.
        - context: Prefix (nested tuple).
        - return: Dictionary with tokens and log-probabilities from LM.
        """
        logp_next = self.lm.logp_next(context)
        return logp_next
    
class Choice(Chart):
    """
    Represents a chart storing scores for token choices at a decoding step.
    """
    def __init__(self, s, chart):
        super().__init__(chart.zero, chart)
        self.s = s

    def transition(self, a, **kwargs):
        """
        Performs a transition using the standard model's transition function.
        """
        return self.s.model.transition(self.s, a, self, **kwargs)

    def transition_tetra(self, a, **kwargs):
        """
        Performs a transition using the tetratagger shaping function.
        """
        return self.s.model.transition_tetra(self.s, a, self, **kwargs)

    def sample(self):
        """
        Samples a key from the chart.
        """
        return sample_dict(self.map_values(np.exp)) # sample keys based on their values

class State:
    """
    Represents a decoding state in the SMC (particle).
    """
    def __init__(
        self,
        weight,
        context,
        n,
        tags,
        target_tree,
        tag_sequence,
        N,
        words,
        logp,
        logq,
        llh,
        parent,
        model,
    ):
        self.target_tree = target_tree # target tree
        self.tags = tags # POS-tags of the tree
        self.n = n # current word-position
        self.N = N # total number of words of the tree
        self.tag_sequence = tag_sequence # tetratagger tag sequence for leaf and internal nodes
        self.context = context # context so far (partial sequence as nested tuple)
        self.words = words # tuple of generated words (nested)
        self.weight = weight # log weight of the particle
        self.logq = logq # log proposal probability
        self.logp = logp # log prior probability
        self.llh = llh # final log-llh of potential
        self.parent = parent # reference to the previous state (particle)
        self.model = model # reference to proposal class that generated this state (particle)

    def transition(self, a):
        assert not self.is_complete()
        return self.choices().transition(a)

    def choices(self):
        assert not self.is_complete()
        return self.model.logp_next(self)

    def is_complete(self):
        return self.llh is not None

    def __repr__(self):
        pp = ("❚").join(
            ("|").join(escape(x) for x in word) for word in flatten(self.words)
        )
        if len(pp) == 0:
            pp = "ε"
        return f"State({self.weight:.3f}, {pp})"

    def tree(self):
        ws = flatten(self.words)
        ws = ws + ("?",) * (len(ws) - self.N)
        t = copy_tree(self.target_tree)
        for leaf_pos, w in zip(t.treepositions("leaves"), ws):
            t[leaf_pos] = "|".join(map(escape, w))
        return t


class StateLM(State):
    def __init__(
        self,
        weight,
        context,
        n,
        tags,
        target_tree,
        tag_sequence,
        N,
        words,
        logp,
        logq,
        llh,
        parent,
        model,
        next_logp,
        next_token,
        next_context,
        count_tokens
    ):
        super().__init__(weight, context, n, tags, target_tree, tag_sequence, N, words, logp, logq, llh, parent, model)
        self.next_logp = next_logp # log prior probability of the next token sampled
        self.next_token = next_token # next token sampled
        self.next_context = next_context # next context 
        self.count_tokens = count_tokens # count of tokens generated (-1 for failure cases)


class ApproximatePosterior:
    """
    Represents a collection of particles to approximate posterior distribution.
    """
    def __init__(self, particles):
        self.particles = particles
        self._update_df()

    def __len__(self):
        return len(self.particles)

    def __getitem__(self, index):
        return self.particles[index]

    def append(self, particle):
        self.particles.append(particle)
        self._update_df()  

    def _update_df(self):
        if len(self.particles) == 0:
            return
        data = []
        for s in self.particles:
            data.append(
                dict(
                    ys=flatten(s.context),
                    words=flatten(s.words),
                    state=s,
                    sentence=b"".join(flatten(s.context)).decode(errors="replace"),
                    llh=s.llh,
                    logw=s.weight,
                    logprior=s.logp,
                    logq=s.logq,
                    lognumerator=s.logp
                    + s.llh,  # numerator of posterior = logproposal + logweight
                )
            )
        df = pd.DataFrame(data)
        logZ = logsumexp(df.logw)
        df["logposterior"] = df.logw - logZ
        self.df = df
        self.ess = np.exp(-logsumexp(2 * df.logposterior))
        self.logZ = logZ

    
    def sample_posterior(self):
        """
        Samples a sentence from particles. Returns (sentence, log-potential probability, log-prior probability)
        """
        if not self.particles:
            return None

        # Extract log-likelihoods, log-priors, and decoded sentence strings from all particles.
        llhs = [s.llh for s in self.particles]
        logps = [s.logp for s in self.particles]       
        sentences = [b"".join(flatten(s.context)).decode(errors="replace") for s in self.particles]
        logweights = [s.weight for s in self.particles]
        
        # Filter out particles with -inf weight.
        new_logweights = []
        new_sentences = []
        new_logps = []
        new_llhs = []
        for i,w in enumerate(logweights):
            if w==-np.inf or w==np.inf:
                continue
            else:
                new_logweights.append(w)
                new_sentences.append(sentences[i])
                new_logps.append(logps[i])
                new_llhs.append(llhs[i])

        if len(new_logweights)!=0:
            # If any valid particles remain, normalize weights and sample from the posterior.
            print(new_logweights)
            new_logweights=np.array(new_logweights)
            logZ = logsumexp(new_logweights)
            posterior = np.exp(new_logweights - logZ)
            posterior /= np.sum(posterior)
            sampled_idx = np.random.choice(len(new_sentences), p=posterior)
        else:
            # If all particles had -inf weight, fall back to uniform sampling over all.
            sampled_idx=np.random.choice(len(sentences))
            new_sentences = sentences
            new_logps = logps
            new_llhs = llhs

        return (new_sentences[sampled_idx], new_llhs[sampled_idx], new_logps[sampled_idx])

    