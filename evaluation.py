import torch
import nltk
import numpy as np
import proposal
import tetratagger
from tqdm import tqdm
from nltk.util import ngrams
from collections import Counter
from argparse import ArgumentParser
import spacy
import benepar
import random
import re
from tokenization.lm import load_model_by_name, load_model_by_name_llama


def main(args):
    N = tetratagger.Potential()
    
    # nlp = spacy.blank("en") 
    spacy.cli.download("en_core_web_md")
    benepar.download("benepar_en3")

    nlp = spacy.load("en_core_web_md")
    if spacy.__version__.startswith("2"):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    trees = read_file(args.dataset_trees_file_path)
    trees=trees[:2]

    # sentences corresponding to trees dataset 
    n_s = read_file(args.dataset_sents_file_path)

    if args.model_string=='meta-llama/Meta-Llama-3-8B':
        lm = load_model_by_name_llama(args.model_string, byte_level=True, temperature=1)
    elif args.model_string=='meta-llama/Meta-Llama-3-8B-Instruct':
        lm = load_model_by_name_llama(args.model_string, byte_level=True, temperature=1)
    else:
        lm = load_model_by_name(args.model_string, byte_level=True)
    
    # Loads proposal and tetratagger.
    if args.proposal=='lm':
        shaping = tetratagger.Shaping(conditioning_model_path=args.conditioning_model_path, model_string=args.tetratagger_model_string)
        proposal_set = proposal.LMProposal(lm=lm,parser=N, Shaping=shaping)
    elif args.proposal == 'ngram':
        shaping = tetratagger.Shaping(conditioning_model_path=args.conditioning_model_path, model_string=args.tetratagger_model_string)
        proposal_set = proposal.NGramProposal.fit(lm=lm, lines=open('English.train'), K=2, parser=N, Shaping=shaping)
    else:
        raise ValueError(f"Unsupported proposal: {proposal}")

    data_gram = {}
    for tree in trees:
        data_gram[tree]={'sentence':[],'llh':[],'logp':[]}
    
    median_llh = []
    mean_logp = []
    mean_f1_scores = []
    mean_acc_scores = []
    exact_matches = []
    exact_structure = []
    
    all_sentences = []
    all_logps = []

    for i in range(args.runtimes):
        trees_predicted = []
        trees_true = []
        trees_true_nltk = []
        trees_predicted_nltk = []
        llh_list_nofail = []
        logp_list = []
        f1_list = []
        acc_list = []
        
        for tree in tqdm(trees, total=len(trees)):
            target_tree = nltk.Tree.fromstring("(TOP " + tree + ")")
            if 'instruct' in args.model_string.lower():
                if args.shots==0:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that generates a sentence. From a given constituency parse tree, output only one grammatical English sentence that matches the syntactic structure. Do not include any explanations, preambles, or quotation marks. Respond with the sentence only."},
                        {"role": "user", "content": f"Now generate a sentence for the following tree.\n\nParse tree: {tree}"},
                    ]
                else:
                    end = len(trees) - 1
                    n1 = random.randint(0, end)
                    n2 = random.randint(0, end)
                    n3 = random.randint(0, end)
                    n4 = random.randint(0, end)
                    n5 = random.randint(0, end)
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that generates a sentence. From a given constituency parse tree, output only one grammatical English sentence that matches the syntactic structure. Do not include any explanations, preambles, or quotation marks. Respond with the sentence only."},
                        {"role": "user", "content": f'Below are same examples for the task: \n\nParse tree: {trees[n1]}\n\nSentence: {n_s[n1]}\n\nParse tree: {trees[n2]}\n\nSentence: {n_s[n2]}\n\nParse tree: {trees[n3]}\n\nSentence: {n_s[n3]}\n\nParse tree: {trees[n4]}\n\nSentence: {n_s[n4]}\n\nParse tree: {trees[n5]}\n\nSentence: {n_s[n5]}\n\nNow generate a sentence for the following tree.\n\nParse tree: {tree}'},
                    ]
                prompt = lm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)+ 'Sentence: '
                lm.set_prompt(prompt)
                smc_posterior = proposal_set.smc(target_tree, n_particles=args.particles, threshold=args.threshold, tetra=args.tetra)
                lm.clear_cache()
            else:
                smc_posterior = proposal_set.smc(target_tree, n_particles=args.particles, threshold=args.threshold, tetra=args.tetra)

            sentence, llh, logp = smc_posterior.sample_posterior()
            data_gram[tree]['sentence'].append(sentence.lstrip())
            data_gram[tree]['llh'].append(llh)
            data_gram[tree]['logp'].append(logp)
            all_sentences.append(sentence)
            all_logps.append(logp)
            if llh!=500:
                # llh=500 represents the failure cases.
                llh_list_nofail.append(llh)
                logp_list.append(logp)
                tree_predicted = generate_tree(sentence.lstrip(), nlp)
                if tree_predicted!=0:
                    trees_predicted.append(tree_predicted)
                    trees_true.append(tree)
                    tree_predicted_nltk = nltk.Tree.fromstring("(TOP " + tree_predicted + ")")
                    scores = precision_recall_f1(target_tree, tree_predicted_nltk)
                    f1_list.append(scores[2])
                    acc_list.append(scores[3])

        exact_matches.append(correct_exact_match(trees_true, trees_predicted))
        exact_structure.append(correct_same_tree_structure(trees_true, trees_predicted))

        median_llh.append(np.log(np.median(np.exp(llh_list_nofail))))
        mean_logp.append(np.mean(np.array(logp_list)))
        mean_f1_scores.append(np.mean(np.array(f1_list)))
        mean_acc_scores.append(np.mean(np.array(acc_list)))
    
    bi_diversities = []
    uni_diversities = []
    tri_diversities = []
    for ind, tree in enumerate(trees):
        sentences = data_gram[tree]['sentence']
        if all(s.strip() == "" for s in sentences):
            continue
        llhs = data_gram[tree]['llh']
        sentences_new = []
        for i,l in enumerate(llhs):
            if l==500:
                continue
            else:
                sentences_new.append(sentences[i])
        if not sentences_new:
            continue
        sents = sentences_new
        tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sents]
        trigrams_list = [list(ngrams(sentence, 3)) for sentence in tokenized_sentences]
        bigrams_list = [list(ngrams(sentence, 2)) for sentence in tokenized_sentences]
        unigrams_list = [list(ngrams(sentence, 1)) for sentence in tokenized_sentences]

        all_trigrams = [trigram for sublist in trigrams_list for trigram in sublist]
        all_bigrams= [bigram for sublist in bigrams_list for bigram in sublist]
        all_unigrams = [unigram for sublist in unigrams_list for unigram in sublist]

        trigrams_counts = Counter(all_trigrams)
        bigrams_counts = Counter(all_bigrams)
        unigrams_counts = Counter(all_unigrams)

        distinct_bigrams = len(bigrams_counts.keys())
        distinct_unigrams = len(unigrams_counts.keys())
        distinct_trigrams = len(trigrams_counts.keys())

        length_of_set = len(all_unigrams)

        bi_diversity = distinct_bigrams/length_of_set
        uni_diversity = distinct_unigrams/length_of_set
        tri_diversity = distinct_trigrams/length_of_set
        bi_diversities.append(bi_diversity)
        tri_diversities.append(tri_diversity)
        uni_diversities.append(uni_diversity)

    print(f"Nikita's log-llh (median): {np.mean(np.array(median_llh))}, {np.std(np.array(median_llh))}")
    print(f"Prior (log): {np.mean(np.array(mean_logp))}, {np.std(np.array(mean_logp))}")
    print(f"Diversity uni: {np.mean(np.array(uni_diversities)), np.std(np.array(uni_diversities))}")
    print(f"Diversity bi: {np.mean(np.array(bi_diversities)), np.std(np.array(bi_diversities))}")
    print(f"Diversity tri: {np.mean(np.array(tri_diversities)), np.std(np.array(tri_diversities))}")
    print(f"F1-score:{np.mean(np.array(mean_f1_scores))}, {np.std(np.array(mean_f1_scores))}")
    print(f"Accuracy constituents:{np.mean(np.array(mean_acc_scores))}, {np.std(np.array(mean_acc_scores))}")
    print(f"Exact Matches: {np.mean(np.array(exact_matches))}, {np.std(np.array(exact_matches))}")
    print(f"Exact Structures: {np.mean(np.array(exact_structure))}, {np.std(np.array(exact_structure))}")


def ptb_unescape(sent):
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
            "\u2013": "--",  # en dash
            "\u2014": "--",  # em dash
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

def replace_words(input_string):
    tree = ""
    for s in input_string.split():
        if s[-1] == ")":
            w = s.replace(")", "")
            new = s.replace(w, "?")
            tree = tree + new + " "
        else:
            tree = tree + s + " "
    return tree[:-1]

def correct_exact_match(true_trees, pred_trees):
    count = 0
    for i in range(len(pred_trees)):
        if true_trees[i] != pred_trees[i]:
            count = count + 1
    return (len(true_trees) - count) / len(true_trees)

def correct_same_tree_structure(true_trees, pred_trees):
    count = 0
    for i in range(len(pred_trees)):
        g = pred_trees[i].split()
        o = true_trees[i].split()
        if len(o) != len(g):
            count = count + 1
            continue
            for j in range(len(g)):
                if g[j].startswith("("):
                    if o[j].startswith("("):
                        continue
                    else:
                        count = count + 1
                        break
                else:
                    c1 = g[j].count(")")
                    c2 = o[j].count(")")
                    if c1 != c2:
                        count = count + 1
                        break
    return (len(true_trees) - count) / len(true_trees)


def precision_recall_f1(gold_tree, predicted_tree):
    gold_constituents = extract_constituents(gold_tree)
    predicted_constituents = extract_constituents(predicted_tree)

    true_positives = gold_constituents & predicted_constituents
    precision = (
        len(true_positives) / len(predicted_constituents)
        if predicted_constituents
        else 0
    )
    recall = len(true_positives) / len(gold_constituents) if gold_constituents else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    accuracy = len(true_positives) / len(gold_constituents) if gold_constituents else 0

    return (precision, recall, f1, accuracy)


def extract_constituents(tree):
    """
    Extracts the set of constituents (brackets) from an NLTK Tree.
    Constituents are represented as tuples of (start_index, end_index, label).
    """

    def helper(node, start):
        if isinstance(node, str):  # Reached a leaf node
            return start + 1, set()

        end = start
        constituents = set()
        for child in node:
            child_end, child_constituents = helper(child, end)
            end = child_end
            constituents.update(child_constituents)

        # Add the current constituent (excluding the root node)
        if node.label() is not None and start != end:  # Avoid adding the root node
            constituents.add((start, end, node.label()))

        return end, constituents

    _, constituents = helper(tree, 0)
    return constituents

def generate_tree(sentence, nlp):
    try:
        doc = nlp(sentence)
        sent = list(doc.sents)[0]
        tree = replace_words(sent._.parse_string)
    except:
        tree = 0
    return tree

def generate_full_parse_trees_list(sentences, true_trees):
    nlp = spacy.load("en_core_web_md")
    if spacy.__version__.startswith("2"):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    trees = []
    new_trees = []
    for i, s in enumerate(sentences):
        try:
            doc = nlp(s)
            sent = list(doc.sents)[0]
            trees.append(replace_words(sent._.parse_string))
            new_trees.append(true_trees[i])
        except:
            x = 1
    return (trees, new_trees)

def find_sentence(text):
    # Using regular expression to find the first word after a space that is not "a" and does not start with an open parenthesis
    match = re.findall(r" (?![aA]\()(?![\(\[])([^ (\n]+)", text)
    if match:
        return match
    else:
        return None

def extract_sentences_from_trees(trees, init):
    n_s = []
    for i in range(len(trees)):
        for j in range(len(init)):
            s = ""
            n = replace_words(init[j])
            if trees[i] == n:
                s = ""
                for k in find_sentence(init[j]):
                    s = s + k.replace(")", "") + " "
                s = s[:-1]
                capitalized_sentence = s[0].upper() + s[1:]
                n_s.append(capitalized_sentence)
                break
    return n_s

def read_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        strings_list = [line.strip() for line in lines]
    return strings_list

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--model_string", 
        type=str, 
        default="gpt2",
        help="Language model used as prior."
    )  

    parser.add_argument(
        "--tetratagger_model_string", type=str, default="gpt2", 
        help="Base model for autoregressive tetratagger. Note that for meta-llama/Meta-Llama-3-8B-Instruct, we reccommend using meta-llama/Meta-Llama-3-8B for Tetratagger."
    ) 
    parser.add_argument(
        "--conditioning_model_path", 
        type=str, 
        default="model_tetra",
        help="Path of autoregressive tetratagger."
    ) 
    parser.add_argument(
        "--dataset_trees_file_path", 
        type=str, 
        default="dataset/eval_dataset_trees.txt", 
        help="Evaluation tree dataset."
    )  

    parser.add_argument(
        "--dataset_sents_file_path", 
        type=str, 
        default="dataset/eval_dataset_sentences.txt",
        help="Sentence dataset that is used for few shots."
    )

    parser.add_argument(
        "--proposal", 
        type=str, 
        default="lm", 
        help="The proposal distribution used. Either ngram or lm."
    ) 

    parser.add_argument(
        "--K",
        type=int,
        default=2,
        help="K-gram model",
    )

    parser.add_argument(
        "--particles",
        type=int,
        default=20,
        help="# particles",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Threshold tau for smc.",
    )

    parser.add_argument(
        "--tetra",
        type=int,
        default=0,
        help="Use or not tetratagger in shaping function.",
    )

    parser.add_argument(
        "--runtimes",
        type=int,
        default=1,
        help="#times to run experiments for whole dataset.",
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="# shots for instruction-tuned models. We support 0 and 5 shots here.",
    )

    args = parser.parse_args()

    main(args)
