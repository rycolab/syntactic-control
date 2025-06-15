from collections import defaultdict


def decode_tokenizer_vocab(tokenizer):
    name = tokenizer.name_or_path.lower()
    if 'gpt2' in name:
        decoded = GPT2Mapping(tokenizer)
    elif 'llama-3' in name:
        decoded = LLaMaMapping(tokenizer)
    else:
        raise ValueError(f'We do not yet support tokenizer: {tokenizer.name_or_path}.')
    check_collisions(decoded)
    return decoded


def check_collisions(decoded):
    # check for vocabulary collisions
    tmp = defaultdict(list)
    for i, t in enumerate(decoded):
        tmp[t].append(i)
    for x in tmp:
        assert len(tmp[x]) == 1, f'surface form {x!r} maps to more than one token> {tmp[x]}'


def GPT2Mapping(tokenizer):
    # Adapted from
    # https://github.com/epfl-dlab/transformers-CFG/blob/main/transformers_cfg/tokenization/mapping.py
    decoded = []
    for token_id in range(len(tokenizer.get_vocab())):
        x = (
            tokenizer.convert_ids_to_tokens(token_id)
            .replace('Ġ', ' ')
            .replace('Ċ', '\n')
            .replace('ĉ', '\t')
            .replace('č', '\r')
            .replace('âĢĵ', '–')
        )
        x = '杜' if x == 'ľ' else x
        decoded.append(x)
    return decoded


def LLaMaMapping(tokenizer):
    # Adapted from
    # https://github.com/epfl-dlab/transformers-CFG/blob/main/transformers_cfg/tokenization/mapping.py
    decoded = []
    for token_id in range(len(tokenizer.get_vocab())):
        decoded.append(
            tokenizer.convert_ids_to_tokens(token_id)
            .replace('Ġ', ' ')
            .replace('Ċ', '\n')
            .replace('ĉ', '\t')
            .replace('č', '\r')
            .replace('âĢĵ', '–')
        )
    return decoded
