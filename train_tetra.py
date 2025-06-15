import os
import torch
import transformers
import bitsandbytes as bnb
from models_train import (
    ModelForTetraTagging,
    ModelForTetraTaggingLlama,
)
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from tetratagging_dataset import TetraTaggingDataset
import tetra_tag
from argparse import ArgumentParser
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# The code is based on the Tetra-Tagging implementation from:
# https://github.com/nikitakit/tetra-tagging


READER = BracketParseCorpusReader(".", ["English.train", "English.dev", "English.test"])
tag_system = tetra_tag.TetraTagSystem(trees=READER.parsed_sents("English.train"))

def main(args):
    
    # Please define your Hugging face token as environment variable.
    HF_HUB_OFFLINE = 1
    HF_TOKEN = os.environ["HF_TOKEN"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer based on model string.
    if args.model_string == "meta-llama/Meta-Llama-3-8B":
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, local_files_only=True
        )
    elif "gpt2" in args.model_string.lower():
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained(args.model_string)
    else:
        ValueError(f"Unsupported model string: {args.model_string}")
    
    # Set padding token if missing.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TetraTaggingDataset("English.train", tokenizer, tag_system)
    eval_dataset = TetraTaggingDataset("English.dev", tokenizer, tag_system, pad_to_len=256)

    # Load config with tag vocab mappings.
    if args.model_string == "meta-llama/Meta-Llama-3-8B":
        config = transformers.AutoConfig.from_pretrained(
            args.model_string,
            num_labels=len(tag_system.tag_vocab),
            id2label={i: label for i, label in enumerate(tag_system.tag_vocab)},
            label2id={label: i for i, label in enumerate(tag_system.tag_vocab)},
            task_specific_params={
                "num_leaf_labels": tag_system.leaf_tag_vocab_size,
                "num_internal_labels": tag_system.internal_tag_vocab_size,
            },
            output_hidden_states=True,
            token=HF_TOKEN,
            local_files_only=True,
        )
    else:
        config = transformers.AutoConfig.from_pretrained(
            args.model_string,
            num_labels=len(tag_system.tag_vocab),
            id2label={i: label for i, label in enumerate(tag_system.tag_vocab)},
            label2id={label: i for i, label in enumerate(tag_system.tag_vocab)},
            task_specific_params={
                "num_leaf_labels": tag_system.leaf_tag_vocab_size,
                "num_internal_labels": tag_system.internal_tag_vocab_size,
            },
            output_hidden_states=True,
        )

    # 4-bit quantization config.
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model using custom ModelForTetraTagging, ModelForTetraTaggingLlama classes and apply necessary setup.
    if "gpt2" in args.model_string.lower():
        model = ModelForTetraTagging.from_pretrained(args.model_string, config=config)
    elif args.model_string == "meta-llama/Meta-Llama-3-8B":
        model = ModelForTetraTaggingLlama.from_pretrained(
            args.model_string,
            config=config,
            quantization_config=bnb_config,
            token=HF_TOKEN,
        )
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        modules = find_all_linear_names(model)
        peft_config = create_peft_config(modules)
        model = get_peft_model(model, peft_config)
        print_trainable_parameters(model)
    else:
        ValueError(f"Unsupported model string: {args.model_string}")

    # Training configuration.
    training_args = transformers.TrainingArguments(
        output_dir="./results",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=160,
        weight_decay=0.01,
        logging_dir="./logs",
        save_steps=2149,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=4,
        fp16=args.enable_fp16,
    )

    # Trainer setup.
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_dataset.collate,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train model.
    trainer.train()
    print("Model trained successfully!")

    # Evaluate model.
    p = trainer.predict(eval_dataset)
    print("Metrics for evaluation dataset:")
    print(p.metrics)

    predicted_dev_trees = []
    for i in range(p.predictions.shape[0]):
        logits = p.predictions[i]
        is_word = p.label_ids[i] != 0
        pos = eval_dataset.trees[i].pos()
        tree = tag_system.tree_from_logits(logits, is_word, pos=pos)
        predicted_dev_trees.append(tree)
    
    # Save predictions of evaluation dataset to file.
    with open("./dev_predictions.txt", "w") as f:
        for tree in predicted_dev_trees:
            f.write(" ".join(str(tree).split()) + "\n")

    # Save final model.
    print("Saving last checkpoint of the model...")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    print("Model saved!")


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    - model: Model to compute parameters.
    - use_4bit: Whether to account for 4-bit quantization.
    - return: Dictionary with total, trainable, and trainable percentage of parameters.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def create_peft_config(modules):
    """
    Creates Parameter-Efficient Fine-Tuning config for your model.
    - param modules: List of module names to which LoRA should be applied.
    - return: A configured LoraConfig object for PEFT.
    """
    config = LoraConfig(
        r=16,  
        lora_alpha=64,  
        target_modules=modules,
        lora_dropout=0.1,  
        bias="none",
        task_type="TOKEN_CLS",  
    )

    return config


def find_all_linear_names(model):
    """
    Finds the names of all linear modules suitable for applying LoRA.
    - param model: The model to inspect for LoRA-compatible linear layers.
    - return: A list of module names to target for LoRA.
    """
    cls = (
        bnb.nn.Linear4bit
    )  
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def compute_metrics(p, num_leaf_labels=tag_system.leaf_tag_vocab_size):
    """
    Computes accuracies for both leaf and internal tagging decisions.
    - p: The output object from Trainer.predict(), containing predictions and label_ids.
    - num_leaf_labels: Number of leaf labels.
    - return: Dictionary containing accuracy for internal and leaf predictions.
    """
    leaf_predictions = p.predictions[..., -num_leaf_labels:]
    internal_predictions = p.predictions[..., :-num_leaf_labels]
    leaf_labels = p.label_ids % (num_leaf_labels + 1) - 1
    internal_labels = p.label_ids // (num_leaf_labels + 1) - 1

    leaf_predictions = leaf_predictions[leaf_labels != -1].argmax(-1)
    internal_predictions = internal_predictions[internal_labels != -1].argmax(-1)

    leaf_labels = leaf_labels[leaf_labels != -1]
    internal_labels = internal_labels[internal_labels != -1]

    return {
        "internal_accuracy": (internal_predictions == internal_labels).mean(),
        "leaf_accuracy": (leaf_predictions == leaf_labels).mean(),
    }


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--model_string",
        type=str,
        default="gpt2",
        help="Model string to be trained from Hugging Face.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="model_tetra",
        help="Output directory to save your model."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1,
        help="Number of epochs."
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size."
    )

    parser.add_argument(
        "--enable_fp16", 
        type=bool, 
        default=True,
        help="Enable fp16 for memory efficiency."
    )
    
    args = parser.parse_args()

    main(args)
