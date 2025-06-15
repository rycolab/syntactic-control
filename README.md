# Syntactic Control of Language Models by Posterior Inference

## Setting Up The Environment

Set up a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Train Autoregressive Tetratagger

To train Tetra-Tagger, clone https://github.com/nikitakit/self-attentive-parser and download dataset:

```bash
./setup.sh
```

Run the training script for Tetratagger:

```bash
python train_tetra.py --model_string [MODEL] --output_dir [OUTPUT DIRECTORY] --epochs [EPOCHS] --batch_size [BATCH_SIZE] --enable_fp16 [FP16_ENABLE]
```

- `--model_string`: Name of the Hugging Face model to train (e.g., *gpt2-large*, *meta-llama/Meta-Llama-3-8B*).

- `--output_dir`: Path where the trained model will be saved.

- `--epochs`: Number of training epochs.

- `--batch_size`: Number of samples per training batch.

- `--enable_fp16`: Use 16-bit floating point precision for faster and more memory-efficient training (*True* or *False*).

## Evaluation

```bash
python evaluation.py --model_string [MODEL] --tetratagger_model_string [TETRATAGGER_MODEL] --conditioning_model_path [TETRATAGGER_PATH] --dataset_trees_file_path [TREES_FILE_PATH] --dataset_sents_file_path [SENTENCES_PATH] --proposal [PROPOSAL] --K [K_GRAM] --particles [PARTICLES] --threshold [THRESHOLD] --tetra [SHAPING] --runtimes [RUNTIMES] --shots [SHOTS]
```


- `--model_string`: Language model from Hugging Face used as prior (e.g., *gpt2-large*, *meta-llama/Meta-Llama-3-8B-Instruct*).
- `--tetratagger_model_string`: Base model for autoregressive Tetratagger (e.g., *gpt2-large*, *meta-llama/Meta-Llama-3-8B*).  
  *Note: For meta-llama/Meta-Llama-3-8B-Instruct, it's recommended to use meta-llama/Meta-Llama-3-8B as the Tetratagger model.*
- `--conditioning_model_path`: Path to the pre-trained autoregressive Tetratagger model.
- `--dataset_trees_file_path`: Path to the evaluation dataset containing syntactic trees.
- `--dataset_sents_file_path`: Path to the evaluation dataset containing sentences used for few-shot learning.
- `--proposal`: Proposal distribution to use. Options: *ngram* or *lm*.
- `--K`: K-gram size for the n-gram model.
- `--particles`: Number of particles used in the sampling process.
- `--threshold`: Threshold τ (tau) used in the sequential Monte Carlo (SMC) process.
    *Note: For sequential imporance sampling you can use τ=0*
- `--tetra`: Whether to use the Tetratagger in the shaping function (*0* = no, *1* = yes).
- `--runtimes`: Number of times to run experiments over the dataset.
- `--shots`: Number of few-shot examples for instruction-tuned models. Supported values: *0* or *5*.

*Note: To reproduce results for meta-llama/Meta-Llama-3-8B, if you load the model in bfloat16 you need to disable the KV cache.*