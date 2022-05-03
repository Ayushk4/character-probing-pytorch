# Experiment 1: English Character Probing

## Dependencies

- [nltk >= 3.6](https://www.nltk.org/install.html)
- [PyTorch >= 1.7.1](https://pytorch.org/get-started/previous-versions/)
- [Transformers > 4.10](https://huggingface.co/docs/transformers/installation)
- [Sentencepiece >= 0.1.91](https://pypi.org/project/sentencepiece/)

NLTK requires additional data dependencies.
```
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Setting up

1. Download [GPT-J's Embeddings](https://github.com/Anonymous-ARR/Releases/releases/download/gptj/gpt-j-6B.Embedding.pth) into the current folder with name `gpt-j-6B.Embedding.pth`.
2. Run the experiments for GPT-J using `python3 train.py --seed=[SEED] --batch_size=[BATCH_SIZE] --lr=[LEARNING_RATE] --n_epochs [NUM_EPOCHS] --wandb`.
3. For control experiments include the additional flag `--control`.
4. For Non-GPT-J based experiments include the additional parameter `--model=<MODEL_CARD>`.

## Results

Mean 

| Model | Case-Oblivious | Case-Oblivious | Case-Sensitive | Case-Sensitive |
|                   | Lemma | Control | Lemma | Control |
| ----------------- | ----- | ------- | ----- | ------- |
| GPT-J-6B          | 93.70 | 48.36   | 94.35 | 52.76   |
| GPT-2-Base        | 84.25 | 52.31   | 84.69 | 51.05   |
| RoBerta-Base      | 86.41 | 47.33   | 83.87 | 49.00   |
| Bert-Base-Cased   | 78.50 | 47.08   | 78.47 | 45.35   |
| Bert-Base-Uncased | 77.48 | 49.37   | 77.48 | 49.37   |
| ----------------- | ----- | ------- | ----- | ------- |

Variance 

| Model | Case-Oblivious | Case-Oblivious | Case-Sensitive | Case-Sensitive |
|                   | Lemma | Control | Lemma | Control |
| ----------------- | ----- | ------- | ----- | ------- |
| GPT-J-6B          | 0.83  | 3.12    | 1.39  | 2.27    |
| GPT-2-Base        | 2.01  | 3.09    | 2.21  | 2.75    |
| RoBerta-Base      | 2.27  | 3.13    | 2.79  | 2.46    |
| Bert-Base-Cased   | 2.93  | 7.46    | 2.77  | 5.67    |
| Bert-Base-Uncased | 3.32  | 4.33    | 3.32  | 4.33    |
| ----------------- | ----- | ------- | ----- | ------- |
