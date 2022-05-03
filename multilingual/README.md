# Experiment 1: Non-English

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

1. Run the experiments for GPT-J using `python3 train.py --seed=[SEED] --batch_size=[BATCH_SIZE] --lr=[LEARNING_RATE] --n_epochs [NUM_EPOCHS] --wandb --language [LANGUAGE]`. where `[LANGUAGE]` is one of `['hindi', 'hiragana', 'arabic', 'russian', 'english', 'german']`.
2. For control experiments include the additional flag `--control`.

## Results

Mean 

| Language/Script       | Model | Control |
| --------------------- | ----- | ------- |
| Latin (English chars) | 80.95 | 39.13   |
| Devanagari            | 78.61 | 50.78   |
| Arabic                | 76.37 | 51.88   |
| Cyrillic              | 81.37 | 45.71   |
| --------------------- | ----- | ------- |

Variance 

| Language/Script       | Model | Control |
| --------------------- | ----- | ------- |
| Latin (English chars) | 3.28  | 7.21    |
| Devanagari            | 6.58  | 5.43    |
| Arabic                | 10.50 | 2.99    |
| Cyrillic              | 3.79  | 5.31    |
| --------------------- | ----- | ------- |
