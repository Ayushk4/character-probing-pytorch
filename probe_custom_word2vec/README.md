# Experiment 3: Probing CBOW-Word2Vec Embeddings.

## Dependencies

- [nltk >= 3.6](https://www.nltk.org/install.html)
- [PyTorch >= 1.7.1](https://pytorch.org/get-started/previous-versions/)
- [Transformers > 4.10](https://huggingface.co/docs/transformers/installation)
- [Sentencepiece >= 0.1.91](https://pypi.org/project/sentencepiece/)
- [Gensim >=4.1.2](https://radimrehurek.com/gensim/)

NLTK requires additional data dependencies.
```
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Setting up

1. Train the Word2Vec embeddings following instructions from `../custom_embeds`.
2. Run the experiments for embedding using `python3 train.py --seed=[SEED] --batch_size=[BATCH_SIZE] --lr=[LEARNING_RATE] --n_epochs [NUM_EPOCHS] --wandb --model=<PATH_TO_WORD2VEC>`.
3. For control experiments include the additional flag `--control`.
4. You must include the `--model=<PATH_TO_WORD2VEC>` telling which Word2Vec model to use. Make sure the folder contains the .vec file as well as .npy and .symb.npy files.

## Results

Coming-Soon
