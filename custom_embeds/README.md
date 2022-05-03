# Train Custom CBOW Model

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

1. Download the [pile dataset's zst files](https://pile.eleuther.ai/). We used the files 00.jsonl.zst to 05.jsonl.zst. If the dataset isn't available at the provided link, ask for access to the backup server at [their discord](https://www.eleuther.ai/)
2. Crosscheck your shasums for these files. 
3. Run `decompress.py` from ../quantify_tokenization on each of the files individually: `python3 decompress.py <PATH/TO/xx.jsonl.zst>`.
4. Run `line_format.py` to bring the dataset into correct format -`python3 line_format.py <PATH_TO_PILE_PORTION>`
5. If you want to merge 'n' files for training Word2Vec, you can use `cat <PATH_TO_FILE_i> >> <PATH_TO_COMBINED_FILE>`, repeated once for each file.
6. Prepare Corpus for GPT-J's tokenization with `python3 gpt_j_corpus_prep.py <PATH_TO_CORPUS>`.
7. Prepare Corpus for controllable randomization with GPT-J using `python3 more_rand_gptj.py <PATH_TO_CORPUS> <RHO_Value>`
8. Finally train using `python3 train_word2vec.py <PATH_TO_CORPUS>`.
9. The output embeddings with be in a folder `vecs`.

