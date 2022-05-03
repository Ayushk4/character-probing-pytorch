# Quantifying Tokenization Variability

## Dependencies

- [zstandard: >= 0.16.0](https://pypi.org/project/zstandard/)
- [fuzzysearch >= 0.7.3](https://pypi.org/project/fuzzysearch/)
- [regex >= 2021.11.10](https://pypi.org/project/regex/)
- [nltk >= 3.6](https://www.nltk.org/install.html)
- [PyTorch >= 1.7.1](https://pytorch.org/get-started/previous-versions/)
- [Transformers > 4.10](https://huggingface.co/docs/transformers/installation)
- [Sentencepiece >= 0.1.91](https://pypi.org/project/sentencepiece/)
- [Python-Levenshtein >= 0.12.2](https://pypi.org/project/python-Levenshtein/)

NLTK requires additional data dependencies.
```
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Setting up

1. Download the [pile dataset's zst files](https://pile.eleuther.ai/). We used the files 00.jsonl.zst to 05.jsonl.zst. If the dataset isn't available at the provided link, ask for access to the backup server at [their discord](https://www.eleuther.ai/)
2. Crosscheck your shasums for these files. 
3. Run `decompress.py` on each of the files individually: `python3 decompress.py <PATH/TO/xx.jsonl.zst>`.
4. Run `pile_splitter.py` on the files individually to split each pice of pile into multiple pieces for parallel processing: `python3 pile_splitter.py <PATH/TO/xx.jsonl.zst>`
5. Run `parallel_filter_long.py` to get sentences where there are long words by regex: `python3 parallel_filter_long.py <PATH/TO/xx.jsonl.zst> <NUM_WORDS>`
6. Get Pseudo Possible Word Matches for words: `python3 parallel_find_pseudo_token_1levi.py`
7. Run `parallel_filter_qtfy.py` to extract the sentences where the words occur with their context: `python3 parallel_filter_long.py <PATH/TO/xx.jsonl.zst> <NUM_WORDS>`
8. Run `parallel_merge_qtfy_filter.py` to merge these based on words.
9. Run `parallel_qtfy_toks.py` so that for each word, the context is trimmed enough that makes difference to the tokenization but contains only the target word and quantifies it: `python3 parallel_qtfy_toks.py`.
10. Run `python3 final_qtfy_tokenization.py` to remove pseudo matches and print the statistics.
11. Run `python3 extra_final_statistics.py` for additional statistics.
