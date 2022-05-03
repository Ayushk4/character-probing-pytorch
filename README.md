# Code

Codebase accompanying the submission `What do tokens know about their characters and how do they know it?`.

## Instructions:

We divide our codebase with the experiments:

### Section 3 and Appendix B

Follow the instructions in `experiment1/README.md` to replicate all our character probing experiments on English language.

Follow the instructions in `multilingual/README.md` to replicate all our character probing experiments on non-English language.

Follow the instructions in `expt1_substring/README.md` to replicate all our substring experiment.

### Section 4 and Appendix C

Follow the instructions in `sec_4.1_train_custom_models/README.md` to train our proposed syntax baselines for character information. You may also directly use our already-trained syntax model linked in that README.

Follow the instructions in `sec_4.1_using_spacy/README.md` to probe our SpaCy-syntax baseline for character information.

Follow the instructions in `sec_4.1_using_spacy/README.md` to probe our subword-syntax baselines for character information.

### Section 5 and Appendix D

Follow the instructions in `quantify_tokenization/README.md` to replicate our experiments to quantify the variability in subword tokenizers. Our code is also compatible with other sub-word tokenizers.

You may use `custom_embeds/README.md` to train custom word embeddings with controllable variability and prepare the corpus for it and you may then probe for character information following `probe_custom_word2vec/README.md`.





