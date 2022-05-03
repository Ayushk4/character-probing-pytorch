import os, sys
import random
import multiprocessing as mp
from datetime import datetime
from transformers import AutoTokenizer

gptj_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
only_gpt_keys = set(gptj_tokenizer.vocab.keys())
unique_keys = set([k[1:] if k[0] == 'Ġ' else k
                for k in gptj_tokenizer.vocab.keys()])
def assert_and_return(w):
    assert w in only_gpt_keys, w
    return w

unique_key_map = {k: k if k in only_gpt_keys else assert_and_return('Ġ' + k)
                 for k in unique_keys}

def variability_gptj_tokenizer(word, prob):
    if random.random() < prob:
        choices = [i for i in range(1, len(word))
        if word[:i] in unique_keys and word[i:] in unique_keys]
        if len(choices) > 0:
            index_choice = random.choice(choices)
            return [unique_key_map[word[:index_choice]],
                    unique_key_map[word[index_choice:]]]
    return gptj_tokenizer.tokenize(' ' + word)

def single_line(lines):
    prob = lines[1]
    return [' '.join([gptj_tokenizer.convert_tokens_to_string(toked).strip()
            for tok in line.split()
            for toked in variability_gptj_tokenizer(tok.strip(), prob)])
        for line in lines[0]]

def main():
    print("Reading the lines", datetime.now().strftime("%H:%M:%S"))
    lines = [lin.strip() for lin in open(sys.argv[1]).readlines()]

    probs = float(sys.argv[2])

    print("Starting the process", datetime.now().strftime("%H:%M:%S"))
    num_procs = min(mp.cpu_count(), 4)
    pool = mp.Pool(num_procs)
    print(num_procs)

    lines_divided_per = int(len(lines)/num_procs)
    lines_divide_at = [(i*lines_divided_per, (i+1)*lines_divided_per) if i < len(lines) -1 else (i*lines_divided_per, len(lines))
                    for i in range(num_procs)]
    
    results = pool.map(single_line, [(lines[i:j], probs) for i,j in lines_divide_at])
    print("Done", datetime.now().strftime("%H:%M:%S"))
    
    flatten = [res for reses in results for res in reses]
    print("Flattened", datetime.now().strftime("%H:%M:%S"))

    open('corpus.gpt' + sys.argv[2] + '.txt', 'w+').write('\n'.join(flatten))
    print("Written", datetime.now().strftime("%H:%M:%S"))

if __name__ == "__main__":
  main()

