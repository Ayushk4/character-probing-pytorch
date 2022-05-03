import os, sys
import random
import multiprocessing as mp
from datetime import datetime
from transformers import AutoTokenizer

gptj_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

def single_line(lines):
    return [' '.join([gptj_tokenizer.convert_tokens_to_string(toked).strip()
            for tok in line.split()
            for toked in gptj_tokenizer.tokenize(' ' + tok)])
        for line in lines]

def main():
    print("Reading the lines", datetime.now().strftime("%H:%M:%S"))
    lines = [lin.strip() for lin in open(sys.argv[1]).readlines()]

    print("Starting the process", datetime.now().strftime("%H:%M:%S"))
    num_procs = min(mp.cpu_count(), 40)
    pool = mp.Pool(num_procs)
    print(num_procs)

    lines_divided_per = int(len(lines)/num_procs)
    lines_divide_at = [(i*lines_divided_per, (i+1)*lines_divided_per) if i < len(lines) -1 else (i*lines_divided_per, len(lines))
                    for i in range(num_procs)]
    
    results = pool.map(single_line, [lines[i:j] for i,j in lines_divide_at])
    print("Done", datetime.now().strftime("%H:%M:%S"))
    
    flatten = [res for reses in results for res in reses]
    print("Flattened", datetime.now().strftime("%H:%M:%S"))

    open('corpus.gpt.txt', 'w+').write('\n'.join(flatten))
    print("Written", datetime.now().strftime("%H:%M:%S"))

if __name__ == "__main__":
  main()

