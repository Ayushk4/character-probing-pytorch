import re,json,torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
from collections import Counter
from torch.utils import data

class ParallelDataset(data.Dataset):
    def __init__(self):
        self.path = "divide"

    def __len__(self):
        return 60

    def __getitem__(self, idx):
        cntr = Counter()
        fp = open("divide/" + str(idx) + ".txt")
        i = 0
        line = fp.readline()
        while len(line) > 0:
            cntr.update(tokenizer.tokenize(json.loads(line)['text']))
            if i % 5000 == 0:
                print(idx, i, max(cntr.items(), key=lambda x: x[1]))
            i += 1
            line = fp.readline()

        json.dump(list(cntr.items()), open("divide/frequency" + str(idx) + ".json", "w+"))
        return i

parallel_loader = data.DataLoader(ParallelDataset(), num_workers=10, batch_size=10)
for x in parallel_loader:
    print(x)

