# import re,json,torch
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# from collections import Counter
# from torch.utils import data

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--file", type=str, required=True)
# params = parser.parse_args()

# class MyIter(torch.utils.data.IterableDataset):
#     def __iter__(self):
#         self.fp = open(params.file)
#         self.i = 0
#         return self

#     def __next__(self):
#         line = self.fp.readline()
#         self.i += 1
#         return line

# ds = MyIter()

# parallel_loader = data.DataLoader(ds, num_workers=1)
# cntr = Counter()
# i = 0
# for line in parallel_loader:
#     cntr.update(tokenizer.tokenize(json.loads(line[0])['text']))
#     if i % 1000 == 0:
#         print(i, max(cntr.items(), key=lambda x: x[1]))
#     i += 1

# json.dump(cntr, open("frequency"+params.file.split('.')[0]+".json","w+"))


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
            replace unicode with ascii
            regex python lib
            cntr.update(tokenizer.tokenize(json.loads(line)['text']))
            if i % 5000 == 0:
                print(idx, i, max(cntr.items(), key=lambda x: x[1]))
            i += 1
            line = fp.readline()

        json.dump(list(cntr.items()), open("divide/frequency" + str(idx) + ".json", "w+"))
        return i

parallel_loader = data.DataLoader(ParallelDataset(), num_workers=50, batch_size=100)
for x in parallel_loader:
    print(x)
