import os, sys

filepath = sys.argv[1]
idx = filepath.split('/')[-1].split('.')[0]
os.mkdir(idx)

split_num = 0
fout = open(idx + "/" + str(split_num) + ".jsonl", 'w+')

fp = open(filepath)

line = fp.readline()
i = 1
while len(line) > 0:
    fout.write(line)
    if i % 50000 == 0:
        split_num += 1
        fout = open(idx + "/" + str(split_num) + ".jsonl", 'w+')
        print(i, split_num)
    i += 1
    line = fp.readline()
