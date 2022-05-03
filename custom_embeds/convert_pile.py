import os, sys, json

filepath = sys.argv[1]
idx = filepath.split('/')[-1].split('.')[0]
os.mkdir(idx)

split_num = 0
fout = open(idx + "/" + str(split_num) + ".jsonl", 'w+')

filepath = '00.jsonl'

fp = open(filepath)

lines = []
line = fp.readline()
i = 1

while len(line) > 0:
    if i % 100000 == 0:
        print(i)
        os.system('ps -a -o pid,%mem,cmd')
    if i > 10000000:
        break
    i += 1
    lines.append(json.loads(line))
    line = fp.readline()

