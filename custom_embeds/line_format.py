import re, os, json, sys
import multiprocessing as mp

filepath = sys.argv[1]
idx = filepath.split('/')[-1].split('.')[0]

splitted_folder = str(idx) + "_split"
try:
    os.mkdir(splitted_folder)
except:
    pass
fp = open(filepath)

line_format = ""
line = fp.readline()
i = 1
split_num = 1

do_wb = lambda x: re.sub(r'([^\w\s]+)', r' \1 ', x).strip()
do_punct = lambda x: re.sub(r'([^\w\.\?]+)', r' ', x).strip()
do_digit = lambda x: re.sub(r'([\d]+)', r' \1 ', x).strip()
do_white = lambda x: re.sub(r'[ \t]+', ' ', x).strip()
do_it = lambda x: do_white(do_digit(do_punct(do_wb(x))))

while len(line) > 0:
    if i % 10000 == 0 and i != 0:
        print(i, split_num)
        open(f'{splitted_folder}/{split_num}.00.txt',
            'w+').write(line_format)
        split_num += 1
        line_format = ""
    i += 1

    new_line_crunched = re.sub(r'\n\s*', '\n',
                            json.loads(line)['text']).strip()
    rm_punct = re.sub(r'([^\w\.\?]+)', r' ', new_line_crunched).strip()
    word_boundary_spaced = re.sub(r'([^\w\s]+)', r' \1 ', rm_punct).strip()
    digit_boundary_spaced = re.sub(r'([\d]+)', r' \1 ', word_boundary_spaced).strip()
    whitespace_crunched = re.sub(r'[ \t]+', ' ', digit_boundary_spaced).strip()

    line_format += whitespace_crunched + '\n'

    line = fp.readline()
