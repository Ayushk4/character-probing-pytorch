import json, os
import string
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

j = json.load(open("toks.json"))

mean = lambda x: sum(x)/len(x)
variance = lambda x: sum([(xx-mean(x))**2 for xx in x])/len(x)
std = lambda x: variance(x)**0.5
rnd4 = lambda x: round(x, 4)

keys = ['all_unique_toks', 'unique_toks_less_leven_dist_than_1_dist_wordnet_words',
         'words_exact_contain', 'words_exact_case_insensitive']

stats_2 = {key: [] for key in keys}
stats_4 = {key: [] for key in keys}
stats_6 = {key: [] for key in keys}
stats_8 = {key: [] for key in keys}

stats_600percent = {key: [] for key in keys}
stats_800percent = {key: [] for key in keys}
stats_900percent = {key: [] for key in keys}
stats_950percent = {key: [] for key in keys}
stats_980percent = {key: [] for key in keys}
stats_990percent = {key: [] for key in keys}
stats_995percent = {key: [] for key in keys}

for key in keys:
    for w in j:
        map_toks_to_freq = {tuple(tok[0]): tok[1] for tok in j[w][1]}
        coupled = [(tokenized, map_toks_to_freq[tuple(tokenized)])
                   for tokenized in j[w][0][key]]
        coupled.sort(key=lambda x: x[1], reverse=True)
        all_freq = sum([x[1] for x in coupled])

        stats_2[key].append(sum([x[1] for x in coupled[:2]])/all_freq)
        stats_4[key].append(sum([x[1] for x in coupled[:4]])/all_freq)
        stats_6[key].append(sum([x[1] for x in coupled[:6]])/all_freq)
        stats_8[key].append(sum([x[1] for x in coupled[:8]])/all_freq)

        for percent in [600, 800, 900, 950, 980, 990, 995]:
            idx = 0
            cnt = 0
            while cnt < percent*all_freq/1000:
                cnt += coupled[idx][1]
                idx += 1
            eval(f'stats_{percent}percent[key].append(idx)')

print("Percentage of occurrence cummulative across n-most-frequent tokenizations")
print()
for value in [2, 4, 6, 8]:
    print(value, 'most frequent tokenizations, mean and standard deviation')
    [print(k, rnd4(100*mean(v)), '%', rnd4(100*std(v))) for k,v in eval(f'stats_{value}').items()]
    print()

print()
print("Minimum number of tokenizations occupying p-percentage of instances")
print()
for percent in [600, 800, 900, 950, 980, 990, 995]:
    print(percent/10, '%, mean and standard deviation')
    [print(k, rnd4(mean(v)), rnd4(std(v)))
     for k, v in eval(f'stats_{percent}percent').items()]
    print()

stats_2_common = {key: [] for key in keys}
stats_4_common = {key: [] for key in keys}
stats_6_common = {key: [] for key in keys}

fn_2_common = [lambda x: x, lambda x: " " + x]
fn_4_common = fn_2_common + [lambda x: x[0].upper() + x[1:], lambda x: " " + x[0].upper() + x[1:]]
fn_6_common = fn_4_common + [lambda x: x.upper(), lambda x: " " + x.upper()]

keys = ['all_unique_toks', 'unique_toks_less_leven_dist_than_1_dist_wordnet_words',
         'words_exact_contain', 'words_exact_case_insensitive']

print()
print()

for key in keys:
    for w in j:
        map_toks_to_freq = {tuple(tok[0]): tok[1] for tok in j[w][1]}
        coupled_tok_dict = {tok.convert_tokens_to_string(tokenized): map_toks_to_freq[tuple(tokenized)]
                   for tokenized in j[w][0][key]}
        coupled_keys = set(coupled_tok_dict.keys())

        all_freq = sum([x[1] for x in coupled_tok_dict.items()])

        for cn in [2,4,6]:
            eval(f'stats_{cn}_common')[key].append(
                sum([coupled_tok_dict[fn(w)] for fn in eval(f'fn_{cn}_common')
                        if fn(w) in coupled_keys])/all_freq)
        if key == 'unique_toks_less_leven_dist_than_1_dist_wordnet_words':
            if stats_6_common[key][-1] < 0.8:
                print(key, w, stats_6_common[key][-1])


print()
print('Percentage of occurrence cummulative across most-frequent forms of a word')
for value in [2, 4, 6]:
    print(value, '; Mean and Std Dev.')
    [print(k, rnd4(100*mean(v)), '%', rnd4(100*std(v))) for k,v in eval(f'stats_{value}_common').items()]
    print()

print()
print('Cases where none of the 6 standard occurrences contribute majority')

for w in ['playstation', 'tripadvisor', 'trackback', 'javascript', 'ringtones', 'mathematic']:
    map_toks_to_freq = {tuple(tok[0]): tok[1] for tok in j[w][1]}
    coupled_tok_dict = {tuple(tokenized): map_toks_to_freq[tuple(tokenized)]
                   for tokenized in j[w][0]['unique_toks_less_leven_dist_than_1_dist_wordnet_words']}
    print(sorted(coupled_tok_dict.items(), key=lambda x: -x[1])[:5])

