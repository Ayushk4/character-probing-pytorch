import json
import multiprocessing as mp
from fuzzysearch import find_near_matches
from transformers import AutoTokenizer
from nltk.corpus import wordnet as wn
from math import log
from Levenshtein import distance as levenshtein_distance

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

longwords = [ n for n in wn.all_lemma_names() if len(n) > 7]

occurences = json.load(open('quantified.json'))

def is1diff(string1, string2):
    if abs(len(string1) - len(string2)) > 1: return False

    if len(string1) == len(string2):
        count_diffs = 0
        for a, b in zip(string1, string2):
            if a!=b:
                if count_diffs: return False
                count_diffs += 1
        if count_diffs == 0: return False
        return True

    if len(string1) < len(string2):
        string1, string2 = string2, string1

    it1 = iter(string1)
    it2 = iter(string2)
    count_diffs = 0
    c1 = next(it1, None)
    c2 = next(it2, None)
    while True:
        if c1 != c2:
            if count_diffs: return False
            count_diffs = 1
            c1 = next(it1)
        else:
            try:
                c1 = next(it1)
                c2 = next(it2)
            except StopIteration: return True

    return True

def single_word_qtfy(word):
    maps_tok_to_freq = [(tuple(tok.tokenize(single_string[0])), single_string[1])
                         for single_string in occurences[word]]

    unique_toks = set([tuple(tok.tokenize(single_string[0]))
                        for single_string in occurences[word]])

    #words_to_remove = list(set([single_string[0].strip().lower()
    #                        for single_string in occurences[word]
    #                            if single_string[0].lower().strip() != word and
    #                            len(wn.synsets(single_string[0].lower().strip())) != 0
    #                    ]))
    query_word_proced = word.strip().lower()
    words_to_remove = [x for x in longwords if is1diff(query_word_proced, x.lower().strip())]

    words_to_remove += list(set([single_string[0].strip().lower()
                            for single_string in occurences[word]
                                if single_string[0].lower().strip() != word and
                                len(wn.synsets(single_string[0].lower().strip())) != 0
                        ]))

    words_to_remove = list(set(words_to_remove))

    removed_words = set([tuple(tok.tokenize(single_string[0]))
                        for single_string in occurences[word]
                        if single_string[0].lower().strip() not in words_to_remove
                        ])


    removed_within_1_dist_less = set([tuple(tok.tokenize(single_string[0]))
                        for single_string in occurences[word]
                        if single_string[0].lower().strip() not in words_to_remove
                        and not any([levenshtein_distance(to_remove.lower().strip(),
                                                        single_string[0].lower().strip()) <
                                        levenshtein_distance(word.lower().strip(),
                                                        single_string[0].lower().strip())
                                    for to_remove in words_to_remove])
                        ])
    
    removed_within_1_dist_less_eq = set([tuple(tok.tokenize(single_string[0]))
                        for single_string in occurences[word]
                        if single_string[0].lower().strip() not in words_to_remove
                        and not any([levenshtein_distance(to_remove.lower().strip(),
                                                        single_string[0].lower().strip()) <=
                                        levenshtein_distance(word.lower().strip(),
                                                        single_string[0].lower().strip())
                                    for to_remove in words_to_remove])
                        ])

    words_removed_0_dist = [single_string[0]
                        for single_string in occurences[word]
                        if single_string[0].lower().strip() in words_to_remove
                        ]

    words_removed_1_dist_less = [single_string[0]
                        for single_string in occurences[word]
                        if single_string[0].lower().strip() in words_to_remove
                        or any([levenshtein_distance(to_remove.lower().strip(),
                                                        single_string[0].lower().strip()) <
                                        levenshtein_distance(word.lower().strip(),
                                                        single_string[0].lower().strip())
                                    for to_remove in words_to_remove])
                        ]

    words_removed_1_dist_less_eq = [single_string[0]
                        for single_string in occurences[word]
                        if single_string[0].lower().strip() in words_to_remove
                        or any([levenshtein_distance(to_remove.lower().strip(),
                                                        single_string[0].lower().strip()) <=
                                        levenshtein_distance(word.lower().strip(),
                                                        single_string[0].lower().strip())
                                    for to_remove in words_to_remove])
                        ]

    words_exact_contain = set([tuple(tok.tokenize(single_string[0]))
                        for single_string in occurences[word]
                        if word.strip().lower() in single_string[0].lower().strip()])

    capitalization = set([tuple(tok.tokenize(single_string[0]))
                        for single_string in occurences[word]
                        if word.strip().lower() == single_string[0].lower()])

    print(word, len(unique_toks), len(removed_words),
            len(removed_within_1_dist_less), len(removed_within_1_dist_less_eq),
            len(words_exact_contain), len(capitalization),
            words_to_remove,
            [x for x in words_removed_1_dist_less if x not in words_removed_0_dist],
            [x for x in words_removed_1_dist_less_eq if x not in words_removed_0_dist and x not in words_removed_1_dist_less],
            '\n')

    return [{'all_unique_toks': list(unique_toks),
            'words_exact_contain': list(words_exact_contain),
            'words_exact_case_insensitive': list(capitalization),
            '1_dist_words_in_wordnet': list(words_to_remove),
            'datapoint_less_eq_leven_dist_away_1_dist_words_in_wordnet': list(words_removed_1_dist_less),
            'datapoint_less_leven_dist_away_1_dist_words_in_wordnet': list(words_removed_1_dist_less_eq),
            'unique_toks_without_1_dist_wordnet_words': list(removed_words),
            'unique_toks_less_eq_leven_dist_than_1_dist_wordnet_words': list(removed_within_1_dist_less),
            'unique_toks_less_leven_dist_than_1_dist_wordnet_words': list(removed_within_1_dist_less_eq),
            }, maps_tok_to_freq]

def print_statishtics(number_list, word_list):
    scale_occurrence_frequency = lambda x: min(16, max(12, round(log(x))))
    word_occurrences_cnt = {k: sum([x[1] for x in occurences[k]]) for k in occurences}

    mean = lambda x : sum(x)/len(x)
    variance = lambda x: sum([(xx - mean(x))**2 for xx in x])/len(x)
    std = lambda x: variance(x) ** 0.5
    round4 = lambda x: round(x, 3)

    print("Max:", round4(max(number_list)))
    print("Min:", round4(min(number_list)))
    print("Mean:", round4(sum(number_list)/len(number_list)))
    # print("Variance:", round4(variance(number_list)))
    print("Std Dev.:", round4(std(number_list)))

    for grp_by_len in [1, 2, 3]:
        if grp_by_len==1:
            grouped = {x: ([], []) for x in set([len(w) for w in word_list])}
            for num, wrd in zip(number_list, word_list):
                grouped[len(wrd)][0].append(num)
                grouped[len(wrd)][1].append(wrd)
            print("Length Wise:")
        elif grp_by_len==2:
            grouped = {x: ([], []) for x in [12,13,14,15,16]}
            for num, wrd in zip(number_list, word_list):
                grouped[scale_occurrence_frequency(word_occurrences_cnt[wrd])][0].append(num)
                grouped[scale_occurrence_frequency(word_occurrences_cnt[wrd])][1].append(wrd)
            print("Occurrence Wise:")
        else:
            grouped = {}#x: ([], []) for x in set([max(13, min(8, len(w))) for w in word_list])}
            for num, wrd in zip(number_list, word_list):
                if 14 < (log(word_occurrences_cnt[wrd])) < 14.5:
                    if min(13, max(8, len(wrd))) in grouped:
                        grouped[min(13, max(8, len(wrd)))][0].append(num)
                        grouped[min(13, max(8, len(wrd)))][1].append(wrd)
                    else:
                        grouped[min(13, max(8, len(wrd)))] =[[num], [wrd]]

            print("Occurrence-Most Wise:")


        print("", "Num Examples")
        print("  ", end="")
        for w_len, grp in sorted(grouped.items(), key=lambda x: x[0]):
            print(f'{w_len}: {len(grp[1])}', end=", ")

        print()
        print("", "Example Words")
        print("  ", end="")
        for w_len, grp in sorted(grouped.items(), key=lambda x: x[0]):
            print(f'{w_len}:{grp[1][0]}', end=",")

        print()
        print("", "Mean")
        print("  ", end="")
        for w_len, grp in sorted(grouped.items(), key=lambda x: x[0]):
            print(f'{w_len}: {round4(mean(grp[0]))}', end=", ")

        print()
        print("", "Avg occurrences")
        print("  ", end="")
        for w_len, grp in sorted(grouped.items(), key=lambda x: x[0]):
            print(f'{w_len}: {round(mean([word_occurrences_cnt[w] for w in grp[1]]))}', end=", ")

        print()
        print("", "Std Dev.")
        print("  ", end="")
        for w_len, grp in sorted(grouped.items(), key=lambda x: x[0]):
            print(f'{w_len}: {round4(std(grp[0]))}', end=", ")

        print()
        print("", "Max")
        print("  ", end="")
        for w_len, grp in sorted(grouped.items(), key=lambda x: x[0]):
            print(f'{w_len}: {round4(max(grp[0]))}', end=", ")

        print()
        print("", "Min")
        print("  ", end="")
        for w_len, grp in sorted(grouped.items(), key=lambda x: x[0]):
            print(f'{w_len}: {round4(min(grp[0]))}', end=", ")
        print()
        # print("List:", number_list)

def main():
    target_words = list(occurences.keys())

    pool = mp.Pool(mp.cpu_count()//2)
    print(mp.cpu_count()//2)

    result = pool.map(single_word_qtfy, target_words)
    # print(result)
    for x in result[0][0]:
        print("=========", x, "=========")
        print_statishtics([len(r[0][x]) for r in result], target_words)
        print("\n")

    print([(x, sum([len(r[0][x]) for r in result])/len(result))
            for x in result[0][0]])

    json.dump({w: r for w, r in zip(target_words, result)},
            open("toks.json", 'w+'))

if __name__ == "__main__":
    main()

