ngram_range = 2

def create_ngram(input_list):
    return set(zip(*[input_list[i:] for i in range(ngram_range)]))

print(create_ngram([1, 4, 9, 4, 1, 4]))

