def read_file(file_path):
    with open(file_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            yield line.strip()

def count_single(line):
        n = 0
        for w in line:
            if len(w) == 1:
                n += 1
        return n

def score(model, test_path):
    count = 1
    count_right = 0
    count_split = 0
    count_gold = 0
    process_count = 0
    for idx, line in enumerate(read_file(test_path)):
        process_count += 1
        line = line.replace('\u3000', '  ')
        gold_word = line.split()
        #print(gold_word)
        line = line.replace('  ', '')
        line = line.replace(' ', '')

        res = model(line)
        #print(res)
        count += 1
        count_split += len(res)
        count_gold += len(gold_word)
        tmp_res = res
        tmp_gold = gold_word

        for word in tmp_res:
            if word in tmp_gold:
                count_right += 1
                tmp_gold.remove(word)

    P = count_right / count_split
    R = count_right / count_gold
    F = 2 * P * R / (P + R)

    return P, R, F
