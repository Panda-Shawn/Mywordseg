import os


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
    res = ''
    for idx, line in enumerate(read_file(test_path)):
        process_count += 1
        line = line.replace('\u3000', '  ')
        tmp_gold = line.split()
        #print(gold_word)
        line = line.replace('  ', '')
        line = line.replace(' ', '')
        tmp_res = model(line)

        count += 1
        count_split += len(tmp_res)
        count_gold += len(tmp_gold)
        res += '/'.join(tmp_res) + '\n'

        for word in tmp_res:
            if word in tmp_gold:
                count_right += 1
                tmp_gold.remove(word)

    test_type = test_path.split('\\')[-1]
    test_type = test_type[:test_type.find('_')]
    output_path = os.path.join('output', test_type + '_output.txt')

    if not os.path.exists('output'):
        os.mkdir('output')

    with open(output_path, 'wt', encoding='utf-8') as f:
        f.write(res)

    P = count_right / count_split
    R = count_right / count_gold
    F = 2 * P * R / (P + R)

    return P, R, F
