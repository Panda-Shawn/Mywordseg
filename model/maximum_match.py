import numpy as np
from utils import *


class max_match_model:
    def __init__(self, dict_path):
        self.dict_path = dict_path
        self.word_dict_set = set()

    def load_dict(self):
        for idx, word in enumerate(read_file(self.dict_path)):
            self.word_dict_set.add(word)

    def forward(self, line):
        n_line = len(line)
        start, end = 0, n_line
        result = []
        while start < n_line:
            n = n_line - start
            if n == 1:
                result.append(line[start:])
                return result
            current_word = line[start: end]
            if current_word in self.word_dict_set:
                result.append(current_word)
                start = end
                end = n_line
                continue
            else:
                if len(current_word) == 1:
                    self.word_dict_set.add(current_word)
                    result.append(current_word)
                    start = end
                    end = n_line
                    continue
                end -= 1
                continue
            start += 1
        return result

    def backward(self, line):
        n_line = len(line)
        start, end = 0, n_line
        result = []
        while end > 0:
            if end == 1:
                result.append(line[start: end])
                return result[::-1]
            current_word = line[start: end]
            if current_word in self.word_dict_set:
                result.append(current_word)
                end = start
                start = 0
                continue
            else:
                if len(current_word) == 1:
                    self.word_dict_set.add(current_word)
                    result.append(current_word)
                    end = start
                    start = 0
                    continue
                start += 1
                continue
            end -= 1
        return result[::-1]
    
    def bi_direction(self, line):
        forward_res = self.forward(line)
        forward_count = len(forward_res)

        backward_res = self.backward(line)
        backward_count = len(backward_res)

        if forward_count < backward_count:
            return forward_res
        elif forward_count > backward_count:
            return backward_res
        else:
            if count_single(forward_res) < count_single(backward_res):
                return forward_res
            else:
                return backward_res
