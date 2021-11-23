import os
import numpy as np
from utils import *


class hmm_model:
    def __init__(self):
        self.state_set = ['B', 'M', 'E', 'S']
        self.trans_prob_mat = {}
        self.emit_prob_mat = {}
        self.init_prob_mat = {}
        self.word_set = set()
        self.word_count_dict = {}
        self.n_line = 0
        self.path = []
        self.save_root = 'model/model_saved/hmm/'


    #初始化所有概率矩阵
    def initialize_mat(self):
        for s in self.state_set:
            self.trans_prob_mat[s] = {}
            for s_ in self.state_set:
                self.trans_prob_mat[s][s_] = 0.0
            self.init_prob_mat[s] = 0.0
            self.emit_prob_mat[s] = {}
            self.word_count_dict[s] = 0

    #对训练集获取状态标签
    def get_state_label(self, word):
        label = []
        if len(word) == 1:
            label = ['S']
        elif len(word) == 2:
            label = ['B', 'E']
        else:
            num = len(word) - 2
            label.append('B')
            label.extend(['M'] * num)
            label.append('E')
        return label

    #将参数估计的概率取对数，对概率0取无穷小-3.14e+100
    def get_log_prob_mat(self):
        for s in self.init_prob_mat:
            if self.init_prob_mat[s] == 0:
                self.init_prob_mat[s] = -3.14e+100
            else:
                self.init_prob_mat[s] = np.log(self.init_prob_mat[s] / self.n_line)
        for s in self.trans_prob_mat:
            for s_ in self.trans_prob_mat[s]:
                if self.trans_prob_mat[s][s_] == 0.0:
                    self.trans_prob_mat[s][s_] = -3.14e+100
                else:
                    self.trans_prob_mat[s][s_] = np.log(self.trans_prob_mat[s][s_] / self.word_count_dict[s])
        for s in self.emit_prob_mat:
            for w in self.emit_prob_mat[s]:
                if self.emit_prob_mat[s][w] == 0.0:
                    self.emit_prob_mat[s][w] = -3.14e+100
                else:
                    self.emit_prob_mat[s][w] = np.log(self.emit_prob_mat[s][w] / self.word_count_dict[s])

    #Viterbi算法求测试集最优状态序列
    def viterbi(self, line):
        rec_tab = [{}]
        path = {}

        if line[0] not in self.emit_prob_mat['B']:
            for s in self.state_set:
                if s == 'S':
                    self.emit_prob_mat[s][line[0]] = 0
                else:
                    self.emit_prob_mat[s][line[0]] = -3.14e+100

        for s in self.state_set:
            rec_tab[0][s] = self.init_prob_mat[s] + self.emit_prob_mat[s][line[0]]
            path[s] = [s]
        for i in range(1,len(line)):
            rec_tab.append({})
            new_path = {}

            for s in self.state_set:
                if s == 'B':
                    self.emit_prob_mat[s]['begin'] = 0
                else:
                    self.emit_prob_mat[s]['begin'] = -3.14e+100
                if s == 'E':
                    self.emit_prob_mat[s]['end'] = 0
                else:
                    self.emit_prob_mat[s]['end'] = -3.14e+100

            for s in self.state_set:
                items = []
                for s_ in self.state_set:
                    if line[i] not in self.emit_prob_mat[s]:
                        if line[i-1] not in self.emit_prob_mat[s]:
                            log_prob = rec_tab[i - 1][s_] + self.trans_prob_mat[s_][s] + self.emit_prob_mat[s]['end']
                        else:
                            log_prob = rec_tab[i - 1][s_] + self.trans_prob_mat[s_][s] + self.emit_prob_mat[s]['begin']
                    else:
                        log_prob = rec_tab[i - 1][s_] + self.trans_prob_mat[s_][s] + self.emit_prob_mat[s][line[i]]
                    items.append((log_prob,s_))
                best_path = max(items)
                rec_tab[i][s] = best_path[0]
                new_path[s] = path[best_path[1]] + [s]
            path = new_path

        _, max_state = max([(rec_tab[len(line) - 1][s], s) for s in self.state_set])
        self.path = path[max_state]

    #根据状态序列进行分词
    def seg_line(self, line, label):
        word_list = []
        start_index = -1
        start_flag = False

        if len(label) != len(line):
            return None

        if len(label) == 1:
            word_list.append(line[0])

        else:
            if label[-1] == 'B' or label[-1] == 'M':
                if label[-2] == 'B' or label[-2] == 'M':
                    label[-1] = 'S'
                else:
                    label[-1] = 'E'


            for i in range(len(label)):
                if label[i] == 'S':
                    if start_flag:
                        start_flag = False
                        word_list.append(line[start_index:i])
                    word_list.append(line[i])
                elif label[i] == 'B':
                    if start_flag:
                        word_list.append(line[start_index:i])
                    start_index = i
                    start_flag = True
                elif label[i] == 'E':
                    start_flag = False
                    word = line[start_index:i + 1]
                    word_list.append(word)
                elif label[i] == 'M':
                    continue

        return word_list

    def train(self, trainset): 
        self.initialize_mat()
        for idx, line in enumerate(read_file(trainset)):
            line = line.replace('\u3000', '  ')
            line = line.replace('  ', ' ')
            self.n_line += 1

            word_list = []
            for i in range(len(line)):
                if line[i] == ' ':
                    continue
                word_list.append(line[i])
            self.word_set = self.word_set | set(word_list)

            line = line.split()
            line_state = []
            for word in line:
                line_state.extend(self.get_state_label(word))

            if len(line_state) == 0:
                continue
            self.init_prob_mat[line_state[0]] += 1

            for i in range(len(line_state)-1):
                self.trans_prob_mat[line_state[i]][line_state[i+1]] += 1

            for i in range(len(line_state)):
                self.word_count_dict[line_state[i]] += 1
                for s in self.state_set:
                    try:
                        if word_list[i] not in self.emit_prob_mat[s]:
                            self.emit_prob_mat[s][word_list[i]] = 0.0
                    except:
                        print(word_list)
                        print(line_state)
                        print(line)
                        print(i)
                self.emit_prob_mat[line_state[i]][word_list[i]] += 1

        self.get_log_prob_mat()

    def load_params(self):
        with open(os.path.join(self.save_root, 'trans_prob_mat.txt'), 'rt') as f:
            self.trans_prob_mat = eval(f.read())

        with open(os.path.join(self.save_root, 'emit_prob_mat.txt'), 'rt') as f:
            self.emit_prob_mat = eval(f.read())

        with open(os.path.join(self.save_root, 'init_prob_mat.txt'), 'rt') as f:
            self.init_prob_mat = eval(f.read())

    def save_params(self):
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)

        with open(os.path.join(self.save_root, 'trans_prob_mat.txt'), 'wt') as f:
            f.write(str(self.trans_prob_mat))

        with open(os.path.join(self.save_root, 'emit_prob_mat.txt'), 'wt') as f:
            f.write(str(self.emit_prob_mat))

        with open(os.path.join(self.save_root, 'init_prob_mat.txt'), 'wt') as f:
            f.write(str(self.init_prob_mat))

    def eval(self, line):
        if line == '':
            return line
        self.viterbi(line)
        res = self.seg_line(line, self.path)

        return res
