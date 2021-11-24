import os
import argparse

from PyQt5.QtCore import reset
from model.maximum_match import max_match_model
from model.hmm import hmm_model
from utils import *


gold_root = 'icwb2-data/gold'
train_root = 'icwb2-data/training'
test_root = 'icwb2-data/testing'


def parse_args():
    parser = argparse.ArgumentParser(description='Chinese word segmentation system')
    parser.add_argument('-d', '--dataset', default='as', help='Choose the dataset: (as, cityu, msr, pku)')
    parser.add_argument('-a', '--alg', default='forward', help='Choose the algorithm: (forward, backward, bi-direction, hmm)')
    
    args = parser.parse_args()
    return args

def inference(input, alg):
    if input == '':
        return
    infer_type = 'pku'
    input = input.split('\n')
    
    res = ''
    if alg == 'forward' or alg == 'backward' or alg == 'bi-direction':
        infer_train_path = os.path.join(gold_root, infer_type + '_training_words.utf8')
        infer_model = max_match_model(infer_train_path)
        infer_model.load_dict()

        for line in input:
            if alg == 'forward':
                res += '/'.join(infer_model.forward(line.strip())) + '\n'
            if alg == 'backward':
                res += '/'.join(infer_model.backward(line.strip())) + '\n'
            if alg == 'bi-direction':
                res += '/'.join(infer_model.bi_direction(line.strip())) + '\n'
    else:
        infer_train_path = os.path.join(train_root, infer_type + '_training.utf8')
        infer_model = hmm_model()
        try:
            infer_model.load_params(infer_type)
        except:
            infer_model.train(infer_train_path)
            infer_model.save_params(infer_type)
        for line in input:
            print(line)
            print(infer_model.eval(line.strip()))
            res += '/'.join(infer_model.eval(line.strip())) + '\n'

    return res

def main():
    args = parse_args()
    try:
        if args.alg == 'forward' or args.alg == 'backward' or args.alg == 'bi-direction':
            train_path = os.path.join(gold_root, args.dataset + '_training_words.utf8')
            test_path = os.path.join(gold_root, args.dataset + '_test_gold.utf8')

            model = max_match_model(train_path)
            model.load_dict()

            if args.alg == 'forward':
                P, R, F = score(model.forward, test_path)
            if args.alg == 'backward':
                P, R, F = score(model.backward, test_path)
            if args.alg == 'bi-direction':
                P, R, F = score(model.bi_direction, test_path)
        
        elif args.alg == 'hmm':
            train_path = os.path.join(train_root, args.dataset + '_training.utf8')
            test_path = os.path.join(gold_root, args.dataset + '_test_gold.utf8')

            model = hmm_model()
            try:
                model.load_params(args.dataset)
            except:
                model.train(train_path)
                model.save_params(args.dataset)
            P, R, F = score(model.eval, test_path)
        
        else:
            print('The chosen algorithm is not included.')
    except:
        print('The chosen dataset is not included.')

    
    print(P, R, F)

if __name__ == '__main__':
    main()
