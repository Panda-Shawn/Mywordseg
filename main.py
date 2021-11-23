import os
import argparse
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

# def ui_word_seg():

def main():
    args = parse_args()
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
        if os.path.exists(model.save_root):
            model.load_params()
        else:
            model.train(train_path)
            model.save_params()
        P, R, F = score(model.eval, test_path)

    print(P, R, F)

if __name__ == '__main__':
    main()
