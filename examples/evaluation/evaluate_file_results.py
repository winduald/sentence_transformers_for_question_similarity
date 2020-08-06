import sys
import os
import argparse
import numpy as np
import torch
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
import glob


def get_model_names(folder_path):
    paths = glob.glob(os.path.join(folder_path, '*'))
    model_names = []
    for path in paths:
        if os.path.isdir(path):
            model_names.append(os.path.basename(path))
    print(model_names)
    return model_names

def get_query(segs, q1_cols):
    return tuple([segs[i] for i in q1_cols]), len(q1_cols)


def map(out, th):
    num_queries = len(out)
    MAP = 0.0
    for qid in out:
        candidates = out[qid]
        # compute the number of relevant docs
        # get a list of precisions in the range(0,th)
        avg_prec = 0
        precisions = []
        num_correct = 0
        for i in range(min(th, len(candidates))):
            if candidates[i][0] == "true":
                num_correct += 1
                precisions.append(num_correct/(i+1))

        if precisions:
            avg_prec = sum(precisions)/len(precisions)

        MAP += avg_prec
    return MAP / num_queries

def main():
    # parser = argparse.ArgumentParser(description="Running evaluation based on prediction results")
    # parser.add_argument("--input-text", type=str, help="input text file", default='/private/home/xiaojianwu/projects/question_similarity/data/semeval/bsc_c/test.tsv')
    # parser.add_argument("--input-prediction", type=str, help="prediction results file, should be a torch file", default='/private/home/xiaojianwu/projects/question_similarity/results/semeval/prediction_bsc_c.txt')
    # parser.add_argument("--label-dict", type=str, help="path for label dict file, we use this to target true label", default='/private/home/xiaojianwu/projects/question_similarity/data/semeval/bsc_c/bin/label/dict.txt')
    # # parser.add_argument("--true-labels", nargs='+', help="names of true label", default=['1','2'])
    # parser.add_argument("--true-labels", type=str, help="names of true label", default='1_2')
    # parser.add_argument("--data-type", type=str, default='bsc_c')
    # parser.add_argument("--method", type=str, default='c', help='c, r, rank')
    # args = parser.parse_args()

    # print(args.true_labels)
    # print(args.data_type)
    # print(args.input_text)
    # print(args.input_prediction)
    # print(args.label_dict)
    # print(args.method)

    # orig_file = '/private/home/xiaojianwu/projects/sentence-transformers/examples/datasets/data/semeval/bs_r/test_sts.tsv'

    data_folder = '/private/home/xiaojianwu/projects/sentence-transformers/examples/datasets/data'
    task_name = 'semeval'
    label_col = 2
    q1_cols = [0]
    q2_cols = [1]
    true_labels = ['2']

    # score_file = '/private/home/xiaojianwu/projects/sentence-transformers/examples/datasets/results/similarity_evaluation_results.csv'
    score_col = 1

    # model_names = ['training_semeval-bs_r-continue_training-distilbert-base-nli-mean-tokens-2020-07-24_16-11-05']
    model_names = get_model_names('/checkpoint/xiaojianwu/sentBert/semeval')

    output_path_root = '/private/home/xiaojianwu/projects/sentence-transformers/examples/datasets/results'
    output_debug = True
    model_names = ['training_semeval-bs_r-continue_training-roberta-base-2020-07-27_12-01-16']
    for model_name in model_names:
        score_file = os.path.join(output_path_root, model_name + '.tsv')

        if 'b_r' in model_name:
            type_name = 'b_r'
        elif 'bs_r' in model_name:
            type_name = 'bs_r'
        elif 'bsc_r' in model_name:
            type_name = 'bsc_r'
        elif 's_r' in model_name:
            type_name = 's_r'
        else:
            raise NotImplementedError

        data_file = os.path.join(data_folder, task_name, type_name, 'test_sts.tsv')

        row_idx = 0
        prediction_map = defaultdict(list) # query => list of (label, score, q2)
        all_labels, all_scores = [], []
        with open(data_file, 'r', encoding='utf-8') as forig, \
            open(score_file, 'r') as fscore:
            for data_line, score_line in zip(forig, fscore):
                # print(line)
                
                segs = data_line.strip('\n').split('\t')
                label = segs[label_col] # assume label is always last column
                sent1, sent2_start = get_query(segs, q1_cols) # assume query is always first column

                scores = score_line.strip('\n').split('\t')

                all_labels.append(float(scores[score_col]))
                all_scores.append(float(label))
                prediction_map[sent1].append(
                    ('true' if label in true_labels else 'false', float(scores[score_col]), '\t'.join(segs[sent2_start: -1]), label)
                )
                row_idx += 1

        # ranking
        for sent1 in prediction_map:
            # prediction_map[sent1] = sorted(
            #     prediction_map[sent1], 
            #     key=lambda x: 1 if x[0] == 'true' else 0, 
            #     reverse=True
            # )
            prediction_map[sent1] = sorted(
                prediction_map[sent1], 
                key=lambda x: x[1], 
                reverse=True
            )
        # print(prediction_map)
        # exit(1)
        #output results
        # debug_output_file = '/private/home/xiaojianwu/projects/sentence-transformers/examples/datasets/results/debug/debug.txt'
        if output_debug:
            debug_output_file = os.path.join('/private/home/xiaojianwu/projects/sentence-transformers/examples/datasets/results/debug/', model_name + '.tsv' )
            with open(debug_output_file, 'w', encoding='utf-8') as fout:
                for sent1 in prediction_map:
                    for label, score, sent2, orig_label in prediction_map[sent1]:
                        # print(sent2)
                        if isinstance(sent1, tuple):
                            sent1 = '\t'.join(sent1)
                        # print(score, label, sent1, sent2)

                        if float(score) > 0.5 and orig_label == '0':
                            fout.write('%f\t%s\t%s#####\t%s\n' %(score, orig_label, sent1, sent2))

        # metrics
        # print(prediction_map)
        #calculating metric
        MAP = map(prediction_map, th=100)
        pearson_score = pearsonr(all_labels, all_scores)
        spearman_score = spearmanr(all_labels, all_scores)

        print('MAP pearsonr spearmanr\t{}\t{}\t{}\t{}'.format(model_name, MAP, pearson_score[0], spearman_score[0]))

        # overall_accuracy(prediction_map, threshold=0.5)

        #error analysis code
        # output_file_error = args.input_prediction.replace('.txt', '_error_analysis.txt')
        # threshold = 0.5
        # with open(output_file_error, 'w', encoding='utf-8') as fout:
        #     for sent1 in prediction_map:
        #         # print(len(prediction_map[sent1]))
        #         for label, score, sent2 in prediction_map[sent1]:
        #             if isinstance(sent1, tuple):
        #             sent1 = '\t'.join(sent1)
        #             if (score > threshold and label == 'false') or (score < threshold and label == 'true'):
        #                 fout.write('%f\t%s\t%s#####\t%s\n' %(score, label, sent1, sent2))

        # print_prediction_map_bf(prediction_map)
        # exit(1)



if __name__ == "__main__":
  main()






