"""
This examples loads a pre-trained model and evaluates it on the STSbenchmark dataset

Usage:
python evaluation_stsbenchmark.py
OR
python evaluation_stsbenchmark.py model_name
"""
import sys
import os

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(root_path)
sys.path.append(root_path)
# exit(1)

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader
import logging

import torch
from sentence_transformers import models, losses
import glob

def get_model_names(folder_path):
    paths = glob.glob(os.path.join(folder_path, '*'))
    model_names = []
    for path in paths:
        if os.path.isdir(path):
            model_names.append(os.path.basename(path))
    print(model_names)
    return model_names


if __name__ == '__main__':

    # script_folder_path = os.path.dirname(os.path.realpath(__file__))
    # print(os.path.realpath(__file__))
    data_folder_base = '/private/home/xiaojianwu/projects/sentence-transformers/examples/datasets/data/semeval'

    #Limit torch to 4 threads
    # exit(1)
    # torch.set_num_threads(4)

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    task_name = 'semeval'

    # model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-nli-mean-tokens'
    model_folder = os.path.join('/checkpoint/xiaojianwu/sentBert', task_name)
    # model_names = ['training_semeval-bs_r-continue_training-distilbert-base-nli-mean-tokens-2020-07-24_16-11-05']
    model_names = get_model_names(os.path.join('/checkpoint/xiaojianwu/sentBert', task_name))
    # model_names = ['training_semeval-b_r-continue_training-roberta-base-2020-07-27_12-01-10']

    output_path_root = '/private/home/xiaojianwu/projects/sentence-transformers/examples/datasets/results'
    # Load a named sentence model (based on BERT). This will download the model from our server.
    # Alternatively, you can also pass a filepath to SentenceTransformer()

    for model_name in model_names:
        
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

        output_path = os.path.join(output_path_root, model_name + '.tsv')
        model_path = os.path.join(model_folder, model_name)
        data_folder = os.path.join(data_folder_base, type_name)

        # if pretrained:
        model = SentenceTransformer(model_path)
        # else:
        #     word_embedding_model = models.Transformer(model_path)
        #     # Apply mean pooling to get one fixed sized sentence vector
        #     pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        #     model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print("loading model")
        if task_name == 'semeval':
            max_score, min_score = 2, 0
        elif task_name == 'askubuntu' or task_name == 'quora':
            max_score, min_score = 1, 0
        else:
            raise NotImplementedError
        # sts_reader = STSBenchmarkDataReader(os.path.join(script_folder_path, '../datasets/stsbenchmark'))
        sts_reader = STSBenchmarkDataReader(data_folder, s1_col_idx=0, s2_col_idx=1, score_col_idx=2, delimiter="\t", min_score=min_score, max_score=max_score)

        test_data = SentencesDataset(examples=sts_reader.get_examples("test_sts.tsv"), model=model, )
        print("DataLoader")
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=8)
        print("EmbeddingSimilarityEvaluator")
        evaluator = EmbeddingSimilarityEvaluator(test_dataloader, show_progress_bar=False)

        # print(model)
        # print(model.evaluate)
        # exit(1)

        model.evaluate(evaluator, output_path=output_path)
