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

# script_folder_path = os.path.dirname(os.path.realpath(__file__))
# print(os.path.realpath(__file__))
data_folder = '/private/home/xiaojianwu/projects/sentence-transformers/examples/datasets/data/semeval/bs_r'
#Limit torch to 4 threads
# exit(1)
# torch.set_num_threads(4)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-nli-mean-tokens'
model_name = '/checkpoint/xiaojianwu/sentBert/test/training_semeval-bs_r-continue_training-bert-base-nli-mean-tokens-2020-07-25_23-02-02'
output_path = '/private/home/xiaojianwu/projects/sentence-transformers/examples/datasets/results/results.tsv'
# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
pretrained = True
if pretrained:
    model = SentenceTransformer(model_name)
else:
    word_embedding_model = models.Transformer('bert-base-uncased')
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# sts_reader = STSBenchmarkDataReader(os.path.join(script_folder_path, '../datasets/stsbenchmark'))
sts_reader = STSBenchmarkDataReader(data_folder, s1_col_idx=0, s2_col_idx=1, score_col_idx=2, delimiter="\t", min_score=0, max_score=1)

test_data = SentencesDataset(examples=sts_reader.get_examples("test_sts.tsv"), model=model, )
print("DataLoader")
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=8)
print("EmbeddingSimilarityEvaluator")
evaluator = EmbeddingSimilarityEvaluator(test_dataloader, show_progress_bar=False)

print(evaluator)
# print(model)
# print(model.evaluate)
# exit(1)

model.evaluate(evaluator, output_path)
