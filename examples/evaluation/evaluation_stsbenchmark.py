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

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader
import logging
import torch
from sentence_transformers import models, losses

# script_folder_path = os.path.dirname(os.path.realpath(__file__))
script_folder_path = '/checkpoint/xiaojianwu/data/sentBERT'
#Limit torch to 4 threads
torch.set_num_threads(4)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-nli-mean-tokens'

# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
pretrained = False
if pretrained:
    model = SentenceTransformer(model_name)
else:
    word_embedding_model = models.Transformer('bert-base-uncased')
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# sts_reader = STSBenchmarkDataReader(os.path.join(script_folder_path, '../datasets/stsbenchmark'))
sts_reader = STSBenchmarkDataReader(os.path.join(script_folder_path, 'stsbenchmark'))

test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=8)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

model.evaluate(evaluator)
