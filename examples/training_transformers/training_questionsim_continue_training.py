"""
This example loads the pre-trained SentenceTransformer model 'bert-base-nli-mean-tokens' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.

Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
import sys
import os

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(root_path)
sys.path.append(root_path)

from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.readers import STSBenchmarkDataReader
import logging
from datetime import datetime
from sentence_transformers import models, losses

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--model-name', type=str, default='bert-base-uncased')
parser.add_argument('--data-folder', type=str, default='/private/home/xiaojianwu/projects/sentence-transformers/examples/datasets/data')
parser.add_argument('--base-folder-path', type=str, default='/checkpoint/xiaojianwu/sentBert/test')
parser.add_argument('--task-name', type=str, default='semeval')
parser.add_argument('--type-name', type=str, default='b_r')
parser.add_argument('--num-epoches', type=int, default=10)
parser.add_argument('--train-batch-size', type=int, default=16)
parser.add_argument('--type', type=str, default='b')
parser.add_argument('--model-length', type=int, default=128)
args = parser.parse_args()

'''
python examples/training_transformers/training_questionsim_continue_training.py --pretrained --task-name semeval --type-name b_r --num-epoches 4 --model-name bert-base-nli-mean-tokens
python examples/training_transformers/training_questionsim_continue_training.py --task-name semeval --type-name bs_r --num-epoches 1 --model-name bert-base-uncased
'''

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
'''
model_name=bert-base-nli-mean-tokens
model_name=bert-base-uncased
'''
pretrained = args.pretrained
print(args.pretrained)
print(args.model_name)
assert ('nli' in args.model_name and args.pretrained) or ('nli' not in args.model_name and args.pretrained == False), 'pretrained=True means model_name contains nli'
model_name = args.model_name

if pretrained:
    # model_name = 'bert-base-nli-mean-tokens'
    model = SentenceTransformer(model_name, max_seq_length=args.model_length)
else:
    # model_name = 'bert-base-uncased'
    word_embedding_model = models.Transformer(model_name, max_seq_length=args.model_length)
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

folder_path = args.base_folder_path
data_path_folder = os.path.join(args.data_folder, args.task_name, args.type_name)
train_name, dev_name, test_name = 'train_sts.tsv', 'dev_sts.tsv', 'test_sts.tsv'

print("data path folder %s" %data_path_folder)

train_batch_size = args.train_batch_size
num_epochs = args.num_epoches
# model_save_path = 'output/training_stsbenchmark_continue_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = os.path.join(folder_path, 'training_' + args.task_name + '-' + args.type_name + '-continue_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

#max score is different
if args.task_name == 'semeval':
    max_score, min_score = 2, 0
elif args.task_name == 'askubuntu' or args.task_name == 'quora':
    max_score, min_score = 1, 0
else:
    raise NotImplementedError
print("max min score", max_score, min_score)

sts_reader = STSBenchmarkDataReader(data_path_folder, normalize_scores=True, s1_col_idx=0, s2_col_idx=1, score_col_idx=2, delimiter="\t", min_score=min_score, max_score=max_score)
# Load a pre-trained sentence transformer model

# Convert the dataset to a DataLoader ready for training
logging.info("Read question similarity train dataset")
train_dataset = SentencesDataset(sts_reader.get_examples(train_name), model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


logging.info("Read question similarity dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples(dev_name), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader, SimilarityFunction.COSINE)

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=sts_reader.get_examples(test_name), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader, SimilarityFunction.COSINE)
model.evaluate(evaluator)
