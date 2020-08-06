from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch

model = SentenceTransformer('bert-base-nli-mean-tokens')

sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)
# print(len(sentence_embeddings))
# print(sentence_embeddings[2])

val = F.cosine_similarity(torch.tensor(sentence_embeddings[0]), torch.tensor(sentence_embeddings[1]), dim=0)
print(val)

 