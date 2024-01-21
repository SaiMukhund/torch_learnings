from transformers import AutoTokenizer ,AutoModel
import torch
"""smalltalkkb.tsv: https://github.com/alyssaong1/botframework-smalltalk/blob/master/smalltalkkb.tsv
smalltalk2: https://github.com/zeloru/small-english-smalltalk-corpus/tree/master 
chatterbot :https://github.com/gunthercox/chatterbot-corpus/tree/master?tab=readme-ov-file"""
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-base')
embeddings = model.encode(input_texts, normalize_embeddings=True)
def read_yaml_data(path):
    pass

def train_model(data):
    pass







def main():
    pass