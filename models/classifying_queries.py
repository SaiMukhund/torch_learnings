from transformers import AutoTokenizer ,AutoModel
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-base')
embeddings = model.encode(input_texts, normalize_embeddings=True)
def read_yaml_data(path):
    pass

def train_model(data):
    pass







def main():
    pass