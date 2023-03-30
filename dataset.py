import numpy as np
import math
import openai
from tqdm import tqdm
import os
from dotenv import load_dotenv
import pandas as pd
from tenacity import retry, stop_after_attempt
from datasets import load_dataset

import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

load_dotenv()

MAX_CONTENT_LENGTHS = {
    "openai": 8191,
    "e5": 1024
}

def replace_newline(text):
    return text.replace('\n', ' ')

def preprocess(texts, model, methods=None):
    max_content_length = MAX_CONTENT_LENGTHS[model]
    for i, text in enumerate(texts):
        if methods is not None:
            for method in methods:
                text = method(text)
        if len(text) > max_content_length:
            text = text[:max_content_length]
            texts[i] = text
    return texts

@retry(stop=stop_after_attempt(5))
def get_openai_embeddings(inputs, model=None):
    # make sure each review is a under a certain length
    methods = [replace_newline]
    inputs = preprocess(inputs, 'openai', methods)

    response = openai.Embedding.create(
        input=inputs,
        model=model)

    embeddings = [embed_data['embedding'] for embed_data in response['data']]
    return embeddings

def get_E5_embeddings(inputs, tokenizer=None, model=None, device='cpu'):
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # For E5 model prefix all inputs with "query: "
    inputs = [f"query: {input}" for input in inputs]
    inputs = preprocess(inputs, 'e5')

    model.to(device)
    model.eval()

    # Tokenize the input texts
    batch_dict = tokenizer(inputs, max_length=512, padding=True, truncation=True, return_tensors='pt')

    # Move the batch to the device
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return embeddings


class Embedder():
    def __init__(self, type):
        self.type = type
        if type == 'openai':
            self.model = "text-embedding-ada-002"
            self.tokenizer = None
            openai.api_key = os.getenv('OPENAI_API_KEY')
        elif type == 'E5':
            self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large')
            self.model = AutoModel.from_pretrained('intfloat/e5-large')

    def embed(self, dataset, batch_size=50, save=False, device='cpu'):
        if dataset.load(embedder=self.type):
            return

        reviews = dataset.texts
        N = len(reviews)
        num_full_batches = math.ceil(N / batch_size)
        embeddings = []
        for index in tqdm(range(num_full_batches)):
            embeddings_batch = self._get_batch_at(index * batch_size, reviews, batch_size=batch_size, device=device)
            embeddings += embeddings_batch

        dataset.embeds = np.array(embeddings, dtype=np.float64)
        if save: dataset.save(self.type)

    def _get_batch_at(self, start, reviews, batch_size=10, device='cpu'):
        end = start + batch_size
        if end > len(reviews): end = len(reviews)
        reviews_subset = reviews[start:end]

        if self.type == 'openai':
            return get_openai_embeddings(reviews_subset, model=self.model)
        elif self.type == 'E5':
            return get_E5_embeddings(reviews_subset, tokenizer=self.tokenizer, model=self.model, device=device)


class Dataset():
    def __init__(self, type, config=None, split='train'):
        self.type = type
        self.split = split
        self.dataset_path = config['path']

        self.local_path = None
        self.config = config

        self.texts = []
        self.ratings = np.uint([])
        self.embeds = np.float64([])

        self.download(split=split)

    def download(self, split='train'):
        dataset = load_dataset(self.dataset_path, split=split)
        text_key, label_key = self.config["columns"]

        def delete_nulls(row):
            return all(v is not None for v in row.values())

        # apply the filter to the dataset
        dataset = dataset.filter(delete_nulls)

        self.texts = dataset[text_key]
        self.ratings = np.array(dataset[label_key], dtype=np.uint)

    def _get_save_path(self, split, embedder):
        dataset_path = self.dataset_path.replace('/', '_')
        path = f'embeddings/{self.type}/{embedder}/{dataset_path}/{split}/'
        return path

    def save(self, split='train', embedder='openai'):
        path = self._get_save_path(split, embedder)

        # make directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        # if too large, save in chunks
        if self.embeds.nbytes > 1e9:
            chunk_size = 10000
            chunks = np.array_split(self.embeds, chunk_size)
            np.savez_compressed(path + '/embeddings.npz', *chunks)
        else:
            np.save(path + '/embeddings.npy', self.embeds)

        np.save(path + '/ratings.npy', self.ratings)

    def load(self, split='train', embedder='openai'):
        path = self._get_save_path(split, embedder)

        if (os.path.exists(path + '/embeddings.npy') or os.path.exists(path + '/embeddings.npz')) and os.path.exists(path + '/ratings.npy'):
            if os.path.exists(path + '/embeddings.npz'):
                loaded_data = np.load(path + '/embeddings.npz')
                self.embeds = np.concatenate([loaded_data[key] for key in loaded_data.keys()])
            else:
                self.embeds = np.load(path + '/embeddings.npy')

            self.ratings = np.load(path + '/ratings.npy')
            return True
        else:
            return False

class MergedDataset():
    def __init__(self, type, config, save=False, split='train'):
        self.datasets = []
        self.ratings = None
        self.embeds = None

        datasets = config["datasets"]
        for config in datasets:
            dataset = Dataset(type, config, save=save, split=split)
            self.datasets.append(dataset)

        self._merge()

    def _merge(self):
        self.embeds = np.vstack([dataset.embeds for dataset in self.datasets])
        self.ratings = np.hstack([dataset.ratings for dataset in self.datasets])

    # def create_text_embeddings(self, model="text-embedding-ada-002", batch_size=50, split='train'):
    #     for dataset in self.datasets:
    #         dataset.create_text_embeddings(model, batch_size, split=split)

    def save(self, split='train'):
        for dataset in self.datasets:
            dataset.save(split=split)

    def load(self, split='train'):
        for dataset in self.datasets:
            is_loaded = dataset.load(split=split)
            if not is_loaded:
                return False
        self._merge()