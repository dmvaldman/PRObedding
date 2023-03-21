import numpy as np
import math
import openai
from tqdm import tqdm
import os
from dotenv import load_dotenv
import pandas as pd
from tenacity import retry, stop_after_attempt
from datasets import load_dataset

load_dotenv()

openai_key = os.getenv('OPENAI_API_KEY')

openai.api_key = openai_key

MAX_CONTENT_LENGTH = 8191

@retry(stop=stop_after_attempt(5))
def get_openai_embeddings(inputs, model):
    # make sure each review is a under a certain length
    for index, input in enumerate(inputs):
        input = input.replace('\n', ' ')
        if len(input) > MAX_CONTENT_LENGTH:
            input = input[:MAX_CONTENT_LENGTH]
        inputs[index] = input

    response = openai.Embedding.create(
        input=inputs,
        model=model)

    embeddings = [embed_data['embedding'] for embed_data in response['data']]
    return embeddings

class Dataset():
    def __init__(self, type, config=None, save=False, split='train'):
        self.type = type
        self.dataset_path = config['path']

        self.local_path = None
        self.config = config

        self.texts = []
        self.ratings = np.uint([])
        self.embeds = np.float64([])

        self.download(split=split)
        self.create_text_embeddings(split=split)

        if save:
            self.save(split)

    def download(self, split='train'):
        dataset = load_dataset(self.dataset_path, split=split)
        text_key, label_key = self.config["columns"]

        def delete_nulls(row):
            return all(v is not None for v in row.values())

        # apply the filter to the dataset
        dataset = dataset.filter(delete_nulls)

        self.texts = dataset[text_key]
        self.ratings = np.array(dataset[label_key], dtype=np.uint)

    def create_text_embeddings(self, model="text-embedding-ada-002", batch_size=50,split='train'):
        # check if embeddings already exist
        is_loaded = self.load(split=split)

        # if not, create them
        if not is_loaded:
            reviews = self.texts
            N = len(reviews)
            num_full_batches = math.ceil(N / batch_size)
            embeddings = []
            for index in tqdm(range(num_full_batches)):
                embeddings_batch = self._get_batch_at(index * batch_size, reviews, model=model, batch_size=batch_size)
                embeddings += embeddings_batch

            self.embeds = np.array(embeddings, dtype=np.float64)

    def _get_batch_at(self, start, reviews, model="text-embedding-ada-002", batch_size=10):
        end = start + batch_size
        if end > len(reviews):
            end = len(reviews)

        reviews_subset = reviews[start:end]
        return get_openai_embeddings(reviews_subset, model)

    def _get_save_path(self, split):
        dataset_path = self.dataset_path.replace('/', '_')
        path = f'embeddings/{self.type}/{dataset_path}/{split}/'
        return path

    def save(self, split='train'):
        path = self._get_save_path(split)

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

    def load(self, split='train'):
        path = self._get_save_path(split)

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

    def create_text_embeddings(self, model="text-embedding-ada-002", batch_size=50, split='train'):
        for dataset in self.datasets:
            dataset.create_text_embeddings(model, batch_size, split=split)

    def save(self, split='train'):
        for dataset in self.datasets:
            dataset.save(split=split)

    def load(self, split='train'):
        for dataset in self.datasets:
            is_loaded = dataset.load(split=split)
            if not is_loaded:
                return False
        self._merge()