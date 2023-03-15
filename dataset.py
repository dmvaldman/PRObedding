import numpy as np
import math
import openai
from tqdm import tqdm
import os
from dotenv import load_dotenv
import kaggle
import pandas as pd
from tenacity import retry, stop_after_attempt

load_dotenv()

openai_key = os.getenv('OPENAI_API_KEY')

openai.api_key = openai_key
kaggle.api.authenticate()

MAX_CONTENT_LENGTH = 8191

@retry(stop=stop_after_attempt(3))
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
    def __init__(self, type, kaggle_path, config=None, save=False):
        self.type = type
        self.kaggle_path = kaggle_path

        self.local_path = None
        self.encoding = 'utf-8'
        self.df = None
        self.config = config

        self.texts = []
        self.ratings = np.uint([])
        self.embeds = np.float64([])

        self.parse_config(config)
        self.download()
        self.preprocess_data()
        self.create_text_embeddings()

        if save:
            self.save()

    def parse_config(self, config):
        unpack_name = config["unpack_name"]
        encoding = config["encoding"] if "encoding" in config else 'utf-8'

        self.local_path = f'datasets/{self.type}/{unpack_name}'
        self.encoding = encoding

    def download(self):
        path = f'datasets/{self.type}/'
        kaggle.api.dataset_download_files(self.kaggle_path, path=path, force=True, unzip=True)
        self.df = pd.read_csv(self.local_path, encoding=self.encoding)

    def preprocess_data(self):
        ratings_key = self.config["columns"]["rating"]["label"]
        text_key = self.config["columns"]["text"]["label"]

        # delete any rows that have missing values
        self.df = self.df.dropna(subset=[ratings_key, text_key])

        if "map" in self.config["columns"]["rating"]:
            map = self.config["columns"]["rating"]["map"]
            self.df[ratings_key] = self.df[ratings_key].replace(map.keys(), map.values())
        self.ratings = np.array(self.df[ratings_key].tolist(), dtype=np.uint)

        text_key = self.config["columns"]["text"]["label"]
        if "map" in self.config["columns"]["text"]:
            map = self.config["columns"]["text"]["map"]
            self.df[ratings_key] = self.df[ratings_key].replace(map.keys(), map.values())
        self.texts = self.df[text_key].tolist()

    def create_text_embeddings(self, model="text-embedding-ada-002", batch_size=50):
        # check if embeddings already exist
        is_loaded = self.load()

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

    def _get_save_path(self):
        kaggle_path = self.kaggle_path.replace('/', '_')
        path = f'embeddings/{self.type}/{kaggle_path}'
        return path

    def save(self):
        path = self._get_save_path()

        # make directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(path + '/embeddings.npy', self.embeds)
        np.save(path + '/ratings.npy', self.ratings)

    def load(self):
        path = self._get_save_path()

        if os.path.exists(path + '/embeddings.npy') and os.path.exists(path + '/ratings.npy'):
            self.embeds = np.load(path + '/embeddings.npy')
            self.ratings = np.load(path + '/ratings.npy')
            return True
        else:
            return False

class MergedDataset():
    def __init__(self, type, config, save=False):
        self.datasets = []
        self.ratings = None
        self.embeds = None

        datasets = config["datasets"]
        for kaggle_path, config_dataset in datasets.items():
            dataset = Dataset(type, kaggle_path, config_dataset, save=save)
            self.datasets.append(dataset)

        self._merge()

    def _merge(self):
        self.embeds = np.vstack([dataset.embeds for dataset in self.datasets])
        self.ratings = np.hstack([dataset.ratings for dataset in self.datasets])

    def create_text_embeddings(self, model="text-embedding-ada-002", batch_size=50):
        for dataset in self.datasets:
            dataset.create_text_embeddings(model, batch_size)

    def save(self):
        for dataset in self.datasets:
            dataset.save()

    def load(self):
        for dataset in self.datasets:
            is_loaded = dataset.load()
            if not is_loaded:
                return False
        self._merge()