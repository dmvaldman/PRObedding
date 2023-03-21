# PRObedding

A simple library for extracting directions in text emebdding space. Supports binary and multiclass classification on any HuggingFace text dataset using OpenAI embeddings. Other models coming soon.

## Why?

Have you ever wondered if there is a "spam/not spam" direction in embedding space? You can often train a linear classifier (ie a probe) on text embeddings and uncover a stable "direction" which can be used for downstream purposes. To classify any text embedding, all you need to do pretty much is dot product against it with a direction.

It turns out that a simple linear classifier over OpenAI embeddings is competitive with current SoTA classifiers.

## How good is it?

Comparison to some text classification tasks on HuggingFace Leaderboards:

| dataset | task | accuracy (leaderboard) | accuracy (ours) | Δ |
|---------|------|------------------------| --------------- | - |
| rotten tomatoes | binary classification | 84.0%  | 88.5% | +1.5% |
| imdb | binary classification | 93.2%  | 93.0% | -0.2% |
| ag news | multi classification | 93.9%  | 92.3% | -1.6% |

## How to use

Follow the notebook examples in the `/examples` directory. You don't need a GPU, the linear probes are computed using old skool methods since they can be solved for exactly. Just make a new config file that specifies some dataset(s) as in the `/examples/configs` directory, follow the simple examples, and you're off to the races.
