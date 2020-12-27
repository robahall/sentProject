import torch
import random
from torch.utils.data import DataLoader
from torchtext import data, datasets
import pytorch_lightning as pl

from src import ROOT_DIR

class IMDBDataModule(pl.LightningDataModule):

    def __init__(self, hparams, random_seed=random.seed(123)):
        super().__init__()
        self.random_seed = random_seed
        self.text = data.Field(tokenize='spacy',  # Use spaCy tokenizer
                               include_lengths=hparams.include_lengths  #how long our sequence is (tuple)
                               )
        self.label = data.LabelField(dtype=torch.float)
        self.batch_size = hparams.batch_size

    def prepare_data(self):
        train, test = datasets.IMDB.splits(self.text, self.label, root=ROOT_DIR / 'data')


    def setup(self):
        # called on every GPU
        MAX_VOCAB_SIZE = 25000
        self.train, self.test = datasets.IMDB.splits(self.text, self.label, root=ROOT_DIR / 'data')
        self.train, self.val = self.train.split(random_state=self.random_seed)
        self.text.build_vocab(self.train,
                              max_size=MAX_VOCAB_SIZE,
                              vectors="glove.6B.100d", # Pre trained embedding https://nlp.stanford.edu/projects/glove/
                              unk_init=torch.Tensor.normal_
                              ) # Creates our word embeddings
        self.label.build_vocab(self.train)
        self.dims = len(self.text.vocab)
        self.pad_idx = self.text.vocab.stoi[self.text.pad_token]
        self.unk_idx = self.text.vocab.stoi[self.text.unk_token]

    def train_dataloader(self):
        #transforms = ...
        return data.BucketIterator(self.train, batch_size=self.batch_size, sort_within_batch=True)

    def val_dataloader(self):
        #transforms = ...
        return data.BucketIterator(self.val, batch_size=self.batch_size, sort_within_batch=True)

    def test_dataloader(self):
        #transforms = ...
        return data.BucketIterator(self.test, batch_size=self.batch_size, sort_within_batch=True)


if __name__ == "__main__":
    #hparams =
    imdb = IMDBDataModule()
    imdb.prepare_data()
    imdb.setup()
    imdb.train_dataloader()
