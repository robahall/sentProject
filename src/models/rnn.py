import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import pytorch_lightning as pl

class RNN(pl.LightningModule):
    def __init__(self, input_dim, hparams):
        super().__init__()
        self.hparams = hparams
        self.embedding = nn.Embedding(input_dim, self.hparams.embedding_dim)
        self.rnn = nn.RNN(self.hparams.embedding_dim, self.hparams.hidden_dim)
        self.fc = nn.Linear(self.hparams.hidden_dim, self.hparams.output_dim)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()


    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)
        logits = logits.squeeze(1)
        assert logits.shape == y.shape
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        #print(logits.shape)
        logits = logits.squeeze(1)
        #print(logits.shape)
        #print(y.shape)
        assert logits.shape == y.shape
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)
        self.valid_acc(logits, y)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)
        metrics = {}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer

    @classmethod
    def add_model_specific_args(
        cls, parser):
        """ Parser for Estimator specific arguments/hyperparameters.
        :param parser: argparse.ArgumentParser
        Returns:
            - updated parser
        """
        parser.add_argument(
            "--embedding_dim",
            default=100,
            type=int,
            help="embedding dimension is the size of the dense word vectors. \
            This is usually around 50-250 dimensions, but depends on the size of the vocabulary.",
        )

        parser.add_argument(
            "--hidden_dim",
            default=256,
            type=int,
            help="hidden dimension is the size of the hidden states. \
            This is usually around 100-500 dimensions, but also depends \
            on factors such as on the vocabulary size, the size of the dense \
            vectors and the complexity of the task.",
        )

        parser.add_argument(
            "--output_dim",
            default=1,
            type=int,
            help="utput dimension is usually the number of classes, however in \
            the case of only 2 classes the output value is between 0 and 1 and \
            thus can be 1-dimensional, i.e. a single scalar real number.",
        )

        return parser




