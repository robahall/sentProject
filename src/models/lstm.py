import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import pytorch_lightning as pl

class LSTM(pl.LightningModule):
    def __init__(self, input_dim, hparams, pad_idx):
        super().__init__()
        self.hparams = hparams
        self.embedding = nn.Embedding(input_dim,
                                      self.hparams.embedding_dim,
                                      padding_idx=pad_idx
                                      )
        self.rnn = nn.LSTM(self.hparams.embedding_dim,
                           self.hparams.hidden_dim,
                           num_layers=self.hparams.n_layers,
                           bidirectional=self.hparams.bidirectional,
                           dropout=self.hparams.dropout
                           )

        self.fc = nn.Linear(self.hparams.hidden_dim, self.hparams.output_dim)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()


    def forward(self, x, x_len):
        embedded = self.dropout(self.embedding(x))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, x_len.cpu())
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)

    def training_step(self, train_batch, batch_idx):
        x, x_len = train_batch.text
        y = train_batch.label
        logits = self(x, x_len)
        logits = logits.squeeze(1)
        assert logits.shape == y.shape
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, x_len = val_batch.text
        y = val_batch.label
        logits = self(x, x_len)
        logits = logits.squeeze(1)
        assert logits.shape == y.shape
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)
        self.valid_acc(logits, y)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        x, x_len = test_batch.text
        y = test_batch.label
        logits = self(x, x_len)
        logits = logits.squeeze(1)
        assert logits.shape == y.shape
        loss = self.criterion(logits, y)
        self.log('test_loss', loss)
        self.test_acc(logits, y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    @classmethod
    def add_model_specific_args(cls, parser):
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
            help="output dimension is usually the number of classes, however in \
            the case of only 2 classes the output value is between 0 and 1 and \
            thus can be 1-dimensional, i.e. a single scalar real number.",
        )
        parser.add_argument(
            "--n_layers",
            default=2,
            type=int,
            help="number of layers for an RNN.",
        )
        parser.add_argument(
            "--bidirectional",
            action='store_false',
            help="Process in both directions.",
        )
        parser.add_argument(
            "--dropout",
            default=0.5,
            type=float,
            help="Regularization method that drops (setting to 0) neurons in a layer during a forward pass.\
             Value is a probability between 0-1.",
        )


        return parser