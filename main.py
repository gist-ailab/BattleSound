import os

from data.dataset import SoundLoader

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.metrics.classification import ConfusionMatrix
import neptune
import pytorch_lightning as pl

import argparse
import os
from glob import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms

from torchvision import transforms
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import scipy.io.wavfile
import pandas as pd
from module import model_
import utility.earlystop as earlystop

import warnings
warnings.filterwarnings(action='ignore')

## Argparse
# Save and Resume Options
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp_num', type=int, default=-1)
parser.add_argument('--load_path', type=str, default='no')
parser.add_argument('--save_folder', type=str, default='./model')
parser.add_argument('--gpu', type=str, default='3,5')

# Label Option
parser.add_argument('--base', type=str, default='/HDD1/sung/dataset/dongwoon')
parser.add_argument('--label_type', type=str, default='voice')
parser.add_argument('--label_folder', type=str, default='label_0.5')

# Pre-processing
parser.add_argument('--window_func', type=str, default='hann')
parser.add_argument('--feature_type', type=str, default='MelSpec')

# Learning
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epoch_num', type=int, default=30)
parser.add_argument('--validation', type=int, default=5)
parser.add_argument('--depth', type=int, default=34)

# Model
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--att_type', type=str, default='no')
args = parser.parse_args()

if args.feature_type == 'RawSignal':
    args.model_types = 'simple1d'
else:
    args.model_types = 'simple2d'

args.save_folder = os.path.join(args.save_folder, 'exp_%d' %args.exp_num)

# Make save_folder dir
os.makedirs(args.save_folder, exist_ok=True)

# Main function
class SoundCls(pl.LightningModule):
    def __init__(self):
        super(SoundCls, self).__init__()
        self.network = model_.AttnNet(type=args.model_types, args=args)
        self.param = self.network.parameters()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch, network):
        sound = batch[0]
        label = batch[1]
        pred = network(sound)
        return pred, self.loss(pred, label)

    def training_step(self, batch, batch_idx):
        self.network.train()
        tr_pred, tr_loss = self.forward(batch, self.network)
        return {'loss': tr_loss, 'pred': tr_pred, 'gt':batch[1]}

    def training_epoch_end(self, outputs):
        tr_loss = torch.stack([x['loss'] for x in outputs]).mean()
        log = {'tr_loss' : tr_loss}
        return log

    def validation_step(self, batch, batch_idx):
        self.network.eval()

        val_pred, val_loss = self.forward(batch, self.network)
        return {'loss': val_loss, 'pred':val_pred, 'gt':batch[1]}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        val_pred = torch.cat([x['pred'].max(dim=1)[1] for x in outputs], dim=0)
        val_gt = torch.cat([x['gt'] for x in outputs], dim=0)
        val_acc = np.mean((val_pred == val_gt).cpu().detach().numpy().astype('float')) * 100.

        # metric = ConfusionMatrix(num_classes=2)
        # confusion = metric(val_pred, val_gt)
        # self.logger.experiment.log_text('confusion_matrix', str(confusion))

        log = {'val_loss': val_loss, 'val_acc':val_acc}
        return {'val_loss':val_loss, 'log': log, 'progress_bar':log}

    def configure_optimizers(self):
        return torch.optim.SGD(self.param, lr=args.lr)

    def train_dataloader(self):
        train_dataset = SoundLoader(args, mode='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = SoundLoader(args, mode='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
        return val_loader

# Logger
if __name__=='__main__':
    token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MTQ3MjY2Yy03YmM4LTRkOGYtOWYxYy0zOTk3MWI0ZDY3M2MifQ=='

    neptune_logger = NeptuneLogger(
        api_key=token,
        project_name="sunghoshin/BattleSound",
        close_after_fit=False,
        experiment_name="init",  # Optional,
        params={"max_epochs": args.epoch_num,
                "batch_size": args.batch_size,
                "lr": args.lr}, # Optional,
        tags=["feature_type:%s" %args.feature_type, 'label_type:%s' %args.label_type, 'label_folder:%s'%args.label_folder, \
              'exp_num:%d' %int(args.exp_num), 'new']  # Optional,
    )

    # Checkpoint
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=args.save_folder,
                                                    save_top_k=1,
                                                    monitor='val_loss',
                                                    mode='min')

    # Trainer
    model = SoundCls()
    trainer = pl.Trainer(gpus=args.gpu,
                         distributed_backend='ddp',
                         max_epochs=args.epoch_num,
                         logger=neptune_logger,
                         checkpoint_callback=model_checkpoint)

    trainer.fit(model)

    # neptune_logger.experiment.log_metric('test_accuracy', accuracy)
    # neptune_logger.experiment.log_image('confusion_matrix', fig)
    # neptune_logger.experiment.log_artifact(CHECKPOINTS_DIR) # UPDATE!
    neptune_logger.experiment.stop() # STOP!

