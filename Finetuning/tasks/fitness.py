import os
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Sequence, Union, List, Tuple, Any
from omegaconf import MISSING
from pathlib import Path
import torch
from fairseq import utils
from fairseq.dataclass import FairseqDataclass
from fairseq.data import FairseqDataset
from fairseq.tasks import FairseqTask, register_task

from ..utils import Alphabet, Converter

logger = logging.getLogger(__name__)


class FitnessDataset(FairseqDataset):
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 alphabet):
        if split == 'valid':
            split = 'test'
        if split == 'train':
            self.data_path = os.path.join(data_path, 'path_to_train_data.csv')
        else:
            self.data_path = os.path.join(data_path, 'path_to_test_data.csv')
        self.data = pd.read_csv(self.data_path)
        self.converter = Converter(alphabet)
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data.loc[index]
        sequence, label = item['seq'], item['fitness']
        return sequence, label

    def collater(self, raw_batch: Sequence[Dict]):
        sequences, labels = zip(*raw_batch)
        tokens = self.converter(sequences)
        return tokens, torch.tensor(labels, dtype=torch.float) * 1000

    def size(self, index):
        return len(self.data.loc[index]['seq'])

    def num_tokens(self, index):
        return len(self.data.loc[index]['seq'])

    def num_tokens_vec(self, indices):
        return np.array([self.num_tokens(index) for index in indices])


@dataclass
class FitnessConfig(FairseqDataclass):
    data: str = field(default=MISSING)


@register_task("fitness", dataclass=FitnessConfig)
class FitnessTask(FairseqTask):
    cfg: FitnessConfig
    """Task for training masked language models (e.g., BERT, RoBERTa)"""

    def __init__(self, cfg: FitnessConfig, alphabet):
        super().__init__(cfg)
        self.alphabet = alphabet
        self.max_length = 1024

    @classmethod
    def setup_task(cls, cfg: FitnessConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        alphabet = Alphabet.build_alphabet()
        logger.info(f"Alphabet: {len(alphabet)} types")
        return cls(cfg, alphabet)

    def load_dataset(self, split: str, epoch=1, combine=False, **kwargs):
        self.datasets[split] = FitnessDataset(self.cfg.data, split, self.alphabet)

    @property
    def source_dictionary(self):
        return self.alphabet

    @property
    def target_dictionary(self):
        return self.alphabet
