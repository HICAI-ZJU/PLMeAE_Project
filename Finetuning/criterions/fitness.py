import math
import numpy as np
from dataclasses import dataclass
from omegaconf import II

import torch
from torch import nn
from fairseq import modules, metrics, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion
from fairseq.criterions import register_criterion


@dataclass
class FitnessConfig(FairseqDataclass):
    tpu: int = II("common.tpu")


@register_criterion("fitness_loss", dataclass=FitnessConfig)
class FitnessLoss(FairseqCriterion):
    def __init__(self, cfg: FitnessConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.criterion = nn.MSELoss()

    def forward(self, model, sample, reduce=True):
        tokens, scores = sample
        batch_size = tokens.size(0)

        # with torch.no_grad():
        prev_result = model.forward_encoder(tokens)
        result = model.forward_decoder(prev_result, tokens, readout='mlp')
        # model.forward_encoder.weight.requires_grad = False

        fitness_loss = self.criterion(result.squeeze(), scores.squeeze()) / batch_size

        logging_output = {
            "loss": fitness_loss.data,
            "nsentences": batch_size,
            "ntokens": batch_size * tokens.size(1),
            "result": result.squeeze(1).cpu().clone().detach().numpy().tolist(),
            "scores": scores.cpu().clone().detach().numpy().tolist()
        }
        return fitness_loss, tokens.size(0), logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
