import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import Trainer, TrainingArguments




def generate_strategy(num_layers, target_pruning_ratio):
    """
    Generate a random vector of per-layer pruning ratios {β_i}
    such that the average ≈ target_pruning_ratio
    """
    v = np.random.uniform(0.0, 1.0, size=num_layers)
    v = target_pruning_ratio * v / v.mean()
    v = np.clip(v, 0.0, 0.98)  # keep safe
    return v

def magnitude_prune_tensor(weight, prune_ratio):
    """
    weight: torch.nn.Parameter
    prune_ratio: float between 0 and 1
    """
    if prune_ratio == 0:
        return weight.clone()

    # Flatten weight for threshold calculation
    w_flat = weight.detach().cpu().numpy().flatten()
    k = int(prune_ratio * w_flat.shape[0])

    # Handle edge cases for k
    if k == 0 and prune_ratio != 0: # If k is 0 but prune_ratio is not, it means no pruning is effectively applied.
        return weight.clone()
    elif k >= w_flat.shape[0]: # If k is greater than or equal to total elements, prune all.
        return torch.zeros_like(weight)

    # Find threshold using numpy on the detached numpy array
    threshold = np.partition(np.abs(w_flat), k)[k]

    # Create mask using torch operations and convert threshold to a torch tensor
    # Ensure threshold is on the same device as the weight tensor
    mask = (torch.abs(weight) > torch.tensor(threshold, device=weight.device)).float()

    return weight * mask

def apply_pruning(model, strategy):
    pruned_model = deepcopy(model)

    encoder_layers = pruned_model.bert.encoder.layer


    for i, layer_prune_ratio in enumerate(strategy): # Renamed 'B' to 'layer_prune_ratio'
        print(f"Applying pruning ratio {layer_prune_ratio} to layer {i}")
        layer = encoder_layers[i]

        # prune attention query, key, value, output dense, and FFN
        modules = [
            layer.attention.self.query, # Attention Q
            layer.attention.self.key, # Attention K
            layer.attention.self.value, # Attention V
            layer.attention.output.dense, # FFN
            layer.intermediate.dense, # FFN
            layer.output.dense
        ]

        for module in modules:
            module.weight.data = magnitude_prune_tensor(module.weight.data, layer_prune_ratio)
            if module.bias is not None:
                module.bias.data = magnitude_prune_tensor(module.bias.data, layer_prune_ratio)

    return pruned_model

def finetune(pruned_model, train_dataset, eval_dataset, args):
    trainer = Trainer(
        model=pruned_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    return trainer
