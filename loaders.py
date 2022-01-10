from binary_op_datasets import (
    SumModDataset,
    SubtractModDataset,
    DivideModDataset,
    SquareSumModDataset,
    CubeSumModDataset,
    PermutationsDataset,
)
from small_transformer import TransformerModel
from tools import create_path

import torch


registry = {}


def register(name):
    def store_func(func):
        registry[name] = func
        return func

    return store_func


def load(config, *args):
    local_config = config.copy()
    name = local_config.pop("name")
    if name not in registry:
        raise NotImplementedError
    print(f"loading {name}: {local_config}")
    return registry[name](local_config, *args)


@register("transformer")
def load_transformer(config, vocab_size, output_dim, device):
    nn = TransformerModel(
        model_config=config["nn_arch"],
        vocab_size=vocab_size,
        output_dim=output_dim,
        device=device,
    ).to(device)
    if config["checkpoint_path"] is not None:
        path = create_path(config["checkpoint_path"])
        nn.load_state_dict(torch.load(path, map_location="cpu"))
    return nn


@register("default_opt")
def default_opt(config, nn):
    config = config["opt"]
    optimizer = getattr(torch.optim, config["algo"])(
        params=nn.parameters(),
        lr=config["lr"],
        betas=tuple(config["betas"]),
        weight_decay=config["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: min(1, step / config["warmup_steps"]),
    )
    return optimizer, lr_scheduler


@register("sum_mod_dataset")
def load_sum_mod_dataset(config):
    return SumModDataset(
        high_val=config["high_val"],
        train_size=config["train_size"],
        outlier_size=config["outlier_size"],
    )


@register("subtract_mod_dataset")
def load_subtract_mod_dataset(config):
    return SubtractModDataset(
        high_val=config["high_val"],
        train_size=config["train_size"],
        outlier_size=config["outlier_size"],
    )


@register("divide_mod_dataset")
def load_divide_mod_dataset(config):
    return DivideModDataset(
        high_val=config["high_val"],
        train_size=config["train_size"],
        outlier_size=config["outlier_size"],
    )


@register("square_sum_mod_dataset")
def load_square_sum_mod_dataset(config):
    return SquareSumModDataset(
        high_val=config["high_val"],
        train_size=config["train_size"],
        outlier_size=config["outlier_size"],
    )


@register("cube_sum_mod_dataset")
def load_square_sum_mod_dataset(config):
    return CubeSumModDataset(
        high_val=config["high_val"],
        train_size=config["train_size"],
        outlier_size=config["outlier_size"],
    )


@register("permutations_dataset")
def load_permutations_dataset(config):
    return PermutationsDataset(
        high_val=config["high_val"],
        train_size=config["train_size"],
        outlier_size=config["outlier_size"],
    )
