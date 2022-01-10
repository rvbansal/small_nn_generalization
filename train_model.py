from binary_op_datasets import BinaryOpTorchDataset
from loaders import load
from tools import concat_logs

import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
import pprint
import torch
from torch.utils.data import DataLoader


def train_model(config):
    print("Running config:")
    pprint.pprint(config, width=1)

    # Get configs
    dataset_config = config["dataset"]
    nn_config = config["nn_arch"]
    opt_config = config["opt"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data, nn, optimizer, lr_scheduler
    dataset = load(dataset_config)
    train_data = DataLoader(
        BinaryOpTorchDataset(dataset, type="train"),
        batch_size=dataset_config["batch_size"],
        num_workers=dataset_config["num_workers"],
    )
    test_data = DataLoader(
        BinaryOpTorchDataset(dataset, type="test"),
        batch_size=dataset_config["batch_size"],
        num_workers=dataset_config["num_workers"],
    )
    nn = load(nn_config, dataset.vocab_size, dataset.output_dim, device)
    optimizer, lr_scheduler = load(opt_config, nn)

    # Optimize and cache results
    nn.train()
    step = 1
    out_logs = []
    for x, y in train_data:
        loss, train_cache = nn.eval_loss(x.to(device), y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if step % dataset_config["eval_step"] == 0:
            nn.eval()
            with torch.no_grad():
                test_logs = []
                for i, (test_x, test_y) in enumerate(test_data):
                    _, test_cache = nn.eval_loss(test_x.to(device), test_y.to(device))
                    test_logs.append(test_cache)
                    if i >= dataset.test_data_n / dataset_config["batch_size"]:
                        break
            out_log = {
                "test": concat_logs(test_logs),
                "train": concat_logs([train_cache]),
                "step": step,
            }
            pprint.pprint(out_log, width=1)
            out_logs.append(out_log)
            if out_log["test"]["accuracy"] >= 0.995:
                break
            nn.train()
        step += 1
        if (
            opt_config["opt"]["max_steps"] is not None
            and step >= opt_config["opt"]["max_steps"]
        ):
            break

    return out_logs


@hydra.main(config_path="config", config_name="run_configs")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)
    results = train_model(cfg)
    pickle.dump(results, open("results.p", "wb"))


if __name__ == "__main__":
    main()
