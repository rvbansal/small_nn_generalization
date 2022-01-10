from collections import defaultdict
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def attention_mask(seq_len, device=torch.device("cpu")):
    # Triu tensor of ones with dim (seq_len, seq_len)
    triu_ones = np.triu(np.ones((seq_len, seq_len)), k=1)
    triu_bool = triu_ones == 1
    return torch.tensor(triu_bool).to(device)


def init_weights_xavier(net):
    for param in net.parameters():
        nn.init.xavier_normal_(param)


def init_weights_kaiming(net):
    for param in net.parameters():
        nn.init.kaiming_normal_(param)


def concat_logs(logs):
    concat_logs_output = defaultdict(float)
    counts = defaultdict(float)
    for log in logs:
        for key, value in log.items():
            if isinstance(value, list):
                if key in concat_logs_output:
                    concat_logs_output[key].append(value)
                else:
                    concat_logs_output[key] = [value]
            else:
                metric, count = value
                concat_logs_output[key] += metric * count
                counts[key] += count
    return {key: concat_logs_output[key] / counts[key] for key in counts.keys()}


def create_path(path):
    if path is None:
        return None
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../", path)


def plot_result(results_dict, outcome = 'accuracy'):
    assert outcome in ['accuracy', 'loss']
    fig = plt.figure()
    x = [i + 1 for i in range(len(results_dict))]
    train_outcome = [rd['train'][outcome] for rd in results_dict]
    test_outcome = [rd['test'][outcome] for rd in results_dict]
    plt.plot(x, train_outcome, label="Train")
    plt.plot(x, test_outcome, label="Test")
    plt.xlabel("Optimization Step")
    plt.ylabel("{}".format(outcome.title()))
    plt.legend()
    if outcome == 'accuracy':
        plt.ylim(0, 1)
    return fig
