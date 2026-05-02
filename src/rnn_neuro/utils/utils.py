import numpy as np
import torch
import yaml
import os
import json
from omegaconf import OmegaConf
import datetime
import logging
import random
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def save_config(cfg, run_dir):  # Save config file explicitly
    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))


def create_run_folder(base_dir: str, cfg):
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def setup_loggers(run_dir):
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)  # DEBUG level captures all messages

    # Create handlers for each level we want: info, warning, error
    info_handler = logging.FileHandler(os.path.join(run_dir, 'info.log'))
    warning_handler = logging.FileHandler(os.path.join(run_dir, 'warning.log'))
    error_handler = logging.FileHandler(os.path.join(run_dir, 'error.log'))

    # Set levels for handlers so they can capture the specific information
    info_handler.setLevel(logging.INFO)
    warning_handler.setLevel(logging.WARNING)
    error_handler.setLevel(logging.ERROR)

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    info_handler.setFormatter(formatter)
    warning_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    # Add handlers to the logger
    log.addHandler(info_handler)
    log.addHandler(warning_handler)
    log.addHandler(error_handler)
    return log


def _json_sanitize(x):
    # numpy scalar -> python scalar
    if isinstance(x, np.generic):
        return x.item()
    # numpy array -> list
    if isinstance(x, np.ndarray):
        return x.tolist()
    # torch tensors -> list
    if torch.is_tensor(x):
        x = x.detach().cpu()
        return x.item() if x.numel() == 1 else x.tolist()
    # recurse containers
    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]
    return x


def _get_config_file(model_path, model_name):
    return os.path.join(model_path, model_name + ".config")


def _get_model_file(model_path, model_name):
    return os.path.join(model_path, model_name + ".tar")


def _get_result_file(model_path, model_name):
    return os.path.join(model_path, model_name + "_results.json")


def load_model(model_path, model_name, net_class):
    config_file, model_file = _get_config_file(model_path, model_name), \
        _get_model_file(model_path, model_name)
    assert os.path.isfile(config_file), \
        f"Could not find the config file \"{config_file}\". \
        Are you sure this is the correct path and you have \
        your model config stored here?"
    assert os.path.isfile(model_file), \
        f"Could not find the modelfile \"{model_file}\". \
        Are you sure this is the correct path and you have \
        your model stored here?"
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    net = net_class(**config_dict)
    net.load_state_dict(torch.load(model_file))
    return net


def save_model(model, model_path, model_name):
    config_dict = _json_sanitize(model.config)
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), \
        _get_model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    torch.save(model.state_dict(), model_file)


def save_input_output(input, output, run_dir):
    """
    Saves the input and output of the model during training for later analysis.
    Args:
        input: Input to the model of shape (num_epochs, seq_len, batch_size, input_size)
        output: Output of the model of shape (num_epochs, seq_len, batch_size, output_size)
    """
    np.save(os.path.join(run_dir, "input.npy"), input)
    np.save(os.path.join(run_dir, "output.npy"), output)


def plot_input_output(run_dir, epoch=-1, sequence_index=6):
    input = np.load(os.path.join(run_dir, "input.npy"))
    output = np.load(os.path.join(run_dir, "output.npy"))
    print(f'Input shape: {input.shape}')
    print(f'Output shape: {output.shape}')
    output_softmax = torch.nn.functional.softmax(torch.from_numpy(output), dim=-1).numpy()
    plt.figure()
    for o in range(output.shape[-1]):  # outputs
        plt.plot(output_softmax[epoch, :, sequence_index, o], label=f"out{o}")
        plt.plot(input[epoch, :, sequence_index, o], label=f"in{o}", linestyle="dashed")
    plt.legend()
    plt.savefig(os.path.join(run_dir, f"input_output_epoch{epoch}_seq{sequence_index}.png"))


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device():
    device = torch.device("cpu") if not torch.cuda.is_available() \
            else torch.device("cuda:0")
    return device


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def log_param_histograms(writer, step, model):
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.detach().cpu(), global_step=step)


# Only if debuggning gradients is needed
def log_grad_histograms(writer, step, model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            writer.add_histogram(f"{name}.grad", p.grad.detach().cpu(), global_step=step)


def plot_pca(run_dir, activity_dict, num_trials, trial_infos, n_components=2):
    activity = np.concatenate(list(activity_dict[i] for i in range(num_trials)), axis=0)
    print('Shape of the neural activity: (Time points, Neurons): ', activity.shape)  # All time points across all trials

    pca = PCA(n_components=n_components)
    pca.fit(activity)  # activity (Time points, Neurons)
    activity_pc = pca.transform(activity)  # transform to low-dimension
    print('Shape of the projected activity: (Time points, PCs): ', activity_pc.shape)
    # Plot all trials in ax1, plot fewer trials in ax2
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6, 3))

    for i in range(num_trials):
        # Transform and plot each trial
        activity_pc = pca.transform(activity_dict[i])  # (Time points, PCs) within a single trial
        trial = trial_infos[i]
        if "context" in trial:
            key = (trial["context"], trial["ground_truth"])
            color_map = {
                (0, 1): "blue",
                (0, 2): "green",
                (1, 1): "orange",
                (1, 2): "purple",
            }
            color = color_map.get(key, "gray")  # fallback if unexpected values
        else:
            gt = trial.get("ground_truth", None)
            if gt == 0:
                color = "red"
            elif gt == 1:
                color = "blue"
            elif gt == 2:
                color = "green"
            else:
                color = "gray"
        _ = ax1.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)
        if i < 3:
            _ = ax2.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)

        # Plot the beginning of a trial with a special symbol
        _ = ax1.plot(activity_pc[0, 0], activity_pc[0, 1], '^', color='black')

        ax1.set_title('{:d} Trials'.format(num_trials))
        ax2.set_title('{:d} Trials'.format(3))
        ax1.set_xlabel('PC 1')
        ax1.set_ylabel('PC 2')
    plt.savefig(os.path.join(run_dir, "pca.png"))
