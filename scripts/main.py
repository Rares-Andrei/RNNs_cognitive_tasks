import torch
from rnn_neuro.utils.utils import set_seed, set_device, create_run_folder, \
                                  save_config, setup_loggers
from rnn_neuro.training.trainer import train_model
from rnn_neuro.models import MODEL_REGISTRY
from rnn_neuro.tasks import build_dataset
import neurogym as ngym
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig, verbose=False):
    SEED = cfg.seed
    # Where to save the model
    CHECKPOINT_PATH = f'../experiments/{cfg.task}/runs'
    run_dir = create_run_folder(CHECKPOINT_PATH, cfg)
    save_config(cfg, run_dir)
    log = setup_loggers(run_dir)
    device = set_device()
    # Some operations on a GPU are implemented stochastic for efficiency
    # We operations to be deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(SEED)
    dataset_train, env_train = build_dataset(cfg, seed=cfg.seed, train=True)
    dataset_eval, env_eval = build_dataset(cfg, seed=cfg.seed, train=False)
    # Extract dimensions
    ob_size = env_train.observation_space.shape[0]
    act_size = env_train.action_space.n
    _ = ngym.utils.plot_env(env_train, num_trials=5,
                            fname=f'{run_dir}/{cfg.task}.png')
    model_class = MODEL_REGISTRY[cfg['model']]
    model = model_class(input_size=ob_size,
                        hidden_size=cfg['hidden_size'],
                        output_size=act_size,
                        dt=env_train.dt,
                        tau=cfg['tau'],
                        act_fn=cfg['act_fn'])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg['lr'], weight_decay=1e-5)
    if cfg['class_weights']:
        # Give a small weight to fixate, and equal weights to the rest
        class_weights = torch.tensor([0.05] + [1.0]*(act_size - 1)).to(device)
    else:
        class_weights = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_model(
            model=model,
            optimizer=optimizer,
            batch_iter=dataset_train,
            eval_env=env_eval,
            loss_fn=loss_fn,
            num_epochs=cfg['num_epochs'],
            device=device,
            model_name=cfg['model'],
            run_dir=run_dir,
            log=log,
            seed=SEED,
            periodic_eval=False,
            mode='single' if cfg['task'] != 'yang19' else 'yang19',
            num_trials_perf=300
        )


if __name__ == "__main__":
    main()
