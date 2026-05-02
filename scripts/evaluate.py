from rnn_neuro.models import MODEL_REGISTRY
from rnn_neuro.analysis.eval import evaluate_model
from rnn_neuro.utils.utils import set_seed, load_model, \
                                  set_device, plot_pca, \
                                  plot_input_output
from rnn_neuro.utils.metrics import task_variance, tsne_task_variance, \
                                    plot_tsne_task_variance, \
                                    yang_heatmap_data, \
                                    plot_yang_heatmap, \
                                    plot_task_variance_unit

from rnn_neuro.tasks import build_dataset
import torch
import hydra
from omegaconf import DictConfig
import os
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig, num_trials=1000):
    SEED = cfg.seed
    set_seed(SEED)
    # Some operations on a GPU are implemented stochastic for efficiency
    # We operations to be deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run_dir = f'../experiments/{cfg.task}/runs/{cfg.load_model}'  # Where to find the model file, task that the model was trained on
    run_cfg_path = os.path.join(run_dir, "config.yaml")
    run_cfg = OmegaConf.load(run_cfg_path)
    OmegaConf.set_struct(cfg, False)  # Allow new keys anywhere
    cfg = OmegaConf.merge(cfg, run_cfg)  # Overwrite cfg with run_cfg such that the same model parameters are used for evaluation
    # Something with overwriting config
    device = set_device()
    dataset_eval, env_eval = build_dataset(cfg, seed=cfg.seed, train=False)
    # Note: environments and schedulers are seeded above before creating the Dataset.
    model_class = MODEL_REGISTRY[cfg["model"]]
    model = load_model(model_path=run_dir,
                       model_name=cfg.model,
                       net_class=model_class).to(device)
    res = evaluate_model(
        model=model,
        num_trials=num_trials,
        env=env_eval,
        device=device,
        seed=SEED,
        mode="yang19" if cfg["task"] == "yang19" else "single",  # If the model was trained on yang19 or otherwise a single task
        record_activity=(cfg["task"] != "yang19"),
        record_yang_data=False,  # flip to True for having action/gt/trial saved per env
        task_to_eval=cfg['task_to_eval'],
        plot_psychometric=True
    )
    if cfg["task"] == "yang19":  # Model was trained on yang19
        # res["perf"] is a dict: {env_id: accuracy}
        print("Average performance across all environments:")
        for env_id, acc in res["perf"].items():
            print(f"{env_id}: {acc:.4f}")
        model.sigma_rec = 0
        tv_by_task, env_ids, acc_by_env = task_variance(model=model,
                                                        env=env_eval,
                                                        num_trials=200,
                                                        device=device)
        tsne, labels = tsne_task_variance(tv_by_task=tv_by_task, k=10)
        plot_tsne_task_variance(tsne=tsne, labels=labels, run_dir=run_dir)
        H_sorted, labels_sorted, order, active_mask = yang_heatmap_data(tv_by_task, k=10)
        plot_yang_heatmap(H_sorted, labels_sorted, env_ids=env_ids, run_dir=run_dir)
        plot_task_variance_unit(tv_by_task=tv_by_task, tasks=env_ids,
                                run_dir=run_dir, unit_idx=104)
    else:
        performance = res["perf"]  # float
        activity_dict = res["activity"]  # dict[trial_idx]: (time, units)
        trial_infos = res["trial_infos"]  # dict[trial_idx]: info dict

        print(f"Average performance in {num_trials} trials: {performance:.2f}")
        plot_pca(run_dir, activity_dict, num_trials, trial_infos)
        plot_input_output(run_dir, epoch=-1, sequence_index=6)


if __name__ == '__main__':
    main()
