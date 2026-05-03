import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


@torch.no_grad()
def task_variance(model, env,
                  num_trials, device):
    model.eval()
    print(f'Model sigma rec: {model.sigma_rec}')
    if not hasattr(env, "envs"):
        raise ValueError('expects env to have attribute env.envs')
    env_ids = [e_i.spec.id for e_i in env.envs]

    # Store per-task trial activities
    activity_by_env = {env_id: [] for env_id in env_ids}
    acc_by_env = {env_id: 0.0 for env_id in env_ids}
    mask_by_env = {env_id: [] for env_id in env_ids}  # Mask to exclude fixation timesteps
    fix_ix = 0  # Index of fixation input
    for env_idx, e_i in enumerate(env.envs):
        env_id = e_i.spec.id
        total_correct = 0
        for trial_idx in range(num_trials):
            trial = e_i.new_trial()
            ob, gt = e_i.ob, e_i.gt
            T = ob.shape[0]
            env_one_hot = np.zeros((T, len(env.envs)), dtype=np.float32)
            env_one_hot[:, env_idx] = 1.0
            ob_with_env = np.concatenate([ob, env_one_hot], axis=1)
            ob_with_env = ob_with_env[:, np.newaxis, :]  # (T, 1, D)
            
            inputs = torch.from_numpy(ob_with_env).float().to(device)
            outputs, rnn_activity = model(inputs)
            pred_actions = torch.argmax(outputs, dim=2)
            decision_idx = T - 1
            is_correct = bool(pred_actions[decision_idx].item() == gt[decision_idx])
            total_correct += int(is_correct)

            # (T, 1, N) -> (T, N)
            act = rnn_activity[:, 0, :].detach().cpu().numpy()
            activity_by_env[env_id].append(act)
            mask = (ob[:, fix_ix] == 0)  # shape (T,), fixation=0 during stimulus, delay and decision periods (=1 during fixation)
            mask_by_env[env_id].append(mask)
        acc_by_env[env_id] = total_correct / float(num_trials)

    # Compute task variance per env: (n_envs, n_units)
    tv_list = []
    for env_id in env_ids:
        trials = activity_by_env[env_id]  # list of (T, N)
        masks = mask_by_env[env_id]  # list of (T,)
        # Handle variable length by truncating to min T
        T_min = min(a.shape[0] for a in trials)
        trials = [a[:T_min] for a in trials]  # each (T_min, N)
        masks = [m[:T_min] for m in masks]
        R = np.stack(trials, axis=0)  # (J, T, N), J = number of trials, T = number of time steps, N = number of hidden units
        stacked = np.stack(masks, axis=0)  # shape (J, T_min)
        analysis_mask = np.logical_and.reduce(stacked, axis=0)  # shape (T_min,), checks if a time t is valid in all trials
        R = R[:, analysis_mask, :]  # Eliminate time steps which are not valid, shape (J, T_filtered, N)
        var = np.var(R, axis=0, ddof=0)  # Variance across trials, shape (T_filtered, N), A[t, n] = variance of unit n at timestep t across the trials
        tv = var.mean(axis=0)  # Average over timesteps, shape (N,): one number per unit = task variance per unit
        tv_list.append(tv)
    tv_by_task = np.stack(tv_list, axis=0)  # (n_tasks, n_units)
    return tv_by_task, env_ids, acc_by_env


def tsne_task_variance(tv_by_task, k=8):
    X = tv_by_task.T  # shape: (n_units, n_tasks)
    active = X.sum(axis=1) > 1e-3  # Only units above a threshold across tasks: (n_active_units, n_tasks)
    X_active = X[active]
    Xn = X_active / (X_active.max(axis=1, keepdims=True) + 1e-8)  # (n_active_units, n_tasks)
    Z = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=100,
        init="pca",
        random_state=0
    ).fit_transform(Xn)  # Shape (n_active_units, n_components)
    labels = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(Xn)  # For each unit, give it a label that tells which cluster it belong to
    score = silhouette_score(Xn, labels)
    print('Silhoutte score:', score)
    return Z, labels


def plot_tsne_task_variance(tsne, labels, run_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, s=100, cmap="tab10")
    plt.title("t-SNE of units using task-variance fingerprints")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(f'{run_dir}/tsne.png')


def plot_task_variance_unit(tv_by_task, tasks, run_dir, unit_idx=10):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(tasks)), tv_by_task[:, unit_idx], marker="o")
    plt.title(f"Task variance of unit {unit_idx}")
    plt.xlabel("Task")
    plt.ylabel("TV")
    plt.xticks(np.arange(len(tasks)), tasks, rotation=90)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/TV_of_unit_{unit_idx}.png")
    plt.close()


def yang_heatmap_data(tv_by_task, k=8, active_thresh=1e-3, eps=1e-8):
    """
    Returns:
      H_sorted: (n_tasks, n_active_units) heatmap matrix normalized per unit by its max across tasks
      labels_sorted: (n_active_units,) cluster labels in the sorted unit order
      order: indices into the active-unit axis giving the sorted order
      active_mask: boolean mask over original units (n_units,) for which units are active
    """
    tv = np.asarray(tv_by_task)  # (n_tasks, n_units)

    # active units criterion: sum over tasks > threshold
    active_mask = tv.sum(axis=0) > active_thresh
    tv_active = tv[:, active_mask]  # (n_tasks, n_active_units)

    # normalize each unit by its peak across tasks (max over tasks for that unit/column)
    peak = tv_active.max(axis=0, keepdims=True)  # (1, n_active_units)
    H = tv_active / (peak + eps)  # (n_tasks, n_active_units)
    # cluster units using their normalized task-variance fingerprints
    # units are datapoints: shape (n_active_units, n_tasks)
    X = H.T  # (n_active_units, n_tasks)
    labels = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(X)
    # sort units by (cluster id)
    order = np.argsort(labels)  # shape (n_units,)
    H_sorted = H[:, order]  # The units are in the order of labels, for each task
    labels_sorted = labels[order]

    return H_sorted, labels_sorted, order, active_mask


def plot_yang_heatmap(H_sorted, labels_sorted, run_dir,
                      env_ids=None, fname="yang_heatmap.png"):
    """
    H_sorted: (n_tasks, n_units_sorted) normalized [0,1]
    labels_sorted: (n_units_sorted,)
    env_ids: list of task names for y-axis (optional)
    """
    n_tasks, n_units = H_sorted.shape
    k = int(labels_sorted.max()) + 1

    # Categorical colormap for the cluster strip
    cmap_strip = plt.get_cmap("tab10", k) if k <= 10 else plt.get_cmap("tab20", k)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20, 1], hspace=0.08)

    # main heatmap
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(H_sorted, aspect="auto", interpolation="nearest")  # Default colormap

    ax.set_title("Task variance heatmap (per-unit peak-normalized), units sorted by k-means cluster")
    ax.set_xlabel("Units (sorted by cluster)")
    ax.set_ylabel("Tasks")

    if env_ids is not None:
        ax.set_yticks(np.arange(n_tasks))
        ax.set_yticklabels(env_ids)
    else:
        ax.set_yticks([])

    ax.set_xticks([])

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Normalized task variance (unit-wise / max over tasks)")

    # cluster membership strip (bottom)
    ax2 = fig.add_subplot(gs[1, 0])
    strip = labels_sorted[np.newaxis, :]  # shape (1, n_units)
    ax2.imshow(strip, aspect="auto", interpolation="nearest", cmap=cmap_strip, vmin=0, vmax=k-1)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlabel("Cluster membership")

    plt.savefig(f"{run_dir}/{fname}", dpi=200, bbox_inches="tight")
    plt.close(fig)
