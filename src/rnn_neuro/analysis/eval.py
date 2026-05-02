import torch
import numpy as np
from rnn_neuro.utils.psychometric import collect_dm1_samples, \
                                         build_dm1_psychometric_curves, \
                                         plot_psychometric_by_period


@torch.no_grad()
def rmse_per_timestep(model, x, y):
    """
    Computes RMSE per timestep, average across batches.
    Applied to the XOR problem.

    Args:
        model: Trained model
        x: Shape (seq_length-1, batch_size, input_size)
        y: Shape (seq_length-1, batch_size, input_size)

    Returns:
        RMSE: Value of error
    """
    model.eval()
    pred, _ = model(x)
    se = (pred - y) ** 2
    se_mean = se.mean(axis=1).squeeze()  # Average across batches
    rmse = torch.sqrt(se_mean)
    return rmse


@torch.no_grad()
def average_rms_per_timstep(model, batch_iter, trials: int):
    """
    Computes average RMSE per timestep across trials.
    Applied to the XOR problem.

    Args:
        model: Trained model
        batch_iter: Iterator object to create input, label pairs
        trials: Number of trials across which to average

    Returns:
        average_rmse: Average error across trials

    """
    assert isinstance(trials, int)
    model.eval()
    average_rmse = 0
    for _ in range(trials):
        batch = next(batch_iter)
        x, y = batch
        acc_timestep = rmse_per_timestep(model, x, y)
        average_rmse += acc_timestep
    average_rmse = average_rmse / trials
    return average_rmse


@torch.no_grad()
def evaluate_model(model, num_trials, env,
                   device, seed, mode="single",
                   verbose=False, record_activity=True,
                   record_yang_data=False, task_to_eval=None,
                   plot_psychometric=False, psych_n_bins=9,
                   psych_min_points=10):
    """"
    Evaluates the trained model on a number of trials
    and provide the average performance.

    Args:
        model: Trained model
        num_trials: Number of trials to evaluate
        env: Neurogym environment (Dataset.env)
        device: Device to run on
        seed: For reproducibility
        mode="single": one NeuroGym env (env.new_trial())
        mode="yang19": env is a collection with env.envs
        verbose: True for printing information
        task_to_eval: If another task is used to evaluate a model than the one (or the yang19 suite) it was trained on
        plot_psychometric: True if we want to plot psychometric curve
        psych_n_bins: How many bins the psychometric curve should have
        psych_min_points: Min points for each bin in the psychometric curve
    Return:
        results: dict with keys:
        - "perf": float (single) OR dict(env_id -> accuracy) (yang19)
        - "activity": dict (trial_idx -> activity) for single if record_activity else {}
        - "trial_infos": dict (trial_idx -> trial_info) for single else {}
        - "yang_data": dict(env_id -> data) if record_yang_data else {}
    """
    model.eval()
    results = {
        "perf": None,
        "activity": {},
        "trial_infos": {},
        "yang_data": {},
    }
    if seed is not None and hasattr(env, "seed") and mode == "single":
        env.seed(seed)
    if mode == "single":
        perf = 0.0
        activity_dict = {}
        trial_infos = {}
        # For psychometric curves in case of yang19.dm1-v0 task
        dm1_by_period = {}  # stim_period: {"xs": [], "ys": []}
        for i in range(num_trials):
            trial_info = env.new_trial()
            ob, gt = env.ob, env.gt
            # (time, 1, obs_dim) - new batch axis
            ob = ob[:, np.newaxis, :]
            inputs = torch.from_numpy(ob).float().to(device)
            if verbose:
                print("observation shape:", ob.shape)
                print("ground truth shape:", gt.shape)
                print("inputs shape:", inputs.shape)
            outputs, rnn_activity = model(inputs)
            pred_actions = outputs.cpu().detach().numpy()[:, 0, :]  # Shape (time, num_actions) because the batch size is 1
            pred_actions = np.argmax(pred_actions, axis=-1)  # For each timestep, pick the action index with the largest score: pred_actions[t] = chosen_action_index
            decision_ts = np.where(gt != 0)[0]
            t = decision_ts[-1]  # last timestep that expects a choice
            correct = (pred_actions[t] == gt[t])
            perf += float(correct)
            # Calcuate psychometric curve for dm1
            pred = int(pred_actions[t].item())
            if env.spec.id == "yang19.dm1-v0" and plot_psychometric:
                collect_dm1_samples(dm1_by_period, trial_info, pred)
            if record_activity:
                # rnn_activity: (time, batch=1, rnn_units) -> (time, rnn_units)
                activity_dict[i] = rnn_activity[:, 0, :].detach().cpu().numpy()
            trial_infos[i] = trial_info  # trial_info is a dictionary
            trial_infos[i].update({'correct': correct})
            if verbose:
                print("outputs shape:", outputs.shape)
                print("predicted actions shape:", pred_actions.shape)
                print("predicted actions:", pred_actions)
                print("ground truth:", gt)
                print("decision t:", t, "correct:", correct)
        if env.spec.id == "yang19.dm1-v0" and plot_psychometric:
            curves = build_dm1_psychometric_curves(dm1_by_period,
                                                   n_bins=psych_n_bins,
                                                   min_points=psych_min_points)
            if plot_psychometric and curves:
                plot_psychometric_by_period(curves)
            results["psychometric_dm1"] = curves
        results["perf"] = perf / num_trials
        results["activity"] = activity_dict
        results["trial_infos"] = trial_infos
        return results

    if mode == "yang19":
        # Expect env.envs
        if not hasattr(env, "envs"):
            raise ValueError('mode="yang19" expects env to have attribute env.envs')
        if task_to_eval:  # If the task to eval is different than the one it was trained on
            env_ids = task_to_eval
            acc_by_env = {env_ids: 0.0}
        else:
            env_ids = [e_i.spec.id for e_i in env.envs]
            acc_by_env = {env_id: 0.0 for env_id in env_ids}
        yang_data = {}
        if record_yang_data:
            for env_id in env_ids:
                yang_data[env_id] = {"action": [], "gt": [], "trial": []}
        # For psychometric curves in case of yang19.dm1-v0 task
        dm1_by_period = {}  # stim_period: {"xs": [], "ys": []}
        for env_idx, e_i in enumerate(env.envs):
            total_correct = 0
            env_id = e_i.spec.id
            if env_id not in env_ids:  # We only evaluate on the required tasks (e.g. case that task_to_eval is not False)
                continue
            for trial_idx in range(num_trials):
                trial = e_i.new_trial()
                ob, gt = e_i.ob, e_i.gt
                T = ob.shape[0]  # Length of trial
                # one-hot env context: (T, n_envs)
                env_one_hot = np.zeros((T, len(env.envs)))
                env_one_hot[:, env_idx] = 1.0
                # concat: (T, obs_dim + n_envs)
                # Concatenate original observation with one-hot encoding
                ob_with_env = np.concatenate([ob, env_one_hot], axis=1)
                ob_with_env = ob_with_env[:, np.newaxis, :]  # (T, 1, D) Add batch dimension
                inputs = torch.from_numpy(ob_with_env).float().to(device)
                outputs, _ = model(inputs)
                pred_actions = torch.argmax(outputs, dim=2)
                decision_idx = T - 1  # Length of trial minus 1
                is_correct = bool(pred_actions[decision_idx] == gt[decision_idx])
                total_correct += int(is_correct)
                # Calcuate psychometric curve for dm1
                pred = int(pred_actions[decision_idx].item())
                if env_id == "yang19.dm1-v0" and plot_psychometric:
                    collect_dm1_samples(dm1_by_period, trial, pred)
                if record_yang_data:
                    yang_data[env_id]["trial"].append(trial)
                    yang_data[env_id]["gt"].append(int(gt[decision_idx]))
                    yang_data[env_id]["action"].append(int(pred_actions[decision_idx]))
            acc_by_env[env_id] = total_correct / num_trials
        if "yang19.dm1-v0" in env_ids and plot_psychometric:
            curves = build_dm1_psychometric_curves(dm1_by_period,
                                                   n_bins=psych_n_bins,
                                                   min_points=psych_min_points)
            print(f'Curves: {curves}')
            if plot_psychometric and curves:
                plot_psychometric_by_period(curves)
            results["psychometric_dm1_by_period"] = curves
        results["perf"] = acc_by_env
        results["yang_data"] = yang_data if record_yang_data else {}
        return results
    raise ValueError(f"Unknown mode: {mode!r}. Use 'single' or 'yang19'.")
