import neurogym as ngym
from neurogym.envs import get_collection
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils import RandomSchedule


# This function is used in the case that we pass kwargs
# to the task in the yang19 collection and some envs
# won't have the passed kwarg (e.g. cohs), so we have
# to ensure that kwarg is not passed to those envs.
def safe_make(task_id: str, kwargs: dict):
    kwargs = dict(kwargs or {})

    while True:
        try:
            return ngym.make(task_id, **kwargs)
        except TypeError as e:
            msg = str(e)

            # Typical error: "__init__() got an unexpected keyword argument 'foo'"
            if "unexpected keyword argument" in msg:
                bad_key = msg.split("'")[1]
                kwargs.pop(bad_key, None)
                continue

            # If it's some other TypeError, re-raise
            raise


def build_dataset(cfg, seed, train=True, verbose=False):
    """
    Dataset builder for single-task and Yang19 suite.

    Args:
        cfg: OmegaConf config with keys: task, batch_size, seq_len, kwargs
        seed: Random seed
        train: If True, build training dataset; if False, eval dataset
        verbose: If True, print relevant information

    Returns:
        dataset: NeuroGym Dataset
        env: The environment object
    """
    if cfg.task == 'yang19':
        # Multi-task: get collection, wrap with scheduler
        tasks = get_collection('yang19')
        envs = [safe_make(task, cfg.get('kwargs')) for task in tasks]
        schedule = RandomSchedule(len(envs))
        env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
        env.seed(seed)
        if verbose:
            # Print environment specifications
            print("Trial timing (in milliseconds):")
            print(env.timing)

            print("\nObservation space structure:")
            print(env.observation_space)

            print("\nAction space structure:")
            print(env.action_space)
            print("Action mapping:")
            print(env.action_space.name)
    else:
        # Single task
        env = safe_make(cfg.task, cfg.get('kwargs'))
        env.seed(seed)
    if verbose:
        # Extract dimensions
        ob_size = env.observation_space.shape[0]
        act_size = env.action_space.n
        # If using yang19:
        # 20 observations for one-hot encoding of
        # the 20 environments, 1 for fixation,
        # 2 for each modality (2 modalities, 4 observations total),
        # for a total of 25 observations
        print(f"Observation size: {ob_size}")
        print(f"Action size: {act_size}")
    # Create dataset
    if train:
        dataset = ngym.Dataset(
            env,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            seed=seed
        )
    else:  # Different seed for evaluation set
        dataset = ngym.Dataset(
            env,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            seed=seed+123
        )

    return dataset, env
