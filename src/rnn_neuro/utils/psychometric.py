import numpy as np
import matplotlib.pyplot as plt


def dm1_trial_to_xy(trial, pred_action: int):
    """
    trial: dict returned by env.new_trial()
    pred_action: int model prediction at decision time (0/1/2)

    Returns:
        x: float = coh1_mod1 - coh2_mod1
        y: int in {0,1} = 1 if chose Stim1, else 0

    Raises:
        ValueError if action is fixation or theta1 not in {0, pi}.
    """
    if pred_action == 0:
        raise ValueError("Fixation/no-response")

    x = float(trial["coh1_mod1"] - trial["coh2_mod1"])

    theta1 = float(trial["theta1"])

    # dim_ring=2 mapping:
    # action 1: theta=0
    # action 2: theta=pi
    if np.isclose(theta1, 0.0):
        stim1_action = 1
    elif np.isclose(theta1, np.pi):
        stim1_action = 2
    else:
        raise ValueError(f"Unexpected theta1={theta1} for dim_ring=2")

    y = 1 if pred_action == stim1_action else 0
    return x, y


def build_psychometric_from_xy(xy_dict, n_bins=9, min_points=2):
    """
    xy_dict: {"xs": [...], "ys": [...]}
    """
    xs, ys = xy_dict["xs"], xy_dict["ys"]
    if len(xs) < min_points:
        return None
    centers, p, counts = bin_psychometric(xs, ys, n_bins=n_bins)
    return {"bin_centers": centers, "p_choose_stim1": p, "counts": counts}


def collect_dm1_samples(dm1_by_period, trial, pred_action: int):
    """Accumulate (x,y) samples grouped by stim_period into dm1_by_period."""
    stim_period = int(trial["stim_period"])
    try:
        x, y = dm1_trial_to_xy(trial, pred_action)  # May raise ValueError (fixation/no-response)
    except ValueError:
        return  # Skip

    if stim_period not in dm1_by_period:
        dm1_by_period[stim_period] = {"xs": [], "ys": []}
    dm1_by_period[stim_period]["xs"].append(x)
    dm1_by_period[stim_period]["ys"].append(y)


def build_dm1_psychometric_curves(dm1_by_period, n_bins=9, min_points=10):
    """Convert collected samples into binned psychometric curves."""
    curves = {}
    for stim_period, d in dm1_by_period.items():
        xs, ys = d["xs"], d["ys"]
        if len(xs) < min_points:
            continue
        centers, p, counts = bin_psychometric(xs, ys, n_bins=n_bins)
        curves[stim_period] = {
            "bin_centers": centers,
            "p_choose_stim1": p,
            "counts": counts,
        }
        print(f'curves: {curves}')
    return curves


def bin_psychometric(xs, ys, n_bins=9):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    edges = np.linspace(xs.min(), xs.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    p = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i < n_bins - 1:
            m = (xs >= edges[i]) & (xs < edges[i + 1])  # Creates a boolean mask m the same shape as xs, where m[j] is True if trial j falls inside the bin
            print(f'm: {m}')
        else:
            m = (xs >= edges[i]) & (xs <= edges[i + 1])

        counts[i] = int(m.sum())  # Counts number of trials in that bin
        if counts[i] > 0:
            p[i] = float(ys[m].mean())  # Compute the estimated probability for that bin

    return centers, p, counts


def plot_psychometric_by_period(curves, title="Psychometric curves (dm1)"):
    """
    curves: dict stim_period: {"bin_centers", "p_choose_stim1", "counts"}
    """
    plt.figure()

    for stim_period in sorted(curves.keys()):
        c = curves[stim_period]
        x = np.asarray(c["bin_centers"], dtype=float)
        y = np.asarray(c["p_choose_stim1"], dtype=float)

        plt.plot(x, y, marker="o", label=f"{stim_period} ms")

    plt.ylim(-0.05, 1.05)
    plt.xlabel("Stimulus 1 - Stimulus 2  (coh1_mod1 - coh2_mod1)")
    plt.ylabel("P(choose Stimulus 1)")
    plt.title(title)
    plt.grid(True)
    plt.legend(title="Stimulus time")
    plt.show()
