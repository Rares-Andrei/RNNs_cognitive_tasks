This repo makes it simple to train RNNs on [Neurogym](https://neurogym.github.io/neurogym/latest/) cognitive tasks, following [Yang et al. 2019](https://www.nature.com/articles/s41593-018-0310-2). Specifically, I reproduced the main findings of the paper using PyTorch + Hydra.

## Contents

Two model types (a vanilla RNN and a continuous-time RNN), a training loop, and some evaluation/analysis code. Tasks come from [Neurogym](https://github.com/neurogym/neurogym). Configs go through Hydra so you can override anything from the command line, and runs get logged to TensorBoard.

## Layout

```
yang_2019/
├── src/rnn_neuro/
│   ├── models/         # base_rnn.py, ctrnn.py
│   ├── training/       # trainer.py
│   ├── analysis/       # eval.py
│   └── utils/          # seeding, logging, checkpointing
├── scripts/
│   ├── main.py         # training entry point
│   └── evaluate.py     # evaluation
├── config/config.yaml  # default config
├── experiments/        # one folder per task, runs go inside
├── notebooks/          # exploratory
├── tests/
├── yang_cpu.yml        # conda env (CPU)
└── yang_gpu.yml        # conda env (GPU)
```

## Setup

```bash
conda env create -f yang_cpu.yml  # or yang_gpu.yml on a GPU box
conda activate yang_2019
pip install -e .
```

Quick sanity check:

```python
from rnn_neuro.models import MODEL_REGISTRY
print(MODEL_REGISTRY.keys())  # dict_keys(['base_rnn', 'ctrnn'])
```

## Running

Training:

```bash
cd scripts/
python main.py task=GoNogo-v0 model=base_rnn
```

Override on the command line:

```bash
python main.py task=ContextDecisionMaking-v0 model=ctrnn hidden_size=256 num_epochs=2000 lr=0.001 tau=1000
```

Evaluation needs a run timestamp (the folder name under `experiments/{task}/runs/`):

```bash
python evaluate.py task=GoNogo-v0 load_model="20260209_091859"
```

## Config

Everything is in `config/config.yaml`. Keys:

- `task` - Neurogym task name (required), e.g. `GoNogo-v0`, `ContextDecisionMaking-v0`
- `model` - `base_rnn` or `ctrnn` (required)
- `hidden_size` - number of hidden units, default 128
- `lr` - Adam learning rate, default 0.01
- `batch_size` - default 16
- `seq_len` - trial duration, default 100
- `num_epochs` - default 1000
- `tau` - CTRNN time constant in ms. ~100 for fast dynamics, ~1000 for slow integration
- `class_weights` - set True for imbalanced tasks
- `seed` - default 42
- `load_model` - timestamp string, only used by `evaluate.py`

To see what tasks Neurogym ships with:

```python
import neurogym as ngym
print(ngym.all_envs())
```

## Models

`base_rnn` is a plain RNN - fast, no time constant. You can use it for most discrete-time tasks.

`ctrnn` is a continuous-time RNN with a `tau` parameter controlling how fast neurons integrate. It's slower to train but the dynamics are closer to what you'd expect from biology, and it's the right choice when the task itself has slow timescales.

## Saving results

Each run creates `experiments/{task}/runs/{YYYYMMDD_HHMMSS}/` containing:

- `config.yaml` - the exact config used
- `{task}.png` - task visualization
- `checkpoints/` - model weights
- `logs/`, `tensorboard/`, `events.out.tfevents...` - for TensorBoard

Point TensorBoard at `experiments/{task}/runs/` to compare runs.

## Evaluation

`evaluate.py` loads a checkpoint, runs the model on the test set, and dumps accuracy + plots (PCA of hidden states, input/output trajectories) into the same run folder. The plotting helpers are in `utils/utils.py` if you want to call them yourself:

```python
from rnn_neuro.utils.utils import plot_pca, plot_input_output
plot_pca(model_outputs, labels, save_path='pca.png')
plot_input_output(inputs, outputs, targets, save_path='io.png')
```

## Notes

GPU is used automatically when available, otherwise CPU. Seeds are applied to torch, numpy, and Neurogym, and CUDA kernels are set to deterministic mode.

A few things that I had problems with:

- If training is too slow, drop `seq_len`/`batch_size`/`num_epochs` first, then consider switching to `base_rnn`.
- Poor performance is usually fixed by bumping `hidden_size` to 256 or training longer. For CTRNN, also try a different `tau`.
- If `evaluate.py` can't find your model, double-check the timestamp matches an actual folder under `experiments/{task}/runs/`.

## Examples

Train and then evaluate:

```bash
python main.py task=GoNogo-v0 model=base_rnn seed=42
# note the timestamp it prints, then:
python evaluate.py task=GoNogo-v0 load_model="20260212_143022"
```

Sweeping a few CTRNN configs:

```bash
python main.py task=ContextDecisionMaking-v0 model=ctrnn hidden_size=128 tau=500
python main.py task=ContextDecisionMaking-v0 model=ctrnn hidden_size=256 tau=500
python main.py task=ContextDecisionMaking-v0 model=ctrnn hidden_size=256 tau=1000
```

Each lands in its own timestamped folder.

## References

Yang et al., "Task representations in neural networks trained to perform many cognitive tasks", Nature Neuroscience (2019).

[Neurogym](https://github.com/neurogym/neurogym) · [PyTorch](https://pytorch.org/) · [Hydra](https://hydra.cc/)