import tqdm
from rnn_neuro.utils.utils import save_model, \
                                  log_param_histograms, save_input_output
from rnn_neuro.analysis.eval import evaluate_model
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def train_model(model, optimizer, batch_iter,
                eval_env, loss_fn, num_epochs, device,
                model_name, run_dir, log, seed,
                periodic_eval, mode, num_trials_perf):
    """
    Loop to train model and save it at the end.

    Args:
        model: Network to be trained
        optimizer: Optimizer to be used (e.g. Adam)
        batch_iter: An iterator that yields (x, y) or (x, y, extra)
                    where x is input of shape
                    (seq_length, batch_size, input_size)
                    will be used for training
        eval_env: The env on which evaluation will be made
        loss_fn: Loss function to be used (e.g. CrossEntropy)
        num_epochs: Total epochs to train model
        device: Which device to run on
        model_name: Under which name to save model
        run_dir: Folder where to save model and data
        log: Logger to log information
        seed: Seed for reproducibility
        periodic_eval: Flag for periodic evaluation
        mode="single": one NeuroGym env (env.new_trial())
        mode="yang19": env is a collection with env.envs
        num_trials_perf: Number of trials to evaluate on
    """
    writer = SummaryWriter(f'{run_dir}')  # The path is relative to cwd when running main.py
    model.to(device)
    model.train()  # Set model in train mode
    running_loss = 0
    output_activity = []
    input_activity = []
    for epoch in tqdm.tqdm(range(num_epochs)):
        batch = next(batch_iter)
        x, y = batch
        x = torch.from_numpy(x).type(torch.float).to(device)
        y = torch.from_numpy(y.flatten()).type(torch.long).to(device)
        pred, _ = model(x)
        # input_activity.append(x.detach().cpu().numpy())
        # output_activity.append(pred.detach().cpu().numpy())
        loss = loss_fn(pred.view(-1, model.output_size), y)  # Flatten pred in the first dimension, such that it keeps the same number of elements, and on the second dimension it's the class label for softmax
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (epoch+1) % 100 == 0:
            log.info('Epoch {:d} loss: {:0.5f}'.format(epoch + 1, running_loss / 100))
            writer.add_scalar('training loss',
                              running_loss / 100,
                              epoch)
            if periodic_eval:
                results = evaluate_model(model=model,
                                         num_trials=num_trials_perf,
                                         env=eval_env,
                                         device=device,
                                         seed=seed,
                                         mode=mode)
                model.train()  # Set model in train mode after evaluation
                if mode == 'single':
                    log.info('Average performance in {:d} trials:\
                            {:0.2f}'.format(num_trials_perf, results['perf']))
                    writer.add_scalar('accuracy',
                                      results['perf'],
                                      epoch)
                elif mode == 'yang19':
                    # for env_id, acc in results["perf"].items():
                    #     print(f"{env_id}: {acc:.4f}")
                    #     writer.add_scalar(f'accuracy/{env_id}',
                    #                       acc,
                    #                       epoch)
                    writer.add_scalars('accuracy',
                                       results['perf'],
                                       epoch)
                else:
                    raise ValueError(f"Unknown mode: {mode!r}. Use 'single' or 'yang19'.")
            log_param_histograms(writer, epoch, model)
            running_loss = 0.0
    writer.flush()
    writer.close()
    save_model(model=model, model_name=model_name, model_path=run_dir)
    # save_input_output(np.array(input_activity), np.array(output_activity), run_dir)