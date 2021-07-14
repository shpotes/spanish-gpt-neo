#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 Santiago Hincapie-Potes & The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Recycle Spanish GPT-Neo from GPT-Neo.
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import time
from pathlib import Path
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
from flax.jax_utils import unreplicate
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from transformers import is_tensorboard_available,

from utils import create_learning_rate_fn, data_loader, TrainState, write_train_metric, write_eval_metric

logger = logging.getLogger(__name__)

def recycle(model, lm_train_dataset, lm_eval_dataset, training_args):

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    steps_per_epoch = len(lm_train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(lm_train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # create adam optimizer
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    trainable_params = {'transformer': {'wte': model.params['transformer']['wte']}}
    params = model.params

    # Setup train state
    state = TrainState.create(apply_fn=model.__call__, params=trainable_params, tx=adamw, dropout_rng=dropout_rng)

    def loss_fn(logits, labels):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        loss = optax.softmax_cross_entropy(shift_logits, onehot(shift_labels, shift_logits.shape[-1]))
        return loss.mean()

    # Define gradient update step fn
    def train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch")

    # Replicate the train state on each device
    state = state.replicate()

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    train_metrics = []
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        train_loader = data_loader(input_rng, lm_train_dataset, train_batch_size, shuffle=True)
        steps_per_epoch = len(lm_train_dataset) // train_batch_size
        # train
        for step in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
            batch = next(train_loader)
            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)

            cur_step = epoch * (len(lm_train_dataset) // train_batch_size) + step

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                # Save metrics
                train_metric = unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                epochs.write(
                    f"Step... ({cur_step} | Loss: {train_metric['loss'].mean()}, Learning Rate: {train_metric['learning_rate'].mean()})"
                )

                train_metrics = []

            if cur_step % training_args.eval_steps == 0 and cur_step > 0:
                # ======================== Evaluating ==============================
                eval_metrics = []
                eval_loader = data_loader(input_rng, lm_eval_dataset, eval_batch_size)
                eval_steps = len(lm_eval_dataset) // eval_batch_size
                for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
                    # Model forward
                    batch = next(eval_loader)
                    metrics = p_eval_step(state.params, batch)
                    eval_metrics.append(metrics)

                # normalize eval metrics
                eval_metrics = get_metrics(eval_metrics)
                eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

                try:
                    eval_metrics["perplexity"] = math.exp(eval_metrics["loss"])
                except OverflowError:
                    eval_metrics["perplexity"] = float("inf")

                # Print metrics and update progress bar
                desc = f"Step... ({cur_step} | Eval Loss: {eval_metrics['loss']} | Eval Perplexity: {eval_metrics['perplexity']})"
                epochs.write(desc)
                epochs.desc = desc

                # Save metrics
                if has_tensorboard and jax.process_index() == 0:
                    cur_step = epoch * (len(lm_train_dataset) // train_batch_size)
                    write_eval_metric(summary_writer, eval_metrics, cur_step)

    trained_params = jax.device_get(unreplicate(state.params))
    params['transformer']['wte'] = trained_params['transformer']['wte']

    return params