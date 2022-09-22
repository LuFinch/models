# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Lint as: python3
"""Learning rate utilities for vision tasks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Mapping, Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

BASE_LEARNING_RATE = 0.1


@tf.keras.utils.register_keras_serializable(package='Custom', name='WarmupDeacySchedule')
class WarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A wrapper for LearningRateSchedule that includes warmup steps."""

  def __init__(self,
               lr_schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
               warmup_steps: int,
               warmup_lr: Optional[float] = None):
    """Add warmup decay to a learning rate schedule.

    Args:
      lr_schedule: base learning rate scheduler
      warmup_steps: number of warmup steps
      warmup_lr: an optional field for the final warmup learning rate. This
        should be provided if the base `lr_schedule` does not contain this
        field.
    """
    super(WarmupDecaySchedule, self).__init__()
    self._lr_schedule = lr_schedule
    self._warmup_steps = warmup_steps
    self._warmup_lr = warmup_lr

  def __call__(self, step: int):
    lr = self._lr_schedule(step)
    if self._warmup_steps:
      if self._warmup_lr is not None:
        initial_learning_rate = tf.convert_to_tensor(
            self._warmup_lr, name="initial_learning_rate")
      else:
        initial_learning_rate = tf.convert_to_tensor(
            self._lr_schedule.initial_learning_rate,
            name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      global_step_recomp = tf.cast(step, dtype)
      warmup_steps = tf.cast(self._warmup_steps, dtype)
      warmup_lr = initial_learning_rate * global_step_recomp / warmup_steps
      lr = tf.cond(global_step_recomp < warmup_steps, lambda: warmup_lr,
                   lambda: lr)
    return lr

  def get_config(self) -> Mapping[str, Any]:
    config = {}
    config.update({
        "warmup_steps": self._warmup_steps,
        "warmup_lr": self._warmup_lr,
        "lr_schedule": self._lr_schedule,
    })
    return config


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Class to generate learning rate tensor."""

  def __init__(self, batch_size: int, total_steps: int, warmup_steps: int):
    """Creates the consine learning rate tensor with linear warmup.

    Args:
      batch_size: The training batch size used in the experiment.
      total_steps: Total training steps.
      warmup_steps: Steps for the warm up period.
    """
    super(CosineDecayWithWarmup, self).__init__()
    base_lr_batch_size = 256
    self._total_steps = total_steps
    self._init_learning_rate = BASE_LEARNING_RATE * batch_size / base_lr_batch_size
    self._warmup_steps = warmup_steps

  def __call__(self, global_step: int):
    global_step = tf.cast(global_step, dtype=tf.float32)
    warmup_steps = self._warmup_steps
    init_lr = self._init_learning_rate
    total_steps = self._total_steps

    linear_warmup = global_step / warmup_steps * init_lr

    cosine_learning_rate = init_lr * (tf.cos(np.pi *
                                             (global_step - warmup_steps) /
                                             (total_steps - warmup_steps)) +
                                      1.0) / 2.0

    learning_rate = tf.where(global_step < warmup_steps, linear_warmup,
                             cosine_learning_rate)
    return learning_rate

  def get_config(self):
    return {
        "total_steps": self._total_steps,
        "warmup_learning_rate": self._warmup_learning_rate,
        "warmup_steps": self._warmup_steps,
        "init_learning_rate": self._init_learning_rate,
    }

@tf.keras.utils.register_keras_serializable(package='Custom', name='PolynomialDeacyWithWarmup')
class PolynomialDecayWithWarmup(
tf.keras.optimizers.schedules.LearningRateSchedule):
  """A LearningRateSchedule that uses a polynomial decay with warmup."""
  def __init__(
        self,
        batch_size,
        steps_per_epoch,
        train_steps,
        initial_learning_rate=None,
        end_learning_rate=None,
        warmup_epochs=None,
        compute_lr_on_cpu=False,
        name=None):
    """Applies a polynomial decay to the learning rate with warmup."""
    super(PolynomialDecayWithWarmup, self).__init__()

    self.batch_size = batch_size
    self.steps_per_epoch = steps_per_epoch
    self.train_steps = train_steps
    self.name = name
    self.learning_rate_ops_cache = {}
    self.compute_lr_on_cpu = compute_lr_on_cpu

    if batch_size < 16384:
        self.initial_learning_rate = 10.0
        warmup_epochs_ = 5
    elif batch_size < 32768:
        self.initial_learning_rate = 25.0
        warmup_epochs_ = 5
    else:
        self.initial_learning_rate = 31.2
        warmup_epochs_ = 25

    # Override default poly learning rate and warmup epochs
    if initial_learning_rate:
        self.initial_learning_rate = initial_learning_rate

    if end_learning_rate:
        self.end_learning_rate = end_learning_rate
    else:
        self.end_learning_rate = 0.0001

    if warmup_epochs is not None:
        warmup_epochs_ = warmup_epochs
    self.warmup_epochs = warmup_epochs_

    warmup_steps = warmup_epochs_ * steps_per_epoch
    self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    self.decay_steps = train_steps - warmup_steps + 1
    self.poly_rate_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=self.initial_learning_rate,
        decay_steps=self.decay_steps,
        end_learning_rate=self.end_learning_rate,
        power=2.0)

  def __call__(self, step):
    if tf.executing_eagerly():
      return self._get_learning_rate(step)

    # In an eager function or graph, the current implementation of optimizer
    # repeatedly call and thus create ops for the learning rate schedule. To
    # avoid this, we cache the ops if not executing eagerly.
    graph = tf.compat.v1.get_default_graph()
    if graph not in self.learning_rate_ops_cache:
      if self.compute_lr_on_cpu:
        with tf.device('/device:CPU:0'):
          self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
      else:
        self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
    return self.learning_rate_ops_cache[graph]

  def _get_learning_rate(self, step):
    with ops.name_scope_v2(self.name or 'PolynomialDecayWithWarmup') as name:
      initial_learning_rate = ops.convert_to_tensor_v2(
        self.initial_learning_rate, name='initial_learning_rate')
      warmup_steps = ops.convert_to_tensor_v2(
        self.warmup_steps, name='warmup_steps')
      step = tf.cast(step, tf.float32)
      warmup_rate = (
            initial_learning_rate * step / warmup_steps)

      poly_steps = math_ops.subtract(step, warmup_steps)
      poly_rate = self.poly_rate_scheduler(poly_steps)

      decay_rate = tf.where(step <= warmup_steps,
                          warmup_rate, poly_rate, name=name)
      return decay_rate

  def get_config(self):
    return {
      'batch_size': self.batch_size,
      'steps_per_epoch': self.steps_per_epoch,
      'train_steps': self.train_steps,
      'initial_learning_rate': self.initial_learning_rate,
      'end_learning_rate': self.end_learning_rate,
      'warmup_epochs': self.warmup_epochs,
      'name': self.name,
    }

