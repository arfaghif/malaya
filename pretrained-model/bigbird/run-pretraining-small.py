# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from bigbird import modeling
from bigbird import utils
import optimization
import tensorflow as tf

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file',
    None,
    'Input TF example files (can be a glob or comma separated).',
)

flags.DEFINE_string(
    'output_dir',
    None,
    'The output directory where the model checkpoints will be written.',
)

## Other parameters
flags.DEFINE_string(
    'init_checkpoint',
    None,
    'Initial checkpoint (usually from a pre-trained BERT model).',
)

flags.DEFINE_integer(
    'max_seq_length',
    512,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded. Must match data generation.',
)

flags.DEFINE_integer(
    'max_predictions_per_seq',
    20,
    'Maximum number of masked LM predictions per sequence. '
    'Must match data generation.',
)

flags.DEFINE_bool('do_train', False, 'Whether to run training.')

flags.DEFINE_bool('do_eval', False, 'Whether to run eval on the dev set.')

flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')

flags.DEFINE_integer('eval_batch_size', 8, 'Total batch size for eval.')

flags.DEFINE_float('learning_rate', 1e-4, 'The initial learning rate for Adam.')

flags.DEFINE_integer('num_train_steps', 700000, 'Number of training steps.')

flags.DEFINE_integer('num_warmup_steps', 10000, 'Number of warmup steps.')

flags.DEFINE_integer(
    'save_checkpoints_steps', 10000, 'How often to save the model checkpoint.'
)

flags.DEFINE_integer(
    'iterations_per_loop',
    1000,
    'How many steps to make in each estimator call.',
)

flags.DEFINE_integer('max_eval_steps', 100, 'Maximum number of eval steps.')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU or GPU/CPU.')

tf.compat.v1.flags.DEFINE_string(
    'tpu_name',
    None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.',
)

tf.compat.v1.flags.DEFINE_string(
    'tpu_zone',
    None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.',
)

tf.compat.v1.flags.DEFINE_string(
    'gcp_project',
    None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.',
)

tf.compat.v1.flags.DEFINE_string('master', None, '[Optional] TensorFlow master URL.')

flags.DEFINE_integer(
    'num_tpu_cores',
    8,
    'Only used if `use_tpu` is True. Total number of TPU cores to use.',
)

bert_config = {
    'attention_probs_dropout_prob': 0.1,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'hidden_size': 512,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 4096,
    'max_encoder_length': 512,
    'num_attention_heads': 8,
    'num_hidden_layers': 6,
    'type_vocab_size': 2,
    'scope': 'bert',
    'use_bias': True,
    'rescale_embedding': False,
    'vocab_model_file': None,
    'attention_type': 'block_sparse',
    'norm_type': 'postnorm',
    'block_size': 16,
    'num_rand_blocks': 3,
    'vocab_size': 32000,
}


class MaskedLMLayer(tf.layers.Layer):
    """Get loss and log probs for the masked LM."""

    def __init__(
        self,
        hidden_size,
        vocab_size,
        embeder,
        initializer = None,
        activation_fn = None,
        name = 'cls/predictions',
    ):
        super(MaskedLMLayer, self).__init__(name = name)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embeder = embeder

        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        self.extra_layer = utils.Dense2dLayer(
            hidden_size, initializer, activation_fn, 'transform'
        )
        self.norm_layer = utils.NormLayer('transform')

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.output_bias = tf.compat.v1.get_variable(
            name + '/output_bias',
            shape = [vocab_size],
            initializer = tf.compat.v1.zeros_initializer(),
        )

    @property
    def trainable_weights(self):
        self._trainable_weights = (
            self.extra_layer
            + self.norm_layer.trainable_weights
            + [self.output_bias]
        )
        return self._trainable_weights

    def call(
        self,
        input_tensor,
        label_ids = None,
        label_weights = None,
        masked_lm_positions = None,
    ):
        if masked_lm_positions is not None:
            input_tensor = tf.compat.v1.gather(
                input_tensor, masked_lm_positions, batch_dims = 1
            )

        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.compat.v1.variable_scope('transform') as sc:
            input_tensor = self.extra_layer(input_tensor, scope = sc)
            input_tensor = self.norm_layer(input_tensor, scope = sc)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        logits = self.embeder.linear(input_tensor)
        logits = tf.compat.v1.nn.bias_add(logits, self.output_bias)
        log_probs = tf.compat.v1.nn.log_softmax(logits, axis = -1)

        if label_ids is not None:
            one_hot_labels = tf.compat.v1.one_hot(
                label_ids, depth = self.vocab_size, dtype = tf.compat.v1.float32
            )

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            per_example_loss = -tf.compat.v1.reduce_sum(
                log_probs * one_hot_labels, axis = -1
            )
            numerator = tf.compat.v1.reduce_sum(label_weights * per_example_loss)
            denominator = tf.compat.v1.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator
        else:
            loss = tf.compat.v1.constant(0.0)

        return loss, log_probs


class NSPLayer(tf.layers.Layer):
    """Get loss and log probs for the next sentence prediction."""

    def __init__(
        self, hidden_size, initializer = None, name = 'cls/seq_relationship'
    ):
        super(NSPLayer, self).__init__(name = name)
        self.hidden_size = hidden_size

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.compat.v1.variable_scope(name):
            self.output_weights = tf.compat.v1.get_variable(
                'output_weights',
                shape = [2, hidden_size],
                initializer = initializer,
            )
            self._trainable_weights.append(self.output_weights)
            self.output_bias = tf.compat.v1.get_variable(
                'output_bias', shape = [2], initializer = tf.compat.v1.zeros_initializer()
            )
            self._trainable_weights.append(self.output_bias)

    def call(self, input_tensor, next_sentence_labels = None):
        logits = tf.compat.v1.matmul(
            input_tensor, self.output_weights, transpose_b = True
        )
        logits = tf.compat.v1.nn.bias_add(logits, self.output_bias)
        log_probs = tf.compat.v1.nn.log_softmax(logits, axis = -1)

        if next_sentence_labels is not None:
            labels = tf.compat.v1.reshape(next_sentence_labels, [-1])
            one_hot_labels = tf.compat.v1.one_hot(labels, depth = 2, dtype = tf.compat.v1.float32)
            per_example_loss = -tf.compat.v1.reduce_sum(
                one_hot_labels * log_probs, axis = -1
            )
            loss = tf.compat.v1.reduce_mean(per_example_loss)
        else:
            loss = tf.compat.v1.constant(0.0)
        return loss, log_probs


def model_fn_builder(
    init_checkpoint,
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    use_tpu,
    use_one_hot_embeddings,
):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(
        features, labels, mode, params
    ):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.compat.v1.logging.info('*** Features ***')
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info(
                '  name = %s, shape = %s' % (name, features[name].shape)
            )

        is_training = mode == tf.compat.v1.estimator.ModeKeys.TRAIN

        model = modeling.BertModel(bert_config)
        masked_lm = MaskedLMLayer(
            bert_config['hidden_size'],
            bert_config['vocab_size'],
            model.embeder,
            initializer = utils.create_initializer(
                bert_config['initializer_range']
            ),
            activation_fn = utils.get_activation(bert_config['hidden_act']),
        )
        next_sentence = NSPLayer(
            bert_config['hidden_size'],
            initializer = utils.create_initializer(
                bert_config['initializer_range']
            ),
        )
        sequence_output, pooled_output = model(
            features['input_ids'],
            training = is_training,
            token_type_ids = features.get('segment_ids'),
        )

        masked_lm_loss, masked_lm_log_probs = masked_lm(
            sequence_output,
            label_ids = features.get('masked_lm_ids'),
            label_weights = features.get('masked_lm_weights'),
            masked_lm_positions = features.get('masked_lm_positions'),
        )

        next_sentence_loss, next_sentence_log_probs = next_sentence(
            pooled_output, features.get('next_sentence_labels')
        )

        total_loss = masked_lm_loss + next_sentence_loss

        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (
                assignment_map,
                initialized_variable_names,
            ) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint
            )
            if use_tpu:

                def tpu_scaffold():
                    tf.compat.v1.train.init_from_checkpoint(
                        init_checkpoint, assignment_map
                    )
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.compat.v1.logging.info('**** Trainable Variables ****')
        for var in tvars:
            init_string = ''
            if var.name in initialized_variable_names:
                init_string = ', *INIT_FROM_CKPT*'
            tf.compat.v1.logging.info(
                '  name = %s, shape = %s%s', var.name, var.shape, init_string
            )

        output_spec = None
        if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss,
                learning_rate,
                num_train_steps,
                num_warmup_steps,
                use_tpu,
            )

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode = mode,
                loss = total_loss,
                train_op = train_op,
                scaffold_fn = scaffold_fn,
            )
        elif mode == tf.compat.v1.estimator.ModeKeys.EVAL:

            def metric_fn(
                masked_lm_example_loss,
                masked_lm_log_probs,
                masked_lm_ids,
                masked_lm_weights,
                next_sentence_example_loss,
                next_sentence_log_probs,
                next_sentence_labels,
            ):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.compat.v1.reshape(
                    masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]]
                )
                masked_lm_predictions = tf.compat.v1.argmax(
                    masked_lm_log_probs, axis = -1, output_type = tf.compat.v1.int32
                )
                masked_lm_example_loss = tf.compat.v1.reshape(
                    masked_lm_example_loss, [-1]
                )
                masked_lm_ids = tf.compat.v1.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.compat.v1.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
                    labels = masked_lm_ids,
                    predictions = masked_lm_predictions,
                    weights = masked_lm_weights,
                )
                masked_lm_mean_loss = tf.compat.v1.metrics.mean(
                    values = masked_lm_example_loss, weights = masked_lm_weights
                )

                next_sentence_log_probs = tf.compat.v1.reshape(
                    next_sentence_log_probs,
                    [-1, next_sentence_log_probs.shape[-1]],
                )
                next_sentence_predictions = tf.compat.v1.argmax(
                    next_sentence_log_probs, axis = -1, output_type = tf.compat.v1.int32
                )
                next_sentence_labels = tf.compat.v1.reshape(next_sentence_labels, [-1])
                next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
                    labels = next_sentence_labels,
                    predictions = next_sentence_predictions,
                )
                next_sentence_mean_loss = tf.compat.v1.metrics.mean(
                    values = next_sentence_example_loss
                )

                return {
                    'masked_lm_accuracy': masked_lm_accuracy,
                    'masked_lm_loss': masked_lm_mean_loss,
                    'next_sentence_accuracy': next_sentence_accuracy,
                    'next_sentence_loss': next_sentence_mean_loss,
                }

            eval_metrics = (
                metric_fn,
                [
                    masked_lm_loss,
                    masked_lm_log_probs,
                    features['masked_lm_ids'],
                    features['masked_lm_weights'],
                    next_sentence_loss,
                    next_sentence_log_probs,
                    features['next_sentence_labels'],
                ],
            )

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode = mode,
                loss = total_loss,
                eval_metrics = eval_metrics,
                scaffold_fn = scaffold_fn,
            )
        else:
            raise ValueError(
                'Only TRAIN and EVAL modes are supported: %s' % (mode)
            )

        return output_spec

    return model_fn


def input_fn_builder(
    input_files,
    max_seq_length,
    max_predictions_per_seq,
    is_training,
    num_cpu_threads = 4,
):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params['batch_size']

        name_to_features = {
            'input_ids': tf.compat.v1.io.FixedLenFeature([max_seq_length], tf.compat.v1.int64),
            'input_mask': tf.compat.v1.io.FixedLenFeature([max_seq_length], tf.compat.v1.int64),
            'segment_ids': tf.compat.v1.io.FixedLenFeature([max_seq_length], tf.compat.v1.int64),
            'masked_lm_positions': tf.compat.v1.io.FixedLenFeature(
                [max_predictions_per_seq], tf.compat.v1.int64
            ),
            'masked_lm_ids': tf.compat.v1.io.FixedLenFeature(
                [max_predictions_per_seq], tf.compat.v1.int64
            ),
            'masked_lm_weights': tf.compat.v1.io.FixedLenFeature(
                [max_predictions_per_seq], tf.compat.v1.float32
            ),
            'next_sentence_labels': tf.compat.v1.io.FixedLenFeature([1], tf.compat.v1.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.compat.v1.data.Dataset.from_tensor_slices(tf.compat.v1.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size = len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.compat.v1.estimator.data.parallel_interleave(
                    tf.compat.v1.data.TFRecordDataset,
                    sloppy = is_training,
                    cycle_length = cycle_length,
                )
            )
            d = d.shuffle(buffer_size = 100)
        else:
            d = tf.compat.v1.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.compat.v1.estimator.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size = batch_size,
                num_parallel_batches = num_cpu_threads,
                drop_remainder = True,
            )
        )
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.compat.v1.io.parse_single_example(record, name_to_features)

    # tf.compat.v1.Example only supports tf.compat.v1.int64, but the TPU only supports tf.compat.v1.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.compat.v1.int64:
            t = tf.compat.v1.to_int32(t)
        example[name] = t

    return example


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.info)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError(
            'At least one of `do_train` or `do_eval` must be True.'
        )

    tf.compat.v1.io.gfile.mkdir(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(','):
        input_files.extend(tf.compat.v1.gfile.Glob(input_pattern))

    tf.compat.v1.logging.info('*** Input Files ***')
    for input_file in input_files:
        tf.compat.v1.logging.info('  %s' % input_file)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.compat.v1.estimator.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone = FLAGS.tpu_zone, project = FLAGS.gcp_project
        )

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster = tpu_cluster_resolver,
        master = FLAGS.master,
        model_dir = FLAGS.output_dir,
        save_checkpoints_steps = FLAGS.save_checkpoints_steps,
        tpu_config = tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop = FLAGS.iterations_per_loop,
            num_shards = FLAGS.num_tpu_cores,
            per_host_input_for_training = is_per_host,
        ),
    )

    model_fn = model_fn_builder(
        init_checkpoint = FLAGS.init_checkpoint,
        learning_rate = FLAGS.learning_rate,
        num_train_steps = FLAGS.num_train_steps,
        num_warmup_steps = FLAGS.num_warmup_steps,
        use_tpu = FLAGS.use_tpu,
        use_one_hot_embeddings = FLAGS.use_tpu,
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu = FLAGS.use_tpu,
        model_fn = model_fn,
        config = run_config,
        train_batch_size = FLAGS.train_batch_size,
        eval_batch_size = FLAGS.eval_batch_size,
    )

    if FLAGS.do_train:
        tf.compat.v1.logging.info('***** Running training *****')
        tf.compat.v1.logging.info('  Batch size = %d', FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files = input_files,
            max_seq_length = FLAGS.max_seq_length,
            max_predictions_per_seq = FLAGS.max_predictions_per_seq,
            is_training = True,
        )
        estimator.train(
            input_fn = train_input_fn, max_steps = FLAGS.num_train_steps
        )

    if FLAGS.do_eval:
        tf.compat.v1.logging.info('***** Running evaluation *****')
        tf.compat.v1.logging.info('  Batch size = %d', FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files = input_files,
            max_seq_length = FLAGS.max_seq_length,
            max_predictions_per_seq = FLAGS.max_predictions_per_seq,
            is_training = False,
        )

        result = estimator.evaluate(
            input_fn = eval_input_fn, steps = FLAGS.max_eval_steps
        )

        output_eval_file = os.path.join(FLAGS.output_dir, 'eval_results.txt')
        with tf.compat.v1.gfile.GFile(output_eval_file, 'w') as writer:
            tf.compat.v1.logging.info('***** Eval results *****')
            for key in sorted(result.keys()):
                tf.compat.v1.logging.info('  %s = %s', key, str(result[key]))
                writer.write('%s = %s\n' % (key, str(result[key])))


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('output_dir')
    tf.compat.v1.app.run()
