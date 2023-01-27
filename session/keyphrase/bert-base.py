import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import bert
from bert import optimization
from bert import tokenization
from bert import modeling
import numpy as np
import json
import tensorflow as tf
import itertools
import collections
import re
import random
import sentencepiece as spm
from unidecode import unidecode
from sklearn.utils import shuffle
from prepro_utils import preprocess_text, encode_ids, encode_pieces
from malaya.text.function import transformer_textcleaning as cleaning
from tensorflow.python.estimator.run_config import RunConfig

with open('/home/husein/alxlnet/topics.json') as fopen:
    topics = set(json.load(fopen).keys())

list_topics = list(topics)

sp_model = spm.SentencePieceProcessor()
sp_model.Load('sp10m.cased.bert.model')

with open('sp10m.cased.bert.vocab') as fopen:
    v = fopen.read().split('\n')[:-1]
v = [i.split('\t') for i in v]
v = {i[0]: i[1] for i in v}


class Tokenizer:
    def __init__(self, v):
        self.vocab = v
        pass

    def tokenize(self, string):
        return encode_pieces(
            sp_model, string, return_unicode = False, sample = False
        )

    def convert_tokens_to_ids(self, tokens):
        return [sp_model.PieceToId(piece) for piece in tokens]

    def convert_ids_to_tokens(self, ids):
        return [sp_model.IdToPiece(i) for i in ids]


tokenizer = Tokenizer(v)


def F(text):
    tokens_a = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    input_id = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_id)
    return input_id, input_mask


def XY(data):

    if len(set(data[1]) & topics) and random.random() > 0.2:
        t = random.choice(data[1])
        label = 1
    else:
        s = set(data[1]) | set()
        t = random.choice(list(topics - s))
        label = 0
    X = F(cleaning(data[0]))
    Y = F(t)

    return X, Y, label


def generate():
    with open('/home/husein/alxlnet/trainset-keyphrase.json') as fopen:
        data = json.load(fopen)
    while True:
        data = shuffle(data)
        for i in range(len(data)):
            X, Y, label = XY(data[i])
            yield {
                'X': X[0],
                'mask': X[1],
                'X_b': Y[0],
                'mask_b': Y[1],
                'label': [label],
            }


def get_dataset(
    batch_size = 60, shuffle_size = 20, thread_count = 24, maxlen_feature = 1800
):
    def get():
        dataset = tf.compat.v1.data.Dataset.from_generator(
            generate,
            {
                'X': tf.compat.v1.int32,
                'mask': tf.compat.v1.int32,
                'X_b': tf.compat.v1.int32,
                'mask_b': tf.compat.v1.int32,
                'label': tf.compat.v1.int32,
            },
            output_shapes = {
                'X': tf.compat.v1.TensorShape([None]),
                'mask': tf.compat.v1.TensorShape([None]),
                'X_b': tf.compat.v1.TensorShape([None]),
                'mask_b': tf.compat.v1.TensorShape([None]),
                'label': tf.compat.v1.TensorShape([None]),
            },
        )
        dataset = dataset.prefetch(tf.compat.v1.contrib.data.AUTOTUNE)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'X': tf.compat.v1.TensorShape([None]),
                'mask': tf.compat.v1.TensorShape([None]),
                'X_b': tf.compat.v1.TensorShape([None]),
                'mask_b': tf.compat.v1.TensorShape([None]),
                'label': tf.compat.v1.TensorShape([None]),
            },
            padding_values = {
                'X': tf.compat.v1.constant(0, dtype = tf.compat.v1.int32),
                'mask': tf.compat.v1.constant(0, dtype = tf.compat.v1.int32),
                'X_b': tf.compat.v1.constant(0, dtype = tf.compat.v1.int32),
                'mask_b': tf.compat.v1.constant(0, dtype = tf.compat.v1.int32),
                'label': tf.compat.v1.constant(0, dtype = tf.compat.v1.int32),
            },
        )
        return dataset

    return get


def create_initializer(initializer_range = 0.02):
    return tf.compat.v1.truncated_normal_initializer(stddev = initializer_range)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match('^(.*):\\d+$', name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.compat.v1.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if 'bert/' + name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable['bert/' + name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ':0'] = 1

    return (assignment_map, initialized_variable_names)


batch_size = 60
warmup_proportion = 0.1
num_train_steps = 1000000
num_warmup_steps = int(num_train_steps * warmup_proportion)
learning_rate = 2e-5


def model_fn(features, labels, mode, params):
    bert_config = modeling.BertConfig.from_json_file(
        'bert-base-2020-03-19/bert_config.json'
    )

    X = features['X']
    input_masks = features['mask']

    X_b = features['X_b']
    input_masks_b = features['mask_b']

    Y = features['label'][:, 0]

    with @@#variable_scope('bert', reuse = False):
        model = modeling.BertModel(
            config = bert_config,
            is_training = True,
            input_ids = X,
            input_mask = input_masks,
            use_one_hot_embeddings = False,
        )

        summary = model.get_pooled_output()

    with @@#variable_scope('bert', reuse = True):
        model = modeling.BertModel(
            config = bert_config,
            is_training = True,
            input_ids = X_b,
            input_mask = input_masks_b,
            use_one_hot_embeddings = False,
        )

        summary_b = model.get_pooled_output()

    vectors_concat = [summary, summary_b, tf.compat.v1.abs(summary - summary_b)]
    vectors_concat = tf.compat.v1.concat(vectors_concat, axis = 1)
    logits = tf.compat.v1.layers.dense(vectors_concat, 2)

    loss = tf.compat.v1.reduce_mean(
        tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits, labels = Y
        )
    )
    tf.compat.v1.identity(loss, 'train_loss')

    accuracy = tf.compat.v1.metrics.accuracy(
        labels = Y, predictions = tf.compat.v1.argmax(logits, axis = 1)
    )
    tf.compat.v1.identity(accuracy[1], name = 'train_accuracy')

    tvars = tf.compat.v1.trainable_variables()
    init_checkpoint = 'bert-base-2020-03-19/model.ckpt-2000002'
    assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
        tvars, init_checkpoint
    )
    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
    if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(
            loss, learning_rate, num_train_steps, num_warmup_steps, False
        )
        estimator_spec = tf.compat.v1.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.compat.v1.estimator.ModeKeys.EVAL:

        estimator_spec = tf.compat.v1.estimator.EstimatorSpec(
            mode = tf.compat.v1.estimator.ModeKeys.EVAL,
            loss = loss,
            eval_metric_ops = {'accuracy': accuracy},
        )

    return estimator_spec


def run_training(
    train_fn,
    model_fn,
    model_dir: str,
    gpu_mem_fraction: float = 0.96,
    log_step: int = 100,
    summary_step: int = 100,
    save_checkpoint_step: int = 1000,
    max_steps: int = 10000,
    eval_step: int = 10,
    eval_throttle: int = 120,
    train_batch_size: int = 128,
    train_hooks = None,
    eval_fn = None,
):
    @@#logging.set_verbosity(@@#logging.info)
    dist_strategy = None

    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction = gpu_mem_fraction
    )
    config = tf.compat.v1.ConfigProto(
        allow_soft_placement = True, gpu_options = gpu_options
    )
    run_config = RunConfig(
        train_distribute = dist_strategy,
        eval_distribute = dist_strategy,
        log_step_count_steps = log_step,
        model_dir = model_dir,
        save_checkpoints_steps = save_checkpoint_step,
        save_summary_steps = summary_step,
        session_config = config,
    )

    estimator = tf.compat.v1.estimator.Estimator(
        model_fn = model_fn, params = {}, config = run_config
    )

    if eval_fn:
        train_spec = tf.compat.v1.estimator.TrainSpec(
            input_fn = train_fn, max_steps = max_steps, hooks = train_hooks
        )

        eval_spec = tf.compat.v1.estimator.EvalSpec(
            input_fn = eval_fn, steps = eval_step, throttle_secs = eval_throttle
        )
        tf.compat.v1.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    else:
        estimator.train(
            input_fn = train_fn, max_steps = max_steps, hooks = train_hooks
        )


train_hooks = [
    tf.compat.v1.train.LoggingTensorHook(
        ['train_accuracy', 'train_loss'], every_n_iter = 1
    )
]

train_dataset = get_dataset()

save_directory = 'bert-base-keyphrase'

run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    log_step = 1,
    save_checkpoint_step = 10000,
    max_steps = num_train_steps,
    train_hooks = train_hooks,
)
