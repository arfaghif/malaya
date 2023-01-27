import tensorflow as tf
from bigbird import modeling, optimization
import re
import collections
import six
import logging
from tensor2tensor.utils import adafactor
import optimization

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

flags.DEFINE_integer(
    'max_seq_length_encoder',
    1536,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded. Must match data generation.',
)

flags.DEFINE_integer(
    'max_seq_length_decoder',
    768,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded. Must match data generation.',
)

flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')

flags.DEFINE_float(
    'learning_rate', 0.0001, 'The initial learning rate for Adafactor.'
)

flags.DEFINE_integer('num_train_steps', 100000, 'Number of training steps.')

flags.DEFINE_integer('num_warmup_steps', 10000, 'Number of warmup steps.')

flags.DEFINE_integer(
    'save_checkpoints_steps', 10000, 'How often to save the model checkpoint.'
)

flags.DEFINE_integer(
    'iterations_per_loop', 100, 'How many steps to make in each estimator call.'
)

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

flags.DEFINE_bool('do_train', True, 'Whether to run training.')

flags.DEFINE_string(
    'init_checkpoint',
    None,
    'Initial checkpoint (usually from a pre-trained PEGASUS model).',
)

bert_config = {
    # transformer basic configs
    'attention_probs_dropout_prob': 0.1,
    'hidden_act': 'relu',
    'hidden_dropout_prob': 0.1,
    'hidden_size': 768,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 4096,
    'max_encoder_length': 1536,
    'max_decoder_length': 768,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'type_vocab_size': 2,
    'scope': 'pegasus',
    'use_bias': False,
    'rescale_embedding': True,
    'vocab_model_file': None,
    # sparse mask configs
    'attention_type': 'block_sparse',
    'norm_type': 'prenorm',
    'block_size': 64,
    'num_rand_blocks': 3,
    'vocab_size': 32000,
    'beam_size': 1,
    'alpha': 0.0,
    'couple_encoder_decoder': False,
    'num_warmup_steps': 10000,
    'learning_rate': 0.0001,
    'label_smoothing': 0.1,
    'optimizer': 'Adafactor',
    'use_tpu': True,
}


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
    with tf.compat.v1.name_scope('loss'):

        if labels is not None:
            with tf.compat.v1.name_scope('smoothing_cross_entropy'):
                confidence = 1.0 - smoothing
                vocab_float = tf.compat.v1.cast(vocab_size - 1, tf.compat.v1.float32)
                low_confidence = (1.0 - confidence) / vocab_float
                soft_targets = tf.compat.v1.one_hot(
                    labels,
                    depth=vocab_size,
                    on_value=confidence,
                    off_value=low_confidence,
                )
                xentropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=soft_targets
                )

                normalizing_constant = -(
                    confidence * tf.compat.v1.math.log(confidence)
                    + vocab_float
                    * low_confidence
                    * tf.compat.v1.math.log(low_confidence + 1e-20)
                )
                xentropy -= normalizing_constant

            weights = tf.compat.v1.cast(tf.compat.v1.not_equal(labels, 0), tf.compat.v1.float32)
            loss = tf.compat.v1.reduce_sum(xentropy * weights) / tf.compat.v1.reduce_sum(weights)

        else:
            loss = tf.compat.v1.constant(0.0)

        return loss


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

        l = 'pegasus/' + name
        l = l.replace('embeddings/weights', 'embeddings/word_embeddings')
        l = l.replace('self/output', 'output')
        l = l.replace('ffn/dense_1', 'output/dense')
        l = l.replace('ffn', 'intermediate')
        l = l.replace('memory_attention/output', 'attention/encdec_output')
        l = l.replace('memory_attention', 'attention/encdec')

        if l not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable[l]
        initialized_variable_names[l + ':0'] = 1

    return (assignment_map, initialized_variable_names)


def input_fn_builder(
    input_files,
    max_seq_length_encoder,
    max_seq_length_decoder,
    is_training,
    num_cpu_threads=4,
):
    def input_fn(params):
        batch_size = params['batch_size']

        name_to_features = {
            'input_ids': tf.compat.v1.io.FixedLenFeature([max_seq_length_encoder], tf.compat.v1.int64),
            'target_ids': tf.compat.v1.io.FixedLenFeature(
                [max_seq_length_decoder], tf.compat.v1.int64
            ),
        }
        if is_training:
            d = tf.compat.v1.data.Dataset.from_tensor_slices(tf.compat.v1.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))
            cycle_length = min(num_cpu_threads, len(input_files))
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.compat.v1.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length,
                )
            )
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.compat.v1.data.TFRecordDataset(input_files)
            d = d.repeat()
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True,
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


def model_fn_builder(
    init_checkpoint, learning_rate, num_train_steps, num_warmup_steps, use_tpu
):
    def model_fn(features, labels, mode, params):
        tf.compat.v1.logging.info('*** Features ***')
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info(
                '  name = %s, shape = %s' % (name, features[name].shape)
            )

        inputs = features['input_ids']
        targets = features['target_ids']

        is_training = mode == tf.compat.v1.estimator.ModeKeys.TRAIN

        model = modeling.TransformerModel(bert_config)
        (llh, logits, pred_ids), _ = model(
            inputs, target_ids=targets, training=is_training
        )

        total_loss = padded_cross_entropy_loss(
            logits,
            targets,
            bert_config['label_smoothing'],
            bert_config['vocab_size'],
        )

        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (
                assignment_map,
                initialized_variable_names,
            ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
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
        print(initialized_variable_names)
        for var in tvars:
            init_string = ''
            if var.name in initialized_variable_names:
                init_string = ', *INIT_FROM_CKPT*'
            tf.compat.v1.logging.info(
                '  name = %s, shape = %s%s', var.name, var.shape, init_string
            )

        output_spec = None
        if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:

            init_lr = learning_rate
            global_step = tf.compat.v1.train.get_global_step()
            lr = (
                init_lr
                / 0.01
                * tf.compat.v1.rsqrt(tf.compat.v1.maximum(tf.compat.v1.to_float(global_step), 10000))
            )

            optimizer = adafactor.AdafactorOptimizer(
                learning_rate=lr,
                decay_rate=adafactor.adafactor_decay_rate_pow(0.8),
                beta1=0.0,
            )
            if use_tpu:
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

            train_op = optimizer.minimize(total_loss, global_step=global_step)

            # if not bert_config['use_bias']:
            #     logging.info('Fixing position embedding, i.e. not trainable.')
            #     posemb = 'pegasus/embeddings/position_embeddings'
            #     tvars = list(
            #         filter(lambda v: v.name.split(':')[0] != posemb, tvars)
            #     )

            # gradients = optimizer.compute_gradients(total_loss, tvars)

            # train_op = optimization.create_optimizer(
            #     total_loss,
            #     learning_rate,
            #     num_train_steps,
            #     num_warmup_steps,
            #     use_tpu,
            # )

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
            )
        elif mode == tf.compat.v1.estimator.ModeKeys.EVAL:

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=None,
                scaffold_fn=scaffold_fn,
            )
        else:
            raise ValueError(
                'Only TRAIN and EVAL modes are supported: %s' % (mode)
            )

        return output_spec

    return model_fn


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
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project
        )

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host,
        ),
    )

    model_fn = model_fn_builder(
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.train_batch_size,
    )

    if FLAGS.do_train:
        tf.compat.v1.logging.info('***** Running training *****')
        tf.compat.v1.logging.info('  Batch size = %d', FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length_encoder=FLAGS.max_seq_length_encoder,
            max_seq_length_decoder=FLAGS.max_seq_length_decoder,
            is_training=True,
        )
        estimator.train(
            input_fn=train_input_fn, max_steps=FLAGS.num_train_steps
        )


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('output_dir')
    tf.compat.v1.app.run()
