import tensorflow as tf
from tensor2tensor.utils import adafactor
from pegasus import transformer
from tensorflow.contrib import layers as contrib_layers
import re
import collections
import six

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
    512,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded. Must match data generation.',
)

flags.DEFINE_integer(
    'max_seq_length_decoder',
    512,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded. Must match data generation.',
)

flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')

flags.DEFINE_float(
    'learning_rate', 0.0001, 'The initial learning rate for Adafactor.'
)

flags.DEFINE_integer('num_train_steps', 1000000, 'Number of training steps.')

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

vocab_size = 32000
hidden_size = 768
filter_size = 3072
num_encoder_layers = 12
num_decoder_layers = 12
num_heads = 12
label_smoothing = 0.0
dropout = 0.1


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
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ':0'] = 1

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
                tf.compat.v1.estimator.data.parallel_interleave(
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
            tf.compat.v1.estimator.data.map_and_batch(
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

        model = transformer.TransformerEncoderDecoderModel(
            vocab_size,
            hidden_size,
            filter_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            label_smoothing,
            dropout,
        )

        loss, outputs = model(
            {'inputs': inputs, 'targets': targets}, training=is_training
        )

        total_loss = loss

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
                optimizer = tf.compat.v1.estimator.tpu.CrossShardOptimizer(optimizer)

            train_op = optimizer.minimize(loss, global_step=global_step)

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
            )
        elif mode == tf.compat.v1.estimator.ModeKeys.EVAL:

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss, scaffold_fn=scaffold_fn
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
        tpu_cluster_resolver = tf.compat.v1.estimator.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project
        )

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
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

    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
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
