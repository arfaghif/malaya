"""doc."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tensorflow as tf
import custom_modeling
import xlnet


def construct_scalar_host_call(
    monitor_dict, model_dir, prefix = '', reduce_fn = None
):
    """
  Construct host calls to monitor training progress on TPUs.
  """

    metric_names = list(monitor_dict.keys())

    def host_call_fn(global_step, *args):
        """actual host call function."""
        step = global_step[0]
        with tf.compat.v1.estimator.summary.create_file_writer(
            logdir = model_dir, filename_suffix = '.host_call'
        ).as_default():
            with tf.compat.v1.estimator.summary.always_record_summaries():
                for i, name in enumerate(metric_names):
                    if reduce_fn is None:
                        scalar = args[i][0]
                    else:
                        scalar = reduce_fn(args[i])
                    with tf.compat.v1.estimator.summary.record_summaries_every_n_global_steps(
                        100, global_step = step
                    ):
                        tf.compat.v1.estimator.summary.scalar(
                            prefix + name, scalar, step = step
                        )

                return tf.compat.v1.estimator.summary.all_summary_ops()

    global_step_tensor = tf.compat.v1.reshape(tf.compat.v1.train.get_or_create_global_step(), [1])
    other_tensors = [tf.compat.v1.reshape(monitor_dict[key], [1]) for key in metric_names]

    return host_call_fn, [global_step_tensor] + other_tensors


def two_stream_loss(FLAGS, features, labels, mems, is_training):
    """Pretraining loss with two-stream attention Transformer-XL."""

    #### Unpack input
    mem_name = 'mems'
    mems = mems.get(mem_name, None)

    inp_k = tf.compat.v1.transpose(features['input_k'], [1, 0])
    inp_q = tf.compat.v1.transpose(features['input_q'], [1, 0])

    seg_id = tf.compat.v1.transpose(features['seg_id'], [1, 0])

    inp_mask = None
    perm_mask = tf.compat.v1.transpose(features['perm_mask'], [1, 2, 0])

    if FLAGS.num_predict is not None:
        # [num_predict x tgt_len x bsz]
        target_mapping = tf.compat.v1.transpose(features['target_mapping'], [1, 2, 0])
    else:
        target_mapping = None

    # target for LM loss
    tgt = tf.compat.v1.transpose(features['target'], [1, 0])

    # target mask for LM loss
    tgt_mask = tf.compat.v1.transpose(features['target_mask'], [1, 0])

    # construct xlnet config and save to model_dir
    xlnet_config = xlnet.XLNetConfig(FLAGS = FLAGS)
    xlnet_config.to_json(os.path.join(FLAGS.model_dir, 'config.json'))

    # construct run config from FLAGS
    run_config = xlnet.create_run_config(is_training, False, FLAGS)

    xlnet_model = xlnet.XLNetModel(
        xlnet_config = xlnet_config,
        run_config = run_config,
        input_ids = inp_k,
        seg_ids = seg_id,
        input_mask = inp_mask,
        mems = mems,
        perm_mask = perm_mask,
        target_mapping = target_mapping,
        inp_q = inp_q,
    )

    output = xlnet_model.get_sequence_output()
    new_mems = {mem_name: xlnet_model.get_new_memory()}
    lookup_table = xlnet_model.get_embedding_table()
    lookup_table_2 = xlnet_model.get_embedding_table2()

    initializer = xlnet_model.get_initializer()

    with tf.compat.v1.variable_scope('model', reuse = tf.compat.v1.AUTO_REUSE):
        # LM loss
        accuracy, lm_loss = custom_modeling.lm_accuracy(
            hidden = output,
            target = tgt,
            n_token = xlnet_config.n_token,
            d_model = xlnet_config.d_model,
            initializer = initializer,
            lookup_table = lookup_table,
            lookup_table_2 = lookup_table_2,
            tie_weight = True,
            bi_data = run_config.bi_data,
            use_tpu = run_config.use_tpu,
        )

    #### Quantity to monitor
    monitor_dict = {}

    if FLAGS.use_bfloat16:
        tgt_mask = tf.compat.v1.cast(tgt_mask, tf.compat.v1.float32)
        lm_loss = tf.compat.v1.cast(lm_loss, tf.compat.v1.float32)
    print(tgt_mask, lm_loss, accuracy)

    total_loss = tf.compat.v1.reduce_sum(lm_loss * tgt_mask) / tf.compat.v1.reduce_sum(tgt_mask)
    #     total_accuracy = tf.compat.v1.reduce_sum(accuracy * tgt_mask) / tf.compat.v1.reduce_sum(
    #         accuracy
    #     )
    total_accuracy = accuracy
    monitor_dict['total_loss'] = total_loss
    # monitor_dict['total_accuracy'] = total_accuracy

    return total_loss, new_mems, monitor_dict


def get_loss(FLAGS, features, labels, mems, is_training):
    """Pretraining loss with two-stream attention Transformer-XL."""
    if FLAGS.use_bfloat16:
        with tf.compat.v1.tpu.bfloat16_scope():
            return two_stream_loss(FLAGS, features, labels, mems, is_training)
    else:
        return two_stream_loss(FLAGS, features, labels, mems, is_training)


def get_classification_loss(FLAGS, features, n_class, is_training):
    """Loss for downstream classification tasks."""

    bsz_per_core = tf.compat.v1.shape(features['input_ids'])[0]

    inp = tf.compat.v1.transpose(features['input_ids'], [1, 0])
    seg_id = tf.compat.v1.transpose(features['segment_ids'], [1, 0])
    inp_mask = tf.compat.v1.transpose(features['input_mask'], [1, 0])
    label = tf.compat.v1.reshape(features['label_ids'], [bsz_per_core])

    xlnet_config = xlnet.XLNetConfig(json_path = FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)

    xlnet_model = xlnet.XLNetModel(
        xlnet_config = xlnet_config,
        run_config = run_config,
        input_ids = inp,
        seg_ids = seg_id,
        input_mask = inp_mask,
    )

    summary = xlnet_model.get_pooled_out(
        FLAGS.summary_type, FLAGS.use_summ_proj
    )

    with tf.compat.v1.variable_scope('model', reuse = tf.compat.v1.AUTO_REUSE):

        if FLAGS.cls_scope is not None and FLAGS.cls_scope:
            cls_scope = 'classification_{}'.format(FLAGS.cls_scope)
        else:
            cls_scope = 'classification_{}'.format(FLAGS.task_name.lower())

        per_example_loss, logits = modeling.classification_loss(
            hidden = summary,
            labels = label,
            n_class = n_class,
            initializer = xlnet_model.get_initializer(),
            scope = cls_scope,
            return_logits = True,
        )

        total_loss = tf.compat.v1.reduce_mean(per_example_loss)

        return total_loss, per_example_loss, logits


def get_regression_loss(FLAGS, features, is_training):
    """Loss for downstream regression tasks."""

    bsz_per_core = tf.compat.v1.shape(features['input_ids'])[0]

    inp = tf.compat.v1.transpose(features['input_ids'], [1, 0])
    seg_id = tf.compat.v1.transpose(features['segment_ids'], [1, 0])
    inp_mask = tf.compat.v1.transpose(features['input_mask'], [1, 0])
    label = tf.compat.v1.reshape(features['label_ids'], [bsz_per_core])

    xlnet_config = xlnet.XLNetConfig(json_path = FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)

    xlnet_model = xlnet.XLNetModel(
        xlnet_config = xlnet_config,
        run_config = run_config,
        input_ids = inp,
        seg_ids = seg_id,
        input_mask = inp_mask,
    )

    summary = xlnet_model.get_pooled_out(
        FLAGS.summary_type, FLAGS.use_summ_proj
    )

    with tf.compat.v1.variable_scope('model', reuse = tf.compat.v1.AUTO_REUSE):
        per_example_loss, logits = modeling.regression_loss(
            hidden = summary,
            labels = label,
            initializer = xlnet_model.get_initializer(),
            scope = 'regression_{}'.format(FLAGS.task_name.lower()),
            return_logits = True,
        )

        total_loss = tf.compat.v1.reduce_mean(per_example_loss)

        return total_loss, per_example_loss, logits


def get_qa_outputs(FLAGS, features, is_training):
    """Loss for downstream span-extraction QA tasks such as SQuAD."""

    inp = tf.compat.v1.transpose(features['input_ids'], [1, 0])
    seg_id = tf.compat.v1.transpose(features['segment_ids'], [1, 0])
    inp_mask = tf.compat.v1.transpose(features['input_mask'], [1, 0])
    cls_index = tf.compat.v1.reshape(features['cls_index'], [-1])

    seq_len = tf.compat.v1.shape(inp)[0]

    xlnet_config = xlnet.XLNetConfig(json_path = FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)

    xlnet_model = xlnet.XLNetModel(
        xlnet_config = xlnet_config,
        run_config = run_config,
        input_ids = inp,
        seg_ids = seg_id,
        input_mask = inp_mask,
    )
    output = xlnet_model.get_sequence_output()
    initializer = xlnet_model.get_initializer()

    return_dict = {}

    # invalid position mask such as query and special symbols (PAD, SEP, CLS)
    p_mask = features['p_mask']

    # logit of the start position
    with tf.compat.v1.variable_scope('start_logits'):
        start_logits = tf.compat.v1.layers.dense(
            output, 1, kernel_initializer = initializer
        )
        start_logits = tf.compat.v1.transpose(tf.compat.v1.squeeze(start_logits, -1), [1, 0])
        start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
        start_log_probs = tf.compat.v1.nn.log_softmax(start_logits_masked, -1)

    # logit of the end position
    with tf.compat.v1.variable_scope('end_logits'):
        if is_training:
            # during training, compute the end logits based on the
            # ground truth of the start position

            start_positions = tf.compat.v1.reshape(features['start_positions'], [-1])
            start_index = tf.compat.v1.one_hot(
                start_positions, depth = seq_len, axis = -1, dtype = tf.compat.v1.float32
            )
            start_features = tf.compat.v1.einsum('lbh,bl->bh', output, start_index)
            start_features = tf.compat.v1.tile(start_features[None], [seq_len, 1, 1])
            end_logits = tf.compat.v1.layers.dense(
                tf.compat.v1.concat([output, start_features], axis = -1),
                xlnet_config.d_model,
                kernel_initializer = initializer,
                activation = tf.compat.v1.tanh,
                name = 'dense_0',
            )
            end_logits = tf.keras.layers.LayerNormalization(
                end_logits, begin_norm_axis = -1
            )

            end_logits = tf.compat.v1.layers.dense(
                end_logits,
                1,
                kernel_initializer = initializer,
                name = 'dense_1',
            )
            end_logits = tf.compat.v1.transpose(tf.compat.v1.squeeze(end_logits, -1), [1, 0])
            end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
            end_log_probs = tf.compat.v1.nn.log_softmax(end_logits_masked, -1)
        else:
            # during inference, compute the end logits based on beam search

            start_top_log_probs, start_top_index = tf.compat.v1.nn.top_k(
                start_log_probs, k = FLAGS.start_n_top
            )
            start_index = tf.compat.v1.one_hot(
                start_top_index, depth = seq_len, axis = -1, dtype = tf.compat.v1.float32
            )
            start_features = tf.compat.v1.einsum('lbh,bkl->bkh', output, start_index)
            end_input = tf.compat.v1.tile(
                output[:, :, None], [1, 1, FLAGS.start_n_top, 1]
            )
            start_features = tf.compat.v1.tile(start_features[None], [seq_len, 1, 1, 1])
            end_input = tf.compat.v1.concat([end_input, start_features], axis = -1)
            end_logits = tf.compat.v1.layers.dense(
                end_input,
                xlnet_config.d_model,
                kernel_initializer = initializer,
                activation = tf.compat.v1.tanh,
                name = 'dense_0',
            )
            end_logits = tf.keras.layers.LayerNormalization(
                end_logits, begin_norm_axis = -1
            )
            end_logits = tf.compat.v1.layers.dense(
                end_logits,
                1,
                kernel_initializer = initializer,
                name = 'dense_1',
            )
            end_logits = tf.compat.v1.reshape(
                end_logits, [seq_len, -1, FLAGS.start_n_top]
            )
            end_logits = tf.compat.v1.transpose(end_logits, [1, 2, 0])
            end_logits_masked = (
                end_logits * (1 - p_mask[:, None]) - 1e30 * p_mask[:, None]
            )
            end_log_probs = tf.compat.v1.nn.log_softmax(end_logits_masked, -1)
            end_top_log_probs, end_top_index = tf.compat.v1.nn.top_k(
                end_log_probs, k = FLAGS.end_n_top
            )
            end_top_log_probs = tf.compat.v1.reshape(
                end_top_log_probs, [-1, FLAGS.start_n_top * FLAGS.end_n_top]
            )
            end_top_index = tf.compat.v1.reshape(
                end_top_index, [-1, FLAGS.start_n_top * FLAGS.end_n_top]
            )

    if is_training:
        return_dict['start_log_probs'] = start_log_probs
        return_dict['end_log_probs'] = end_log_probs
    else:
        return_dict['start_top_log_probs'] = start_top_log_probs
        return_dict['start_top_index'] = start_top_index
        return_dict['end_top_log_probs'] = end_top_log_probs
        return_dict['end_top_index'] = end_top_index

    # an additional layer to predict answerability
    with tf.compat.v1.variable_scope('answer_class'):
        # get the representation of CLS
        cls_index = tf.compat.v1.one_hot(
            cls_index, seq_len, axis = -1, dtype = tf.compat.v1.float32
        )
        cls_feature = tf.compat.v1.einsum('lbh,bl->bh', output, cls_index)

        # get the representation of START
        start_p = tf.compat.v1.nn.softmax(
            start_logits_masked, axis = -1, name = 'softmax_start'
        )
        start_feature = tf.compat.v1.einsum('lbh,bl->bh', output, start_p)

        # note(zhiliny): no dependency on end_feature so that we can obtain
        # one single `cls_logits` for each sample
        ans_feature = tf.compat.v1.concat([start_feature, cls_feature], -1)
        ans_feature = tf.compat.v1.layers.dense(
            ans_feature,
            xlnet_config.d_model,
            activation = tf.compat.v1.tanh,
            kernel_initializer = initializer,
            name = 'dense_0',
        )
        ans_feature = tf.compat.v1.layers.dropout(
            ans_feature, FLAGS.dropout, training = is_training
        )
        cls_logits = tf.compat.v1.layers.dense(
            ans_feature,
            1,
            kernel_initializer = initializer,
            name = 'dense_1',
            use_bias = False,
        )
        cls_logits = tf.compat.v1.squeeze(cls_logits, -1)

        return_dict['cls_logits'] = cls_logits

    return return_dict


def get_race_loss(FLAGS, features, is_training):
    """Loss for downstream multi-choice QA tasks such as RACE."""

    bsz_per_core = tf.compat.v1.shape(features['input_ids'])[0]

    def _transform_features(feature):
        out = tf.compat.v1.reshape(feature, [bsz_per_core, 4, -1])
        out = tf.compat.v1.transpose(out, [2, 0, 1])
        out = tf.compat.v1.reshape(out, [-1, bsz_per_core * 4])
        return out

    inp = _transform_features(features['input_ids'])
    seg_id = _transform_features(features['segment_ids'])
    inp_mask = _transform_features(features['input_mask'])
    label = tf.compat.v1.reshape(features['label_ids'], [bsz_per_core])

    xlnet_config = xlnet.XLNetConfig(json_path = FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)

    xlnet_model = xlnet.XLNetModel(
        xlnet_config = xlnet_config,
        run_config = run_config,
        input_ids = inp,
        seg_ids = seg_id,
        input_mask = inp_mask,
    )
    summary = xlnet_model.get_pooled_out(
        FLAGS.summary_type, FLAGS.use_summ_proj
    )

    with tf.compat.v1.variable_scope('logits'):
        logits = tf.compat.v1.layers.dense(
            summary, 1, kernel_initializer = xlnet_model.get_initializer()
        )
        logits = tf.compat.v1.reshape(logits, [bsz_per_core, 4])

        one_hot_target = tf.compat.v1.one_hot(label, 4)
        per_example_loss = -tf.compat.v1.reduce_sum(
            tf.compat.v1.nn.log_softmax(logits) * one_hot_target, -1
        )
        total_loss = tf.compat.v1.reduce_mean(per_example_loss)

    return total_loss, per_example_loss, logits
