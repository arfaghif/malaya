from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def gelu(x):
    """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
    cdf = 0.5 * (
        1.0 + tf.compat.v1.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.compat.v1.pow(x, 3))))
    )
    return x * cdf


def embedding_lookup(
    x,
    n_token,
    d_embed,
    hidden_size,
    initializer,
    use_tpu = True,
    scope = 'embedding',
    reuse = None,
    dtype = tf.compat.v1.float32,
):
    """TPU and GPU embedding_lookup function."""
    with tf.compat.v1.variable_scope(scope, reuse = reuse):
        lookup_table = tf.compat.v1.get_variable(
            'lookup_table',
            [n_token, d_embed],
            dtype = dtype,
            initializer = initializer,
        )

        project_variable = tf.compat.v1.get_variable(  # [embedding_size, hidden_size]
            name = 'lookup_table_2',
            shape = [d_embed, hidden_size],
            initializer = initializer,
        )

        if use_tpu:
            one_hot_idx = tf.compat.v1.one_hot(x, n_token, dtype = dtype)
            if one_hot_idx.shape.ndims == 2:
                return (
                    tf.compat.v1.einsum('in,nd->id', one_hot_idx, lookup_table),
                    lookup_table,
                )
            else:
                return (
                    tf.compat.v1.einsum('ibn,nd->ibd', one_hot_idx, lookup_table),
                    lookup_table,
                )
        else:
            output_middle = tf.compat.v1.nn.embedding_lookup(lookup_table, x)
            output = tf.compat.v1.matmul(output_middle, project_variable)
            return output, lookup_table, project_variable


def positional_embedding(pos_seq, inv_freq, bsz = None):
    sinusoid_inp = tf.compat.v1.einsum('i,d->id', pos_seq, inv_freq)
    pos_emb = tf.compat.v1.concat([tf.compat.v1.sin(sinusoid_inp), tf.compat.v1.cos(sinusoid_inp)], -1)
    pos_emb = pos_emb[:, None, :]

    if bsz is not None:
        pos_emb = tf.compat.v1.tile(pos_emb, [1, bsz, 1])

    return pos_emb


def positionwise_ffn(
    inp,
    d_model,
    d_inner,
    dropout,
    kernel_initializer,
    activation_type = 'relu',
    scope = 'ff',
    is_training = True,
    reuse = None,
):
    """Position-wise Feed-forward Network."""
    if activation_type == 'relu':
        activation = tf.compat.v1.nn.relu
    elif activation_type == 'gelu':
        activation = gelu
    else:
        raise ValueError(
            'Unsupported activation type {}'.format(activation_type)
        )

    output = inp
    with tf.compat.v1.variable_scope(scope, reuse = reuse):
        output = tf.compat.v1.layers.dense(
            output,
            d_inner,
            activation = activation,
            kernel_initializer = kernel_initializer,
            name = 'layer_1',
        )
        output = tf.compat.v1.layers.dropout(
            output, dropout, training = is_training, name = 'drop_1'
        )
        output = tf.compat.v1.layers.dense(
            output,
            d_model,
            kernel_initializer = kernel_initializer,
            name = 'layer_2',
        )
        output = tf.compat.v1.layers.dropout(
            output, dropout, training = is_training, name = 'drop_2'
        )
        output = tf.keras.layers.LayerNormalization(
            output + inp, begin_norm_axis = -1, scope = 'LayerNorm'
        )
    return output


def head_projection(h, d_model, n_head, d_head, kernel_initializer, name):
    """Project hidden states to a specific head with a 4D-shape."""
    proj_weight = tf.compat.v1.get_variable(
        '{}/kernel'.format(name),
        [d_model, n_head, d_head],
        dtype = h.dtype,
        initializer = kernel_initializer,
    )
    head = tf.compat.v1.einsum('ibh,hnd->ibnd', h, proj_weight)

    return head


def post_attention(
    h,
    attn_vec,
    d_model,
    n_head,
    d_head,
    dropout,
    is_training,
    kernel_initializer,
    residual = True,
):
    """Post-attention processing."""
    # post-attention projection (back to `d_model`)
    proj_o = tf.compat.v1.get_variable(
        'o/kernel',
        [d_model, n_head, d_head],
        dtype = h.dtype,
        initializer = kernel_initializer,
    )
    attn_out = tf.compat.v1.einsum('ibnd,hnd->ibh', attn_vec, proj_o)

    attn_out = tf.compat.v1.layers.dropout(attn_out, dropout, training = is_training)
    if residual:
        output = tf.keras.layers.LayerNormalization(
            attn_out + h, begin_norm_axis = -1, scope = 'LayerNorm'
        )
    else:
        output = tf.keras.layers.LayerNormalization(
            attn_out, begin_norm_axis = -1, scope = 'LayerNorm'
        )

    return output


def abs_attn_core(
    q_head, k_head, v_head, attn_mask, dropatt, is_training, scale
):
    """Core absolute positional attention operations."""

    attn_score = tf.compat.v1.einsum('ibnd,jbnd->ijbn', q_head, k_head)
    attn_score *= scale
    if attn_mask is not None:
        attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = tf.compat.v1.nn.softmax(attn_score, 1)
    attn_prob = tf.compat.v1.layers.dropout(attn_prob, dropatt, training = is_training)

    # attention output
    attn_vec = tf.compat.v1.einsum('ijbn,jbnd->ibnd', attn_prob, v_head)

    return attn_vec


def rel_attn_core(
    q_head,
    k_head_h,
    v_head_h,
    k_head_r,
    seg_embed,
    seg_mat,
    r_w_bias,
    r_r_bias,
    r_s_bias,
    attn_mask,
    dropatt,
    is_training,
    scale,
):
    """Core relative positional attention operations."""

    # content based attention score
    ac = tf.compat.v1.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)

    # position based attention score
    bd = tf.compat.v1.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
    bd = rel_shift(bd, klen = tf.compat.v1.shape(ac)[1])

    # segment based attention score
    if seg_mat is None:
        ef = 0
    else:
        ef = tf.compat.v1.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed)
        ef = tf.compat.v1.einsum('ijbs,ibns->ijbn', seg_mat, ef)

    # merge attention scores and perform masking
    attn_score = (ac + bd + ef) * scale
    if attn_mask is not None:
        # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
        attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = tf.compat.v1.nn.softmax(attn_score, 1)
    attn_prob = tf.compat.v1.layers.dropout(attn_prob, dropatt, training = is_training)

    # attention output
    attn_vec = tf.compat.v1.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

    return attn_vec


def rel_shift(x, klen = -1):
    """perform relative shift to form the relative attention score."""
    x_size = tf.compat.v1.shape(x)

    x = tf.compat.v1.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
    x = tf.compat.v1.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.compat.v1.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
    x = tf.compat.v1.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

    return x


def _create_mask(qlen, mlen, dtype = tf.compat.v1.float32, same_length = False):
    """create causal attention mask."""
    attn_mask = tf.compat.v1.ones([qlen, qlen], dtype = dtype)
    mask_u = tf.compat.v1.matrix_band_part(attn_mask, 0, -1)
    mask_dia = tf.compat.v1.matrix_band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.compat.v1.zeros([qlen, mlen], dtype = dtype)
    ret = tf.compat.v1.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.compat.v1.matrix_band_part(attn_mask, -1, 0)
        ret = tf.compat.v1.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)

    return ret


def _cache_mem(curr_out, prev_mem, mem_len, reuse_len = None):
    """cache hidden states into memory."""
    if mem_len is None or mem_len == 0:
        return None
    else:
        if reuse_len is not None and reuse_len > 0:
            curr_out = curr_out[:reuse_len]

        if prev_mem is None:
            new_mem = curr_out[-mem_len:]
        else:
            new_mem = tf.compat.v1.concat([prev_mem, curr_out], 0)[-mem_len:]

    return tf.compat.v1.stop_gradient(new_mem)


def relative_positional_encoding(
    qlen, klen, d_model, clamp_len, attn_type, bi_data, bsz = None, dtype = None
):
    """create relative positional encoding."""
    freq_seq = tf.compat.v1.range(0, d_model, 2.0)
    if dtype is not None and dtype != tf.compat.v1.float32:
        freq_seq = tf.compat.v1.cast(freq_seq, dtype = dtype)
    inv_freq = 1 / (10000 ** (freq_seq / d_model))

    if attn_type == 'bi':
        # beg, end = klen - 1, -qlen
        beg, end = klen, -qlen
    elif attn_type == 'uni':
        # beg, end = klen - 1, -1
        beg, end = klen, -1
    else:
        raise ValueError('Unknown `attn_type` {}.'.format(attn_type))

    if bi_data:
        fwd_pos_seq = tf.compat.v1.range(beg, end, -1.0)
        bwd_pos_seq = tf.compat.v1.range(-beg, -end, 1.0)

        if dtype is not None and dtype != tf.compat.v1.float32:
            fwd_pos_seq = tf.compat.v1.cast(fwd_pos_seq, dtype = dtype)
            bwd_pos_seq = tf.compat.v1.cast(bwd_pos_seq, dtype = dtype)

        if clamp_len > 0:
            fwd_pos_seq = tf.compat.v1.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
            bwd_pos_seq = tf.compat.v1.clip_by_value(bwd_pos_seq, -clamp_len, clamp_len)

        tf.compat.v1.logging.info('bsz here', bsz)
        if bsz is not None:
            # With bi_data, the batch size should be divisible by 2.
            # assert bsz % 2 == 0
            fwd_pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
            bwd_pos_emb = positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
        else:
            fwd_pos_emb = positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb = positional_embedding(bwd_pos_seq, inv_freq)

        pos_emb = tf.compat.v1.concat([fwd_pos_emb, bwd_pos_emb], axis = 1)
    else:
        fwd_pos_seq = tf.compat.v1.range(beg, end, -1.0)
        if dtype is not None and dtype != tf.compat.v1.float32:
            fwd_pos_seq = tf.compat.v1.cast(fwd_pos_seq, dtype = dtype)
        if clamp_len > 0:
            fwd_pos_seq = tf.compat.v1.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
        pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz)

    return pos_emb


def multihead_attn(
    q,
    k,
    v,
    attn_mask,
    d_model,
    n_head,
    d_head,
    dropout,
    dropatt,
    is_training,
    kernel_initializer,
    residual = True,
    scope = 'abs_attn',
    reuse = None,
):
    """Standard multi-head attention with absolute positional embedding."""

    scale = 1 / (d_head ** 0.5)
    with tf.compat.v1.variable_scope(scope, reuse = reuse):
        # attention heads
        q_head = head_projection(
            q, d_model, n_head, d_head, kernel_initializer, 'q'
        )
        k_head = head_projection(
            k, d_model, n_head, d_head, kernel_initializer, 'k'
        )
        v_head = head_projection(
            v, d_model, n_head, d_head, kernel_initializer, 'v'
        )

        # attention vector
        attn_vec = abs_attn_core(
            q_head, k_head, v_head, attn_mask, dropatt, is_training, scale
        )

        # post processing
        output = post_attention(
            v,
            attn_vec,
            d_model,
            n_head,
            d_head,
            dropout,
            is_training,
            kernel_initializer,
            residual,
        )

    return output


def rel_multihead_attn(
    h,
    r,
    r_w_bias,
    r_r_bias,
    seg_mat,
    r_s_bias,
    seg_embed,
    attn_mask,
    mems,
    d_model,
    n_head,
    d_head,
    dropout,
    dropatt,
    is_training,
    kernel_initializer,
    scope = 'rel_attn',
    reuse = None,
):
    """Multi-head attention with relative positional encoding."""

    scale = 1 / (d_head ** 0.5)
    with tf.compat.v1.variable_scope(scope, reuse = reuse):
        if mems is not None and mems.shape.ndims > 1:
            cat = tf.compat.v1.concat([mems, h], 0)
        else:
            cat = h

        # content heads
        q_head_h = head_projection(
            h, d_model, n_head, d_head, kernel_initializer, 'q'
        )
        k_head_h = head_projection(
            cat, d_model, n_head, d_head, kernel_initializer, 'k'
        )
        v_head_h = head_projection(
            cat, d_model, n_head, d_head, kernel_initializer, 'v'
        )

        # positional heads
        k_head_r = head_projection(
            r, d_model, n_head, d_head, kernel_initializer, 'r'
        )

        # core attention ops
        attn_vec = rel_attn_core(
            q_head_h,
            k_head_h,
            v_head_h,
            k_head_r,
            seg_embed,
            seg_mat,
            r_w_bias,
            r_r_bias,
            r_s_bias,
            attn_mask,
            dropatt,
            is_training,
            scale,
        )

        # post processing
        output = post_attention(
            h,
            attn_vec,
            d_model,
            n_head,
            d_head,
            dropout,
            is_training,
            kernel_initializer,
        )

    return output


def two_stream_rel_attn(
    h,
    g,
    r,
    mems,
    r_w_bias,
    r_r_bias,
    seg_mat,
    r_s_bias,
    seg_embed,
    attn_mask_h,
    attn_mask_g,
    target_mapping,
    d_model,
    n_head,
    d_head,
    dropout,
    dropatt,
    is_training,
    kernel_initializer,
    scope = 'rel_attn',
):
    """Two-stream attention with relative positional encoding."""

    scale = 1 / (d_head ** 0.5)
    with tf.compat.v1.variable_scope(scope, reuse = False):

        # content based attention score
        if mems is not None and mems.shape.ndims > 1:
            cat = tf.compat.v1.concat([mems, h], 0)
        else:
            cat = h

        # content-based key head
        k_head_h = head_projection(
            cat, d_model, n_head, d_head, kernel_initializer, 'k'
        )

        # content-based value head
        v_head_h = head_projection(
            cat, d_model, n_head, d_head, kernel_initializer, 'v'
        )

        # position-based key head
        k_head_r = head_projection(
            r, d_model, n_head, d_head, kernel_initializer, 'r'
        )

        ##### h-stream
        # content-stream query head
        q_head_h = head_projection(
            h, d_model, n_head, d_head, kernel_initializer, 'q'
        )

        # core attention ops
        attn_vec_h = rel_attn_core(
            q_head_h,
            k_head_h,
            v_head_h,
            k_head_r,
            seg_embed,
            seg_mat,
            r_w_bias,
            r_r_bias,
            r_s_bias,
            attn_mask_h,
            dropatt,
            is_training,
            scale,
        )

        # post processing
        output_h = post_attention(
            h,
            attn_vec_h,
            d_model,
            n_head,
            d_head,
            dropout,
            is_training,
            kernel_initializer,
        )

    with tf.compat.v1.variable_scope(scope, reuse = True):
        ##### g-stream
        # query-stream query head
        q_head_g = head_projection(
            g, d_model, n_head, d_head, kernel_initializer, 'q'
        )

        # core attention ops
        if target_mapping is not None:
            q_head_g = tf.compat.v1.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
            attn_vec_g = rel_attn_core(
                q_head_g,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_embed,
                seg_mat,
                r_w_bias,
                r_r_bias,
                r_s_bias,
                attn_mask_g,
                dropatt,
                is_training,
                scale,
            )
            attn_vec_g = tf.compat.v1.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
        else:
            attn_vec_g = rel_attn_core(
                q_head_g,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_embed,
                seg_mat,
                r_w_bias,
                r_r_bias,
                r_s_bias,
                attn_mask_g,
                dropatt,
                is_training,
                scale,
            )

        # post processing
        output_g = post_attention(
            g,
            attn_vec_g,
            d_model,
            n_head,
            d_head,
            dropout,
            is_training,
            kernel_initializer,
        )

        return output_h, output_g


def transformer_xl(
    inp_k,
    n_token,
    n_layer,
    d_model,
    n_head,
    d_head,
    d_inner,
    dropout,
    dropatt,
    attn_type,
    bi_data,
    initializer,
    is_training,
    mem_len = None,
    inp_q = None,
    mems = None,
    same_length = False,
    clamp_len = -1,
    untie_r = False,
    use_tpu = True,
    input_mask = None,
    perm_mask = None,
    seg_id = None,
    reuse_len = None,
    ff_activation = 'relu',
    target_mapping = None,
    use_bfloat16 = False,
    scope = 'transformer',
    **kwargs
):
    """
    Defines a Transformer-XL computation graph with additional
    support for XLNet.

    Args:

    inp_k: int32 Tensor in shape [len, bsz], the input token IDs.
    seg_id: int32 Tensor in shape [len, bsz], the input segment IDs.
    input_mask: float32 Tensor in shape [len, bsz], the input mask.
      0 for real tokens and 1 for padding.
    mems: a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
      from previous batches. The length of the list equals n_layer.
      If None, no memory is used.
    perm_mask: float32 Tensor in shape [len, len, bsz].
      If perm_mask[i, j, k] = 0, i attend to j in batch k;
      if perm_mask[i, j, k] = 1, i does not attend to j in batch k.
      If None, each position attends to all the others.
    target_mapping: float32 Tensor in shape [num_predict, len, bsz].
      If target_mapping[i, j, k] = 1, the i-th predict in batch k is
      on the j-th token.
      Only used during pretraining for partial prediction.
      Set to None during finetuning.
    inp_q: float32 Tensor in shape [len, bsz].
      1 for tokens with losses and 0 for tokens without losses.
      Only used during pretraining for two-stream attention.
      Set to None during finetuning.

    n_layer: int, the number of layers.
    d_model: int, the hidden size.
    n_head: int, the number of attention heads.
    d_head: int, the dimension size of each attention head.
    d_inner: int, the hidden size in feed-forward layers.
    ff_activation: str, "relu" or "gelu".
    untie_r: bool, whether to untie the biases in attention.
    n_token: int, the vocab size.

    is_training: bool, whether in training mode.
    use_tpu: bool, whether TPUs are used.
    use_bfloat16: bool, use bfloat16 instead of float32.
    dropout: float, dropout rate.
    dropatt: float, dropout rate on attention probabilities.
    init: str, the initialization scheme, either "normal" or "uniform".
    init_range: float, initialize the parameters with a uniform distribution
      in [-init_range, init_range]. Only effective when init="uniform".
    init_std: float, initialize the parameters with a normal distribution
      with mean 0 and stddev init_std. Only effective when init="normal".
    mem_len: int, the number of tokens to cache.
    reuse_len: int, the number of tokens in the currect batch to be cached
      and reused in the future.
    bi_data: bool, whether to use bidirectional input pipeline.
      Usually set to True during pretraining and False during finetuning.
    clamp_len: int, clamp all relative distances larger than clamp_len.
      -1 means no clamping.
    same_length: bool, whether to use the same attention length for each token.
    summary_type: str, "last", "first", "mean", or "attn". The method
      to pool the input to get a vector representation.
    initializer: A tf initializer.
    scope: scope name for the computation graph.
  """
    tf.compat.v1.logging.info('memory input {}'.format(mems))
    tf_float = tf.compat.v1.bfloat16 if use_bfloat16 else tf.compat.v1.float32
    tf.compat.v1.logging.info('Use float type {}'.format(tf_float))

    new_mems = []
    with tf.compat.v1.variable_scope(scope):
        if untie_r:
            r_w_bias = tf.compat.v1.get_variable(
                'r_w_bias',
                [n_layer, n_head, d_head],
                dtype = tf_float,
                initializer = initializer,
            )
            r_r_bias = tf.compat.v1.get_variable(
                'r_r_bias',
                [n_layer, n_head, d_head],
                dtype = tf_float,
                initializer = initializer,
            )
        else:
            r_w_bias = tf.compat.v1.get_variable(
                'r_w_bias',
                [n_head, d_head],
                dtype = tf_float,
                initializer = initializer,
            )
            r_r_bias = tf.compat.v1.get_variable(
                'r_r_bias',
                [n_head, d_head],
                dtype = tf_float,
                initializer = initializer,
            )

        bsz = tf.compat.v1.shape(inp_k)[1]
        qlen = tf.compat.v1.shape(inp_k)[0]
        mlen = tf.compat.v1.shape(mems[0])[0] if mems is not None else 0
        klen = mlen + qlen

        ##### Attention mask
        # causal attention mask
        if attn_type == 'uni':
            attn_mask = _create_mask(qlen, mlen, tf_float, same_length)
            attn_mask = attn_mask[:, :, None, None]
        elif attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError('Unsupported attention type: {}'.format(attn_type))

        # data mask: input mask & perm mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            mems_mask = tf.compat.v1.zeros(
                [tf.compat.v1.shape(data_mask)[0], mlen, bsz], dtype = tf_float
            )
            data_mask = tf.compat.v1.concat([mems_mask, data_mask], 1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = tf.compat.v1.cast(attn_mask > 0, dtype = tf_float)

        if attn_mask is not None:
            non_tgt_mask = -tf.compat.v1.eye(qlen, dtype = tf_float)
            non_tgt_mask = tf.compat.v1.concat(
                [tf.compat.v1.zeros([qlen, mlen], dtype = tf_float), non_tgt_mask],
                axis = -1,
            )
            non_tgt_mask = tf.compat.v1.cast(
                (attn_mask + non_tgt_mask[:, :, None, None]) > 0,
                dtype = tf_float,
            )
        else:
            non_tgt_mask = None

        ##### Word embedding
        word_emb_k, lookup_table, lookup_table_2 = embedding_lookup(
            x = inp_k,
            n_token = n_token,
            d_embed = 128,
            hidden_size = d_model,
            initializer = initializer,
            use_tpu = use_tpu,
            dtype = tf_float,
            scope = 'word_embedding',
        )

        if inp_q is not None:
            with tf.compat.v1.variable_scope('mask_emb'):
                mask_emb = tf.compat.v1.get_variable(
                    'mask_emb', [1, 1, d_model], dtype = tf_float
                )
                if target_mapping is not None:
                    word_emb_q = tf.compat.v1.tile(
                        mask_emb, [tf.compat.v1.shape(target_mapping)[0], bsz, 1]
                    )
                else:
                    inp_q_ext = inp_q[:, :, None]
                    word_emb_q = (
                        inp_q_ext * mask_emb + (1 - inp_q_ext) * word_emb_k
                    )
        output_h = tf.compat.v1.layers.dropout(
            word_emb_k, dropout, training = is_training
        )
        if inp_q is not None:
            output_g = tf.compat.v1.layers.dropout(
                word_emb_q, dropout, training = is_training
            )

        ##### Segment embedding
        if seg_id is not None:
            if untie_r:
                r_s_bias = tf.compat.v1.get_variable(
                    'r_s_bias',
                    [n_layer, n_head, d_head],
                    dtype = tf_float,
                    initializer = initializer,
                )
            else:
                # default case (tie)
                r_s_bias = tf.compat.v1.get_variable(
                    'r_s_bias',
                    [n_head, d_head],
                    dtype = tf_float,
                    initializer = initializer,
                )

            seg_embed = tf.compat.v1.get_variable(
                'seg_embed',
                [n_layer, 2, n_head, d_head],
                dtype = tf_float,
                initializer = initializer,
            )

            # Convert `seg_id` to one-hot `seg_mat`
            mem_pad = tf.compat.v1.zeros([mlen, bsz], dtype = tf.compat.v1.int32)
            cat_ids = tf.compat.v1.concat([mem_pad, seg_id], 0)

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = tf.compat.v1.cast(
                tf.compat.v1.logical_not(tf.compat.v1.equal(seg_id[:, None], cat_ids[None, :])),
                tf.compat.v1.int32,
            )
            seg_mat = tf.compat.v1.one_hot(seg_mat, 2, dtype = tf_float)
        else:
            seg_mat = None

        ##### Positional encoding
        pos_emb = relative_positional_encoding(
            qlen,
            klen,
            d_model,
            clamp_len,
            attn_type,
            bi_data,
            bsz = bsz,
            dtype = tf_float,
        )
        pos_emb = tf.compat.v1.layers.dropout(pos_emb, dropout, training = is_training)

        ##### Attention layers
        if mems is None:
            mems = [None] * n_layer

        name_variable_scope = 'layer_shared'

        for i in range(n_layer):
            # cache new mems
            new_mems.append(_cache_mem(output_h, mems[i], mem_len, reuse_len))

            # segment bias
            if seg_id is None:
                r_s_bias_i = None
                seg_embed_i = None
            else:
                r_s_bias_i = r_s_bias if not untie_r else r_s_bias[i]
                seg_embed_i = seg_embed[i]

            with tf.compat.v1.variable_scope(
                name_variable_scope, reuse = True if i > 0 else False
            ):
                if inp_q is not None:
                    output_h, output_g = two_stream_rel_attn(
                        h = output_h,
                        g = output_g,
                        r = pos_emb,
                        r_w_bias = r_w_bias if not untie_r else r_w_bias[i],
                        r_r_bias = r_r_bias if not untie_r else r_r_bias[i],
                        seg_mat = seg_mat,
                        r_s_bias = r_s_bias_i,
                        seg_embed = seg_embed_i,
                        attn_mask_h = non_tgt_mask,
                        attn_mask_g = attn_mask,
                        mems = mems[i],
                        target_mapping = target_mapping,
                        d_model = d_model,
                        n_head = n_head,
                        d_head = d_head,
                        dropout = dropout,
                        dropatt = dropatt,
                        is_training = is_training,
                        kernel_initializer = initializer,
                    )
                    reuse = True
                else:
                    reuse = False

                    output_h = rel_multihead_attn(
                        h = output_h,
                        r = pos_emb,
                        r_w_bias = r_w_bias if not untie_r else r_w_bias[i],
                        r_r_bias = r_r_bias if not untie_r else r_r_bias[i],
                        seg_mat = seg_mat,
                        r_s_bias = r_s_bias_i,
                        seg_embed = seg_embed_i,
                        attn_mask = non_tgt_mask,
                        mems = mems[i],
                        d_model = d_model,
                        n_head = n_head,
                        d_head = d_head,
                        dropout = dropout,
                        dropatt = dropatt,
                        is_training = is_training,
                        kernel_initializer = initializer,
                        reuse = reuse,
                    )

                if inp_q is not None:
                    output_g = positionwise_ffn(
                        inp = output_g,
                        d_model = d_model,
                        d_inner = d_inner,
                        dropout = dropout,
                        kernel_initializer = initializer,
                        activation_type = ff_activation,
                        is_training = is_training,
                    )

                output_h = positionwise_ffn(
                    inp = output_h,
                    d_model = d_model,
                    d_inner = d_inner,
                    dropout = dropout,
                    kernel_initializer = initializer,
                    activation_type = ff_activation,
                    is_training = is_training,
                    reuse = reuse,
                )

        if inp_q is not None:
            output = tf.compat.v1.layers.dropout(
                output_g, dropout, training = is_training
            )
        else:
            output = tf.compat.v1.layers.dropout(
                output_h, dropout, training = is_training
            )

        return output, new_mems, lookup_table, lookup_table_2


def lm_loss(
    hidden,
    target,
    n_token,
    d_model,
    initializer,
    lookup_table = None,
    tie_weight = False,
    bi_data = True,
    use_tpu = False,
):
    """doc."""

    with tf.compat.v1.variable_scope('lm_loss'):
        if tie_weight:
            assert (
                lookup_table is not None
            ), 'lookup_table cannot be None for tie_weight'
            softmax_w = lookup_table
        else:
            softmax_w = tf.compat.v1.get_variable(
                'weight',
                [n_token, d_model],
                dtype = hidden.dtype,
                initializer = initializer,
            )

        softmax_b = tf.compat.v1.get_variable(
            'bias',
            [n_token],
            dtype = hidden.dtype,
            initializer = tf.compat.v1.zeros_initializer(),
        )

        logits = tf.compat.v1.einsum('ibd,nd->ibn', hidden, softmax_w) + softmax_b

        if use_tpu:
            one_hot_target = tf.compat.v1.one_hot(target, n_token, dtype = logits.dtype)
            loss = -tf.compat.v1.reduce_sum(
                tf.compat.v1.nn.log_softmax(logits) * one_hot_target, -1
            )
        else:
            loss = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
                labels = target, logits = logits
            )

        return loss


def lm_accuracy(
    hidden,
    target,
    n_token,
    d_model,
    initializer,
    lookup_table = None,
    lookup_table_2 = None,
    tie_weight = False,
    bi_data = True,
    use_tpu = False,
):
    """doc."""

    with tf.compat.v1.variable_scope('lm_loss'):
        if tie_weight:
            assert (
                lookup_table is not None
            ), 'lookup_table cannot be None for tie_weight'
            softmax_w = lookup_table
        else:
            softmax_w = tf.compat.v1.get_variable(
                'weight',
                [n_token, d_model],
                dtype = hidden.dtype,
                initializer = initializer,
            )

        softmax_b = tf.compat.v1.get_variable(
            'bias',
            [n_token],
            dtype = hidden.dtype,
            initializer = tf.compat.v1.zeros_initializer(),
        )
        softmax_w = tf.compat.v1.matmul(softmax_w, lookup_table_2)

        logits = tf.compat.v1.einsum('ibd,nd->ibn', hidden, softmax_w) + softmax_b
        next_sentence_predictions = tf.compat.v1.argmax(
            logits, axis = -1, output_type = tf.compat.v1.int32
        )
        print(target, next_sentence_predictions)
        next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
            labels = target, predictions = next_sentence_predictions
        )

        if use_tpu:
            one_hot_target = tf.compat.v1.one_hot(target, n_token, dtype = logits.dtype)
            loss = -tf.compat.v1.reduce_sum(
                tf.compat.v1.nn.log_softmax(logits) * one_hot_target, -1
            )
        else:
            loss = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
                labels = target, logits = logits
            )

        return next_sentence_accuracy, loss


def summarize_sequence(
    summary_type,
    hidden,
    d_model,
    n_head,
    d_head,
    dropout,
    dropatt,
    input_mask,
    is_training,
    initializer,
    scope = None,
    reuse = None,
    use_proj = True,
):

    """
      Different classification tasks may not may not share the same parameters
      to summarize the sequence features.

      If shared, one can keep the `scope` to the default value `None`.
      Otherwise, one should specify a different `scope` for each task.
  """

    with tf.compat.v1.variable_scope(scope, 'sequnece_summary', reuse = reuse):
        if summary_type == 'last':
            summary = hidden[-1]
        elif summary_type == 'first':
            summary = hidden[0]
        elif summary_type == 'mean':
            summary = tf.compat.v1.reduce_mean(hidden, axis = 0)
        elif summary_type == 'attn':
            bsz = tf.compat.v1.shape(hidden)[1]

            summary_bias = tf.compat.v1.get_variable(
                'summary_bias',
                [d_model],
                dtype = hidden.dtype,
                initializer = initializer,
            )
            summary_bias = tf.compat.v1.tile(summary_bias[None, None], [1, bsz, 1])

            if input_mask is not None:
                input_mask = input_mask[None, :, :, None]

            summary = multihead_attn(
                summary_bias,
                hidden,
                hidden,
                input_mask,
                d_model,
                n_head,
                d_head,
                dropout,
                dropatt,
                is_training,
                initializer,
                residual = False,
            )
            summary = summary[0]
        else:
            raise ValueError('Unsupported summary type {}'.format(summary_type))

        # use another projection as in BERT
        if use_proj:
            summary = tf.compat.v1.layers.dense(
                summary,
                d_model,
                activation = tf.compat.v1.tanh,
                kernel_initializer = initializer,
                name = 'summary',
            )

        # dropout
        summary = tf.compat.v1.layers.dropout(
            summary, dropout, training = is_training, name = 'dropout'
        )

    return summary


def classification_loss(
    hidden,
    labels,
    n_class,
    initializer,
    scope,
    reuse = None,
    return_logits = False,
):
    """
      Different classification tasks should use different scope names to ensure
      different dense layers (parameters) are used to produce the logits.

      An exception will be in transfer learning, where one hopes to transfer
      the classification weights.
  """

    with tf.compat.v1.variable_scope(scope, reuse = reuse):
        logits = tf.compat.v1.layers.dense(
            hidden, n_class, kernel_initializer = initializer, name = 'logit'
        )

        one_hot_target = tf.compat.v1.one_hot(labels, n_class, dtype = hidden.dtype)
        loss = -tf.compat.v1.reduce_sum(tf.compat.v1.nn.log_softmax(logits) * one_hot_target, -1)

        if return_logits:
            return loss, logits

        return loss


def regression_loss(
    hidden, labels, initializer, scope, reuse = None, return_logits = False
):
    with tf.compat.v1.variable_scope(scope, reuse = reuse):
        logits = tf.compat.v1.layers.dense(
            hidden, 1, kernel_initializer = initializer, name = 'logit'
        )

        logits = tf.compat.v1.squeeze(logits, axis = -1)
        loss = tf.compat.v1.square(logits - labels)

        if return_logits:
            return loss, logits

        return loss
