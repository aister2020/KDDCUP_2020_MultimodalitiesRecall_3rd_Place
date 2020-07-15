# -*- coding: utf-8 -*-
import tensorflow as tf
import math

def multiplication_attention(query, doc, mask, name=None):
    query = tf.expand_dims(query, axis=1)
    query = tf.tile(query, [1, tf.shape(doc)[1], 1])
    enc = tf.concat([doc, query], axis=2)
    e = tf.layers.dense(enc, 1, kernel_initializer=tf.initializers.truncated_normal(0, 0.05), name='query_doc_' + name, reuse=True)
    mask = tf.expand_dim(mask, axis=1)
    e = tf.exp(e) * mask
    e = tf.reduce_sum(e, axis=1)

    weights = tf.nn.softmax(e, axis=1)

def attention_net_v2(enc, dec, bias=None, name="", scope="intra_session_attention_"):

    with tf.variable_scope(scope + name, reuse=tf.AUTO_REUSE):
        dec = tf.tile(tf.expand_dims(dec, axis=1), [1, tf.shape(enc)[1], 1])
        if bias is None:
            concate = tf.concat([enc, dec], axis=2)
        else:
            concate = tf.concat([enc, dec, bias], axis=2)
        e = tf.layers.dense(concate, 1, kernel_initializer=tf.initializers.truncated_normal(0, 0.05))
        weights = tf.nn.softmax(e, axis=1)
        return tf.squeeze(tf.matmul(tf.transpose(weights, [0, 2, 1]), enc), axis=1), weights


    
def tanh_sigmoid(inputs,units,activation=tf.nn.tanh):
    outputs = tf.layers.dense(inputs,units=units, activation=activation)
    gate = tf.layers.dense(inputs, units=units, activation=tf.nn.sigmoid)
    outputs = outputs * gate
    return outputs


def dot_attention_with_query(att_query,att_key,att_value,mask,activation=None,add_bias=True,scale=None,scale_dot=False):

    weights = tf.matmul(tf.expand_dims(att_query,axis=1),tf.transpose(att_key,[0,2,1]))
    if scale_dot:
        weights = weights/(att_query.get_shape().as_list()[-1]**0.5)
    
    if add_bias:
        
        bias = tf.get_variable('att_bias',[1],initializer=tf.zeros_initializer())
        weights = weights + bias

    if activation:
        weights = activation(weights)
    
    if scale is not None:
        weights = scale * weights
    
    score = tf.exp(weights)
    

    if mask is not None:
        mask = tf.expand_dims(mask,axis=1)
        score = score * mask

    score_sum = tf.reduce_sum(score,axis=2,keepdims=True) + 1e-10
    score = score/score_sum

    value = tf.squeeze(tf.matmul(score,att_value),axis=1)
    return value,score

def attention_with_query(att_query,att_key,att_value,mask,activation=None,add_bias=True,scale=None):
    
    
    att_query_tile = tf.expand_dims(att_query,axis=1)
    att_query_tile = tf.tile(att_query_tile,[1,tf.shape(att_key)[1],1])
    
    weights = tanh_sigmoid(tf.concat([att_query_tile,att_key],axis=-1),units=512)
    weights = tf.layers.dense(weights,units=1,activation=activation,use_bias=add_bias)
    weights = tf.transpose(weights,[0,2,1])
    
    # weights = tf.matmul(tf.expand_dims(att_query,axis=1),tf.transpose(att_key,[0,2,1]))#/(att_query.get_shape().as_list()[-1]**0.5)
    
    # if add_bias:
        
    #     bias = tf.get_variable('att_bias',[1],initializer=tf.zeros_initializer())
    #     weights = weights + bias

    # if activation:
    #     weights = activation(weights)
    
    # if scale is not None:
    #     weights = scale * weights
    
    score = tf.exp(weights)
    

    if mask is not None:
        mask = tf.expand_dims(mask,axis=1)
        score = score * mask

    score_sum = tf.reduce_sum(score,axis=2,keepdims=True) + 1e-10
    score = score/score_sum

    value = tf.squeeze(tf.matmul(score,att_value),axis=1)
    return value,score


def layer_norm_vars(units):
  """Create Variables for layer norm."""
  scale = tf.get_variable(
      "layer_norm_scale", [units], initializer=tf.ones_initializer())
  bias = tf.get_variable(
      "layer_norm_bias", [units], initializer=tf.zeros_initializer())
  return scale, bias


def conditional_layer_norm_vars(units, conditional_input):
    """Conditional Layer Norm"""
    scale = tf.layers.dense(conditional_input, units, use_bias=True, name='layer_norm_scale_dense')
    bias = tf.layers.dense(conditional_input, units, use_bias=True, name='layer_norm_scale_bias')
    return scale, bias


def layer_norm_compute(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def conditional_layer_norm(x, units=None, epsilon=1e-6, name=None, reuse=None, conditional_input=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    with tf.variable_scope(name, default_name="layer_norm", reuse=reuse):
    
        scale, bias = conditional_layer_norm_vars(units, conditional_input=conditional_input)

    return layer_norm_compute(x, epsilon, scale, bias)


def conditional_layer_norm_with_query(x, units=None, epsilon=1e-6, name=None, reuse=None, conditional_input=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    with tf.variable_scope(name, default_name="layer_norm", reuse=reuse):
    
        scale, bias = conditional_layer_norm_vars(units, conditional_input=conditional_input)
        scale = tf.expand_dims(scale, axis=1)
        bias = tf.expand_dims(bias, axis=1)
        
    return layer_norm_compute(x, epsilon, scale, bias)


def layer_norm(x, units=None, epsilon=1e-6, name=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    with tf.variable_scope(name, default_name="layer_norm",  reuse=reuse):
        scale, bias = layer_norm_vars(units)
        return layer_norm_compute(x, epsilon, scale, bias)



def attention(seq,mask,activation=None):
    weights = tf.layers.dense(seq,units=1,activation=activation)


    score = tf.exp(weights)
    score = tf.transpose(score,[0,2,1])

    if mask is not None:
        mask = tf.expand_dims(mask,axis=1)
        score = score * mask

    score_sum = tf.reduce_sum(score,axis=2,keepdims=True) + 1e-10
    score = score/score_sum

    value = tf.squeeze(tf.matmul(score,seq),axis=1)
    return value




def get_shape_list(tensor, expected_rank=None, name=None):
    if name is None:
      name = tensor.name
    
    shape = tensor.shape.as_list()
    
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
      if dim is None:
        non_static_indexes.append(index)
    
    if not non_static_indexes:
      return shape
    
    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
      shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
      raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                       (input_tensor.shape))
    if ndims == 2:
      return input_tensor
    
    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)

def dropout(input_tensor, dropout_prob):
    """Perform dropout.
    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).
    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
      return input_tensor
    
    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.
    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    
    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]
    
    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
    
    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
    
    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask
    
    return mask



def self_attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])
    
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
    
    if len(from_shape) != len(to_shape):
      raise ValueError(
          "The rank of `from_tensor` must match the rank of `to_tensor`.")
    
    if len(from_shape) == 3:
      batch_size = from_shape[0]
      from_seq_length = from_shape[1]
      to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
      if (batch_size is None or from_seq_length is None or to_seq_length is None):
        raise ValueError(
            "When passing in rank 2 tensors to attention_layer, the values "
            "for `batch_size`, `from_seq_length`, and `to_seq_length` "
            "must all be specified.")
    
    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    
    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)
    
    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))
    
    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))
    
    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))
    
    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)
    
    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)
    
    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,1.0 / math.sqrt(float(size_per_head)))
    
    if attention_mask is not None:
      # `attention_mask` = [B, 1, F, T]
      attention_mask = tf.expand_dims(attention_mask, axis=[1])
    
      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
    
      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder
    
    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)
    
    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
    
    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])
    
    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
    
    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)
    
    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    
    if do_return_2d_tensor:
      # `context_layer` = [B*F, N*H]
      context_layer = tf.reshape(
          context_layer,
          [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
      # `context_layer` = [B, F, N*H]
      context_layer = tf.reshape(
          context_layer,
          [batch_size, from_seq_length, num_attention_heads * size_per_head])
    
    return context_layer

def append_idx(name, i):
    return name + '_' + str(i)

image_feature_mean=[0]
image_feature_std=[0]
