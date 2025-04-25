import tensorflow as tf
import math


# Spectral Graph Convolution

def gconv(x, theta, Ks, c_in, c_out):
    kernel = tf.compat.v1.get_collection('graph_kernel')[0]  # [N, Ks*N]
    n = tf.shape(kernel)[0]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])        # [B*c_in, N]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n]) # [B, c_in, Ks, N]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])  # [B, N, c_out]
    return x_gconv

# Layer Normalization

def layer_norm(x, scope):
    with tf.compat.v1.variable_scope(scope):
        mu, sigma = tf.nn.moments(x, axes=[-1], keepdims=True)
        C = x.get_shape().as_list()[-1]
        gamma = tf.compat.v1.get_variable('gamma', initializer=tf.ones([C]))
        beta  = tf.compat.v1.get_variable('beta',  initializer=tf.zeros([C]))
        x_norm = (x - mu) / tf.sqrt(sigma + 1e-6)
        return gamma * x_norm + beta


# Temporal Convolution 

def temporal_conv_layer(x, Kt, c_in, c_out, act_func='relu'):
    _, T, N, _ = x.get_shape().as_list()
    # Residual channel match
    if c_in > c_out:
        w_in = tf.compat.v1.get_variable('wt_input', shape=[1,1,c_in,c_out])
        tf.compat.v1.add_to_collection('weight_decay', tf.nn.l2_loss(w_in))
        x_input = tf.nn.conv2d(x, w_in, [1,1,1,1], 'SAME')
    elif c_in < c_out:
        pad = c_out - c_in
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0],T,N,pad])], axis=-1)
    else:
        x_input = x
    x_input = x_input[:, Kt-1:T, :, :]

    if act_func == 'GLU':
        wt = tf.compat.v1.get_variable('wt', shape=[Kt,1,c_in,2*c_out])
        bt = tf.compat.v1.get_variable('bt', initializer=tf.zeros([2*c_out]))
        tf.compat.v1.add_to_collection('weight_decay', tf.nn.l2_loss(wt))
        conv = tf.nn.conv2d(x, wt, [1,1,1,1], 'VALID') + bt
        return (conv[..., :c_out] + x_input) * tf.nn.sigmoid(conv[..., c_out:])
    else:
        wt = tf.compat.v1.get_variable('wt', shape=[Kt,1,c_in,c_out])
        bt = tf.compat.v1.get_variable('bt', initializer=tf.zeros([c_out]))
        tf.compat.v1.add_to_collection('weight_decay', tf.nn.l2_loss(wt))
        conv = tf.nn.conv2d(x, wt, [1,1,1,1], 'VALID') + bt
        if act_func == 'linear':
            return conv
        if act_func == 'sigmoid':
            return tf.nn.sigmoid(conv)
        if act_func == 'relu':
            return tf.nn.relu(conv + x_input)
        raise ValueError(f'Unsupported act_func "{act_func}"')


# Temporal Self-Attention

def temporal_self_attention_layer(x, d_model, keep_prob, scope='temp_att'):
    with tf.compat.v1.variable_scope(scope):
        B = tf.shape(x)[0]; T = tf.shape(x)[1]; N = tf.shape(x)[2]
        C = x.get_shape().as_list()[3]
        x_in = x
        flat = tf.reshape(x, [B*N, T, C])
        Q = tf.compat.v1.layers.dense(flat, d_model, name='q')
        K = tf.compat.v1.layers.dense(flat, d_model, name='k')
        V = tf.compat.v1.layers.dense(flat, d_model, name='v')

        # stable scaled dot-product
        eps = 1e-8
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(d_model,tf.float32) + eps)
        scores = scores - tf.reduce_max(scores, axis=-1, keepdims=True)
        exp_s = tf.exp(scores)
        exp_s = exp_s + eps
        weights = exp_s / tf.reduce_sum(exp_s, axis=-1, keepdims=True)
        weights = tf.clip_by_value(weights, 1e-6, 1.0 - 1e-6)
        sanitized = tf.where(tf.math.is_finite(weights), weights, tf.zeros_like(weights))
        tf.compat.v1.summary.histogram(f'{scope}/weights', sanitized)

        attn = tf.matmul(sanitized, V)
        attn = tf.reshape(attn, [B, N, T, d_model])
        attn = tf.transpose(attn, [0,2,1,3])

        out = layer_norm(attn + x_in, scope+'_ln')
        return tf.nn.dropout(out, keep_prob)

# Graph Attention (Masked by adjacency)

def graph_attention_layer(x, d_model, keep_prob, scope='graph_att'):
    with tf.compat.v1.variable_scope(scope):
        B = tf.shape(x)[0]; T = tf.shape(x)[1]; N = tf.shape(x)[2]
        C = x.get_shape().as_list()[3]
        x_in = x
        flat = tf.reshape(x, [B*T, N, C])
        Q = tf.compat.v1.layers.dense(flat, d_model, name='q')
        K = tf.compat.v1.layers.dense(flat, d_model, name='k')
        V = tf.compat.v1.layers.dense(flat, d_model, name='v')

        eps = 1e-8
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(d_model,tf.float32) + eps)
        A = tf.compat.v1.get_collection('adjacency')[0]       # [N,N]
        mask = tf.tile(tf.expand_dims(A>0,0), [B*T,1,1])
        inf_mask = tf.fill(tf.shape(scores), -1e9)
        scores = tf.where(mask, scores, inf_mask)
        scores = scores - tf.reduce_max(scores, axis=-1, keepdims=True)
        exp_s = tf.exp(scores); exp_s = exp_s + eps
        weights = exp_s / tf.reduce_sum(exp_s, axis=-1, keepdims=True)
        weights = tf.clip_by_value(weights, 1e-6, 1.0 - 1e-6)
        sanitized = tf.where(tf.math.is_finite(weights), weights, tf.zeros_like(weights))
        tf.compat.v1.summary.histogram(f'{scope}/weights', sanitized)

        attn = tf.matmul(sanitized, V)
        attn = tf.reshape(attn, [B, T, N, d_model])

        out = layer_norm(attn + x_in, scope+'_ln')
        return tf.nn.dropout(out, keep_prob)


# Spatio-Temporal Block Combining Convs + Attention

def spatio_conv_layer(x, Ks, c_in, c_out, keep_prob):
    _, T, n, _ = x.get_shape().as_list()
    if c_in > c_out:
        w_in = tf.compat.v1.get_variable('ws_input', shape=[1, 1, c_in, c_out])
        tf.compat.v1.add_to_collection('weight_decay', tf.nn.l2_loss(w_in))
        x_input = tf.nn.conv2d(x, w_in, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=-1)
    else:
        x_input = x

    ws = tf.compat.v1.get_variable('ws', shape=[Ks * c_in, c_out])
    tf.compat.v1.add_to_collection('weight_decay', tf.nn.l2_loss(ws))
    variable_summaries(ws, 'theta')
    bs = tf.compat.v1.get_variable('bs', initializer=tf.zeros([c_out]))
    gconv_out = gconv(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs
    gconv_out = tf.reshape(gconv_out, [-1, T, n, c_out])

    ga_out = graph_attention_layer(x_input, d_model=c_out, keep_prob=keep_prob)

    fused = gconv_out + ga_out
    return tf.nn.relu(fused + x_input)


# Full Spatio-Temporal Convolutional Block

def st_conv_block(x, Ks, Kt, channels, scope, keep_prob, act_func='GLU'):
    c_si, c_t, c_oo = channels
    with tf.compat.v1.variable_scope(f'stn_block_{scope}_in'):
        x_conv = temporal_conv_layer(x, Kt, c_si, c_t, act_func=act_func)
        x_attn = temporal_self_attention_layer(x_conv, d_model=c_t, keep_prob=keep_prob)
        x_s = x_conv + x_attn
        x_t = spatio_conv_layer(x_s, Ks, c_t, c_t, keep_prob)
    with tf.compat.v1.variable_scope(f'stn_block_{scope}_out'):
        x_o = temporal_conv_layer(x_t, Kt, c_t, c_oo)
    x_ln = layer_norm(x_o, f'layer_norm_{scope}')
    return tf.nn.dropout(x_ln, keep_prob)


# Helper for TensorBoard summaries

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.scalar(f'mean/{name}', tf.reduce_mean(var))
        tf.summary.scalar(f'stddev/{name}', tf.math.reduce_std(var))
        tf.summary.scalar(f'max/{name}', tf.reduce_max(var))
        tf.summary.scalar(f'min/{name}', tf.reduce_min(var))
        tf.summary.histogram(name, var)

def fully_con_layer(x, n, channel, scope):
    w = tf.compat.v1.get_variable(name=f'w_{scope}',
                                  shape=[1, 1, channel, 1],
                                  dtype=tf.float32)
    tf.compat.v1.add_to_collection('weight_decay', tf.nn.l2_loss(w))
    b = tf.compat.v1.get_variable(name=f'b_{scope}',
                                  initializer=tf.zeros([n, 1]),
                                  dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b

def output_layer(x, T, scope, act_func='GLU'):
    _, _, n, channel = x.get_shape().as_list()
    with tf.compat.v1.variable_scope(f'{scope}_in'):
        x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    with tf.compat.v1.variable_scope(f'{scope}_out'):
        x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    x_fc = fully_con_layer(x_o, n, channel, scope)
    return x_fc
