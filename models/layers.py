import tensorflow as tf
import math

NUM_HEADS = 8


#Spectral Graph Convolution

    kernel = tf.compat.v1.get_collection('graph_kernel')[0]  # [N, Ks*N]
    n = tf.shape(kernel)[0]
    x_tmp = tf.reshape(tf.transpose(x, [0,2,1]), [-1, n])            # [B*c_in, N]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])  # [B, c_in, Ks, N]
    x_ker = tf.reshape(tf.transpose(x_mul, [0,3,1,2]), [-1, c_in*Ks])
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])    # [B, N, c_out]
    return x_gconv

#Layer Normalization
def layer_norm(x, scope):
    with tf.compat.v1.variable_scope(scope):
        mu, sigma = tf.nn.moments(x, axes=[-1], keepdims=True)
        C = x.get_shape().as_list()[-1]
        gamma = tf.compat.v1.get_variable('gamma', initializer=tf.ones([C]))
        beta  = tf.compat.v1.get_variable('beta',  initializer=tf.zeros([C]))
        x_norm = (x - mu) / tf.sqrt(sigma + 1e-6)
        return gamma * x_norm + beta

#Temporal Convolution (GLU or other activations)
def temporal_conv_layer(x, Kt, c_in, c_out, act_func='relu'):
    _, T, N, _ = x.get_shape().as_list()

    #residual channel match
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

#Multi-Head Temporal Self-Attention
def multihead_temporal_attention(x, d_model, num_heads, keep_prob, scope='temp_att'):
    with tf.compat.v1.variable_scope(scope):
        B = tf.shape(x)[0]; T = tf.shape(x)[1]; N = tf.shape(x)[2]
        C = x.get_shape().as_list()[3]
        depth = d_model // num_heads
        x_res = x

        flat = tf.reshape(x, [B*N, T, C])
        Q = tf.compat.v1.layers.dense(flat, d_model, name='q')
        K = tf.compat.v1.layers.dense(flat, d_model, name='k')
        V = tf.compat.v1.layers.dense(flat, d_model, name='v')

        def split_heads(t):
            t2 = tf.reshape(t, [B*N, T, num_heads, depth])
            return tf.transpose(t2, [0,2,1,3])  # [B*N, heads, T, depth]

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
        scores = tf.matmul(Qh, Kh, transpose_b=True) / tf.sqrt(tf.cast(depth,tf.float32)+1e-8)
        weights = tf.nn.softmax(scores, axis=-1)

        #sanitize any NaN/Inf before logging
        weights = tf.where(tf.math.is_finite(weights), weights, tf.zeros_like(weights))
        tf.compat.v1.summary.histogram(f'{scope}/weights', weights)

        attn = tf.matmul(weights, Vh)  # [B*N, heads, T, depth]
        attn = tf.transpose(attn, [0,2,1,3])           # [B*N, T, heads, depth]
        attn = tf.reshape(attn, [B*N, T, d_model])     # [B*N, T, d_model]
        attn = tf.reshape(attn, [B, N, T, d_model])    # [B, N, T, d_model]
        attn = tf.transpose(attn, [0,2,1,3])           # [B, T, N, d_model]

        out = layer_norm(attn + x_res, scope + '_ln')
        return tf.nn.dropout(out, keep_prob)


#Multi-Head Graph Attention (Masked)

def multihead_graph_attention(x, d_model, num_heads, keep_prob, scope='graph_att'):
    with tf.compat.v1.variable_scope(scope):
        B = tf.shape(x)[0]; T = tf.shape(x)[1]; N = tf.shape(x)[2]
        C = x.get_shape().as_list()[3]
        depth = d_model // num_heads
        x_res = x

        flat = tf.reshape(x, [B*T, N, C])
        Q = tf.compat.v1.layers.dense(flat, d_model, name='q')
        K = tf.compat.v1.layers.dense(flat, d_model, name='k')
        V = tf.compat.v1.layers.dense(flat, d_model, name='v')

        def split_heads(t):
            t2 = tf.reshape(t, [B*T, N, num_heads, depth])
            return tf.transpose(t2, [0,2,1,3])  # [B*T, heads, N, depth]

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
        scores = tf.matmul(Qh, Kh, transpose_b=True) / tf.sqrt(tf.cast(depth,tf.float32)+1e-8)

        A = tf.compat.v1.get_collection('adjacency')[0]  # [N,N]
        mask = tf.expand_dims(tf.expand_dims(A>0,0),1)   # [1,1,N,N]
        mask = tf.tile(mask, [B*T, num_heads, 1, 1])
        inf_mask = tf.fill(tf.shape(scores), float('-1e9'))
        scores = tf.where(mask, scores, inf_mask)

        weights = tf.nn.softmax(scores, axis=-1)
        weights = tf.where(tf.math.is_finite(weights), weights, tf.zeros_like(weights))
        tf.compat.v1.summary.histogram(f'{scope}/weights', weights)

        attn = tf.matmul(weights, Vh)  # [B*T, heads, N, depth]
        attn = tf.transpose(attn, [0,2,1,3])          # [B*T, N, heads, depth]
        attn = tf.reshape(attn, [B, T, N, d_model])   # [B, T, N, d_model]

        out = layer_norm(attn + x_res, scope + '_ln')
        return tf.nn.dropout(out, keep_prob)

#Spatio-Temporal Block: Graph Conv + Attention

def spatio_conv_layer(x, Ks, c_in, c_out, keep_prob):
    _, T, N, _ = x.get_shape().as_list()
    #residual channel match
    if c_in > c_out:
        w_in = tf.compat.v1.get_variable('ws_input', shape=[1,1,c_in,c_out])
        tf.compat.v1.add_to_collection('weight_decay', tf.nn.l2_loss(w_in))
        x_input = tf.nn.conv2d(x, w_in, [1,1,1,1], 'SAME')
    elif c_in < c_out:
        pad = c_out - c_in
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0],T,N,pad])], axis=-1)
    else:
        x_input = x

    #Cheb conv branch
    ws = tf.compat.v1.get_variable('ws', shape=[Ks*c_in, c_out])
    tf.compat.v1.add_to_collection('weight_decay', tf.nn.l2_loss(ws))
    variable_summaries(ws, 'theta')
    bs = tf.compat.v1.get_variable('bs', initializer=tf.zeros([c_out]))
    g = gconv(tf.reshape(x, [-1,N,c_in]), ws, Ks, c_in, c_out) + bs
    g = tf.reshape(g, [-1,T,N,c_out])

    #graph attention branch
    ga = multihead_graph_attention(x_input, d_model=c_out, num_heads=NUM_HEADS, keep_prob=keep_prob)

    fused = g + ga
    return tf.nn.relu(fused + x_input)

#Full ST-Conv Block (Temporal + Attention + Spatial)

def st_conv_block(x, Ks, Kt, channels, scope, keep_prob, act_func='GLU'):
    c_si, c_t, c_oo = channels
    with tf.compat.v1.variable_scope(f'stn_block_{scope}_in'):
        x_conv = temporal_conv_layer(x, Kt, c_si, c_t, act_func=act_func)
        x_attn = multihead_temporal_attention(x_conv, d_model=c_t,
                                              num_heads=NUM_HEADS,
                                              keep_prob=keep_prob)
        x_s = x_conv + x_attn
        x_t = spatio_conv_layer(x_s, Ks, c_t, c_t, keep_prob)
    with tf.compat.v1.variable_scope(f'stn_block_{scope}_out'):
        x_o = temporal_conv_layer(x_t, Kt, c_t, c_oo)
    x_ln = layer_norm(x_o, f'layer_norm_{scope}')
    return tf.nn.dropout(x_ln, keep_prob)

#Output and Helpers

def fully_con_layer(x, N, channel, scope):
    w = tf.compat.v1.get_variable(f'w_{scope}', shape=[1,1,channel,1])
    tf.compat.v1.add_to_collection('weight_decay', tf.nn.l2_loss(w))
    b = tf.compat.v1.get_variable(f'b_{scope}', initializer=tf.zeros([N,1]))
    return tf.nn.conv2d(x, w, [1,1,1,1], 'SAME') + b

def output_layer(x, T, scope, act_func='GLU'):
    _,_,N,channel = x.get_shape().as_list()
    with tf.compat.v1.variable_scope(f'{scope}_in'):
        x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    with tf.compat.v1.variable_scope(f'{scope}_out'):
        x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    return fully_con_layer(x_o, N, channel, scope)

# TensorBoard summaries for variables

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.scalar(f'mean/{name}', tf.reduce_mean(var))
        tf.summary.scalar(f'stddev/{name}', tf.math.reduce_std(var))
        tf.summary.scalar(f'max/{name}', tf.reduce_max(var))
        tf.summary.scalar(f'min/{name}', tf.reduce_min(var))
        tf.summary.histogram(name, var)
