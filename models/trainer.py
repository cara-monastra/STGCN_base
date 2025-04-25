# models/trainer.py

from data_loader.data_utils import gen_batch
from models.tester        import model_inference
from models.base_model    import build_model, model_save
from os.path              import join as pjoin
import tensorflow as tf
import numpy as np
import time

def model_train(inputs, blocks, args, sum_path='./output/tensorboard'):
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt           = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt

    x = tf.compat.v1.placeholder(tf.float32, [None, n_his+1, n, 1], name='data_input')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    tf.compat.v1.summary.scalar('train_loss', train_loss)
    copy_loss = tf.compat.v1.add_n(tf.compat.v1.get_collection('copy_loss'))
    tf.compat.v1.summary.scalar('copy_loss', copy_loss)

    # learning rate schedule
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    epoch_step = (len_train + batch_size - 1)//batch_size
    lr = tf.compat.v1.train.exponential_decay(args.lr, global_steps,
                                              decay_steps=5*epoch_step,
                                              decay_rate=0.7, staircase=True)
    tf.compat.v1.summary.scalar('lr', lr)
    step_op = tf.compat.v1.assign_add(global_steps, 1)

    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            optimizer = tf.compat.v1.train.RMSPropOptimizer(lr)
        elif opt == 'ADAM':
            optimizer = tf.compat.v1.train.AdamOptimizer(lr)
        else:
            raise ValueError(f'Unknown optimizer "{opt}"')

        grads_and_vars = optimizer.compute_gradients(train_loss)
        grads, vars_    = zip(*grads_and_vars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
        train_op = optimizer.apply_gradients(zip(clipped_grads, vars_))

    merged = tf.compat.v1.summary.merge_all()

    with tf.compat.v1.Session() as sess:
        writer = tf.compat.v1.summary.FileWriter(pjoin(sum_path,'train'), sess.graph)
        sess.run(tf.compat.v1.global_variables_initializer())

        # set up inference tracking...
        # (unchanged up through training loop)

        for e in range(epoch):
            start = time.time()
            for j, x_batch in enumerate(
                gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)
            ):
                summary, _ = sess.run([merged, train_op],
                                      feed_dict={x: x_batch, keep_prob:0.9})
                writer.add_summary(summary, e*epoch_step + j)
                if j % 50 == 0:
                    tl, cl = sess.run([train_loss, copy_loss],
                                      {x: x_batch, keep_prob:1.0})
                    print(f'Epoch {e:2d}, Step {j:3d}: loss=[{tl:.3f}, {cl:.3f}]')
            print(f'Epoch {e:2d} train time: {time.time()-start:.3f}s')

            # validation & model_save logic unchanged...

        writer.close()
    print('Training complete.')

