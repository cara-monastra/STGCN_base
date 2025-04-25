from data_loader.data_utils import gen_batch
from models.tester import model_inference
from models.base_model import build_model, model_save
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time


def model_train(inputs, blocks, args, sum_path='./output/tensorboard'):
    """
    Train the STGCN model with added attention mechanisms.
    :param inputs: Dataset instance providing training/validation/test data.
    :param blocks: list of channel configurations for each spatio-temporal block.
    :param args: argparse namespace containing hyperparameters.
    """
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, num_epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt

    # placeholders: input sequence length = history + one step target
    x = tf.compat.v1.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

    # build model graph & losses
    train_loss, y_pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    copy_loss = tf.compat.v1.add_n(tf.compat.v1.get_collection('copy_loss'))

    # TensorBoard summaries
    tf.compat.v1.summary.scalar('train_loss', train_loss)
    tf.compat.v1.summary.scalar('copy_loss', copy_loss)

    # learning rate schedule
    global_steps = tf.Variable(0, trainable=False, name='global_steps')
    len_train = inputs.get_len('train')
    epoch_steps = (len_train + batch_size - 1) // batch_size
    lr = tf.compat.v1.train.exponential_decay(
        args.lr, global_steps,
        decay_steps=5 * epoch_steps, decay_rate=0.7, staircase=True
    )
    tf.compat.v1.summary.scalar('learning_rate', lr)
    increment_step = tf.compat.v1.assign_add(global_steps, 1)

    with tf.control_dependencies([increment_step]):
        if opt == 'RMSProp':
            optimizer = tf.compat.v1.train.RMSPropOptimizer(lr)
        elif opt == 'ADAM':
            optimizer = tf.compat.v1.train.AdamOptimizer(lr)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')
        train_op = optimizer.minimize(train_loss)

    merged = tf.compat.v1.summary.merge_all()

    with tf.compat.v1.Session() as sess:
        writer = tf.compat.v1.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        sess.run(tf.compat.v1.global_variables_initializer())

        # initialize validation/test metrics
        if inf_mode == 'sep':
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_va_val = min_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_va_val = min_val = np.tile(np.array([4e1, 1e5, 1e5]), len(step_idx))
        else:
            raise ValueError(f'ERROR: inference mode "{inf_mode}" is not defined.')

        for epoch in range(num_epoch):
            start_time = time.time()
            for step, batch in enumerate(
                gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)
            ):
                # always feed only history + one target slice
                feed_x = batch[:, : n_his + 1, :, :]
                summary, _ = sess.run([merged, train_op],
                                      feed_dict={x: feed_x, keep_prob: 1.0})
                writer.add_summary(summary, epoch * epoch_steps + step)

                if step % 50 == 0:
                    tl, cl = sess.run([train_loss, copy_loss],
                                     feed_dict={x: feed_x, keep_prob: 1.0})
                    print(f'Epoch {epoch:2d}, Step {step:3d}: ' +
                          f'[train_loss={tl:.3f}, copy_loss={cl:.3f}]')

            print(f'Epoch {epoch:2d} Training Time {time.time() - start_time:.3f}s')

            # validation/inference after each epoch
            start_inf = time.time()
            min_va_val, min_val = model_inference(
                sess, y_pred, inputs, batch_size, n_his, n_pred,
                step_idx, min_va_val, min_val
            )

            for ix in tmp_idx:
                va = min_va_val[ix - 2: ix + 1]
                te = min_val[ix - 2: ix + 1]
                print(f'Time Step {ix + 1}: ' +
                      f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; ' +
                      f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; ' +
                      f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
            print(f'Epoch {epoch:2d} Inference Time {time.time() - start_inf:.3f}s')

            # checkpoint
            if (epoch + 1) % args.save == 0:
                model_save(sess, global_steps, 'STGCN')

        writer.close()
    print('Training model finished!')


