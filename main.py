# main.py

# ── Disable eager execution BEFORE anything else ──
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import os
from os.path import join as pjoin
import argparse

# GPU config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# (we don't actually need to keep a Session here; trainer.py will open its own)
_ = tf.compat.v1.Session(config=config)

# Your graph & data utils
from utils.math_graph import weight_matrix, scaled_laplacian, cheb_poly_approx
from data_loader.data_utils import data_gen
from models.trainer import model_train
from models.tester import model_test


# Arg parsing
parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=228)
parser.add_argument('--n_his',   type=int, default=12)
parser.add_argument('--n_pred',  type=int, default=9)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch',     type=int, default=50)
parser.add_argument('--save',      type=int, default=10)
parser.add_argument('--ks',        type=int, default=3)
parser.add_argument('--kt',        type=int, default=3)
parser.add_argument('--lr',        type=float, default=1e-3)
parser.add_argument('--opt',       type=str, default='RMSProp')
parser.add_argument('--graph',     type=str, default='default')
parser.add_argument('--inf_mode',  type=str, default='merge')
args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
blocks = [[1, 32, 64], [64, 32, 128]]


# Load adjacency & register for GAT

if args.graph == 'default':
    adj_path = pjoin('./dataset/PeMSD7_Full', f'PeMSD7_W_{n}.csv')
else:
    adj_path = args.graph

if not os.path.exists(adj_path):
    raise FileNotFoundError(f'Adjacency file not found: {adj_path}')

W = weight_matrix(adj_path)

# Store a binary mask for graph‑attention
tf.compat.v1.add_to_collection(
    'adjacency',
    tf.constant((W > 0).astype('float32'))
)


# Build graph kernel for Chebyshev GCN

L   = scaled_laplacian(W)
Lk  = cheb_poly_approx(L, Ks, n)
tf.compat.v1.add_to_collection(
    name='graph_kernel',
    value=tf.cast(tf.constant(Lk), tf.float32)
)

# Load & preprocess data

data_path = pjoin('./dataset/PeMSD7_Full', f'PeMSD7_V_{n}.csv')
if not os.path.exists(data_path):
    raise FileNotFoundError(f'Data file not found: {data_path}')

# split 34 train, 5 val, 5 test
PeMS = data_gen(data_path, (34, 5, 5), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')


# Train & test

if __name__ == '__main__':
    model_train(PeMS, blocks, args)
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)
