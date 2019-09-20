# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import tensornetwork
from tensornetwork.contractors.opt_einsum_paths import utils
#import tensorflow as tf
#tf.enable_v2_behavior()


def test_find_copy_nodes(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones([2, 2]))
  b = net.add_node(np.ones([2, 2]))
  c = net.add_node(np.ones([2, 2, 2]))
  cn1 = net.add_node(tensornetwork.CopyNode(dimension=2, rank=2))
  cn2 = net.add_node(tensornetwork.CopyNode(dimension=2, rank=4))
  a[0] ^ cn1[0]
  b[0] ^ cn2[0]
  c[0] ^ a[1]
  c[1] ^ cn2[1]
  copy_neighbors, edge_map = utils.find_copy_nodes(net)

  assert copy_neighbors[cn1] == {a}
  assert copy_neighbors[cn2] == {b, c}

  edge_map_set = set(edge_map[edge] for edge in cn1.edges
                     if not edge.is_dangling())
  assert len(edge_map_set) == 1
  edge_map_set = set(edge_map[edge] for edge in cn2.edges
                     if not edge.is_dangling())
  assert len(edge_map_set) == 1


def test_disconnect_copy_edge(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones([2, 2]))
  b = net.add_node(np.ones([2, 2]))
  cn = net.add_node(tensornetwork.CopyNode(dimension=2, rank=2))
  a[0] ^ cn[0]
  b[0] ^ cn[1]
  node_edge, copy_edge = utils.disconnect_copy_edge(net, cn[1], b)
  assert node_edge is b[0]
  assert b[0].is_dangling()
  assert copy_edge is cn[1]
  assert cn[1].is_dangling()
  node_edge, copy_edge = utils.disconnect_copy_edge(net, cn[0], a)
  assert node_edge is a[0]
  assert a[0].is_dangling()
  assert copy_edge is cn[0]
  assert cn[0].is_dangling()


def test_isolate_copy_node(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones([2, 2]))
  b = net.add_node(np.ones([2, 2]))
  c = net.add_node(np.ones([2, 2]))
  cn = net.add_node(tensornetwork.CopyNode(dimension=2, rank=3))
  a[0] ^ cn[0]
  b[0] ^ cn[1]
  c[0] ^ cn[2]
  new_copy = utils.isolate_copy_node(net, cn, a, b)
  assert cn.rank == 2
  assert new_copy.rank == 3

  assert c[0] in cn.edges
  cn_edge = cn[0] if cn[0] != c[0] else cn[1]
  assert cn_edge in new_copy.edges
  assert a[0] in new_copy.edges
  assert b[0] in new_copy.edges
