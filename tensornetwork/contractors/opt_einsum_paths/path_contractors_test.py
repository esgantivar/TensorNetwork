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
from tensornetwork.contractors.opt_einsum_paths import path_contractors
#import tensorflow as tf
#tf.enable_v2_behavior()


@pytest.fixture(name="path_algorithm",
                params=["optimal", "branch", "greedy", "auto"])
def path_algorithm_fixture(request):
  return getattr(path_contractors, request.param)


def test_sanity_check(backend, path_algorithm):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.ones((2, 7, 11)))
  c = net.add_node(np.ones((7, 11, 13, 2)))
  d = net.add_node(np.eye(13))
  # pylint: disable=pointless-statement
  a[0] ^ b[0]
  b[1] ^ c[0]
  b[2] ^ c[1]
  c[2] ^ d[1]
  c[3] ^ a[1]
  final_node = path_algorithm(net).get_final_node()
  assert final_node.shape == (13,)


def test_trace_edge(backend, path_algorithm):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2, 2, 2, 2)))
  b = net.add_node(np.ones((2, 2, 2)))
  c = net.add_node(np.ones((2, 2, 2)))
  # pylint: disable=pointless-statement
  a[0] ^ a[1]
  a[2] ^ b[0]
  a[3] ^ c[0]
  b[1] ^ c[1]
  b[2] ^ c[2]
  node = path_algorithm(net).get_final_node()
  np.testing.assert_allclose(node.tensor, np.ones(2) * 32.0)


def test_disconnected_network(backend, path_algorithm):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([2, 2]))
  b = net.add_node(np.array([2, 2]))
  c = net.add_node(np.array([2, 2]))
  d = net.add_node(np.array([2, 2]))
  # pylint: disable=pointless-statement
  a[0] ^ b[0]
  c[0] ^ d[0]
  with pytest.raises(ValueError):
    net = path_algorithm(net)


def test_single_node(backend, path_algorithm):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2, 2)))
  # pylint: disable=pointless-statement
  a[0] ^ a[1]
  node = path_algorithm(net).get_final_node()
  np.testing.assert_allclose(node.tensor, np.ones(2) * 2.0)


def test_custom_sanity_check(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones(2))
  b = net.add_node(np.ones((2, 5)))
  # pylint: disable=pointless-statement
  a[0] ^ b[0]

  class PathOptimizer:

    def __call__(self, inputs, output, size_dict, memory_limit=None):
      return [(0, 1)]

  optimizer = PathOptimizer()
  final_node = path_contractors.custom(net, optimizer).get_final_node()
  np.testing.assert_allclose(final_node.tensor, np.ones(5) * 2.0)


def test_copy_node(backend, path_algorithm):
  net = tensornetwork.TensorNetwork(backend=backend)
  x = net.add_node(np.ones([3, 3]))
  y = net.add_node(np.ones([3, 3, 3]))
  c = net.add_node(tensornetwork.CopyNode(rank=2, dimension=3))
  x[0] ^ y[1]
  x[1] ^ c[0]
  y[2] ^ c[1]
  node = path_algorithm(net).get_final_node()
  np.testing.assert_allclose(node.tensor, 9 * np.ones(3))


def test_copy_with_dangling(backend, path_algorithm):
  net = tensornetwork.TensorNetwork(backend=backend)
  x = net.add_node(np.ones([3, 3]))
  y = net.add_node(np.ones([3, 3, 3]))
  c = net.add_node(tensornetwork.CopyNode(rank=3, dimension=3))
  x[0] ^ y[1]
  x[1] ^ c[0]
  y[2] ^ c[1]
  edge_order = [y[0], c[2]]
  node = path_algorithm(net, output_edge_order=edge_order).get_final_node()
  np.testing.assert_allclose(node.tensor, 3 * np.ones([3, 3]))


def test_multiple_copies(backend, path_algorithm):
  net = tensornetwork.TensorNetwork(backend=backend)
  x = net.add_node(np.ones([2, 2, 2]))
  y = net.add_node(np.ones([2, 2, 2]))
  a = net.add_node(tensornetwork.CopyNode(rank=3, dimension=2))
  b = net.add_node(tensornetwork.CopyNode(rank=2, dimension=2))
  x[0] ^ a[1]
  x[1] ^ y[1]
  y[0] ^ a[2]
  x[2] ^ b[0]
  y[2] ^ b[1]
  node = path_algorithm(net).get_final_node()
  np.testing.assert_allclose(node.tensor, 4 * np.ones(2))


def test_multiple_copies2(backend, path_algorithm):
  net = tensornetwork.TensorNetwork(backend=backend)
  x = net.add_node(np.ones([2, 2, 2, 2]))
  y = net.add_node(np.ones([2, 2, 2]))
  z = net.add_node(np.ones([2, 2]))
  a = net.add_node(tensornetwork.CopyNode(rank=4, dimension=2))
  b = net.add_node(tensornetwork.CopyNode(rank=2, dimension=2))
  x[0] ^ a[1]
  x[1] ^ y[1]
  y[0] ^ a[2]
  x[2] ^ b[0]
  y[2] ^ b[1]
  x[3] ^ z[0]
  z[1] ^ a[3]
  node = path_algorithm(net).get_final_node()
  np.testing.assert_allclose(node.tensor, 8 * np.ones(2))