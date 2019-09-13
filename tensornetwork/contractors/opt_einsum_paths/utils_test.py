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
import tensorflow as tf
tf.enable_v2_behavior()


def test_contract_between_copy(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  x = net.add_node(np.ones([3, 3]))
  y = net.add_node(np.ones([3, 3, 3]))
  c = net.add_node(tensornetwork.CopyNode(rank=2, dimension=3))
  x[0] ^ y[1]
  x[1] ^ c[0]
  y[2] ^ c[1]
  node = utils.contract_between_copy(net, c)
  np.testing.assert_allclose(node.tensor, 9 * np.ones(3))