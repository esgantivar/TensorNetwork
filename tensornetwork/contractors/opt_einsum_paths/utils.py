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
"""Helper methods for `path_contractors`."""
from tensornetwork import network
from tensornetwork import network_components
from typing import Any, Callable, Dict, List, Set, Tuple

# `opt_einsum` algorithm method typing
Algorithm = Callable[[List[Set[int]], Set[int], Dict[int, int]],
                     List[Tuple[int, int]]]


def multi_remove(elems: List[Any], indices: List[int]) -> List[Any]:
  """Remove multiple indicies in a list at once."""
  return [i for j, i in enumerate(elems) if j not in indices]


def find_copy_nodes(net: network.TensorNetwork) -> Tuple[
    Dict[network_components.CopyNode, network_components.Node],
    Dict[network_components.Node, network_components.CopyNode],
    Dict[network_components.Edge, network_components.Edge]]:
  # TODO: Docstring
  edge_map = {} # Maps all non-dangling edges of a copy node to a specific
                # non-dangling edge of this copy node
  copy_neighbors = {} # Maps nodes to their copy node neighbors
  node_neighbors = {node: set() for node in net.nodes_set} # Maps copy nodes to their node neighbors

  for copy in net.nodes_set:
    if isinstance(copy, network_components.CopyNode):
      node_neighbors.pop(copy)
      copy_neighbors[copy] = set()
      representative_edge = None
      for edge in copy.edges:
        if not edge.is_dangling():

          # Update `edge_map`
          if representative_edge is None:
            representative_edge = edge
          edge_map[edge] = representative_edge

          # Update `neighbors_of_copy`
          node = ({edge.node1, edge.node2} - {copy}).pop()
          copy_neighbors[copy].add(node)

          # Update `copy_neighbors_of`
          node_neighbors[node].add(copy)

  return copy_neighbors, node_neighbors, edge_map


def disconnect_copy_edge(net: network.TensorNetwork,
                         edge: network_components.Edge,
                         node: network_components.BaseNode):
  edge_node, edge_copy = net.disconnect(edge)
  if edge_node.node1 is not node:
    assert edge_copy.node1 is node
    edge_node, edge_copy = edge_copy, edge_node
  return edge_node, edge_copy


def isolate_copy_node(net: network.TensorNetwork,
                      copy: network_components.CopyNode,
                      node1: network_components.BaseNode,
                      node2: network_components.BaseNode
                      ) -> network_components.CopyNode:
  # TODO: Handle the case in which the copy and node1 or node2 share more
  # than one edges!

  # Find shared edges
  edge1, edge2 = None, None
  for edge in copy.edges:
    if edge.node1 is node1 or edge.node2 is node1:
      edge1 = edge
    elif edge.node1 is node2 or edge.node2 is node2:
      edge2 = edge
    if edge1 is not None and edge2 is not None:
      break

  edge1_node, edge1_copy = disconnect_copy_edge(net, edge1, node1)
  edge2_node, edge2_copy = disconnect_copy_edge(net, edge2, node2)

  copy.remove_edge(edge2_copy)
  new_copy = network_components.CopyNode(dimension=copy.dimension, rank=3)
  new_copy[0] ^ edge1_copy
  new_copy[1] ^ edge1_node
  new_copy[2] ^ edge2_node
  return new_copy


def contract_between_with_copies(net, node1, node2, copies):
  new_node = None
  for copy in copies:
    n = len(copy.edges)
    if n == 2:
      _, broken_edges = net.remove_node(copy)
      broken_edges[0] ^ broken_edges[1]
    elif n == 3:
      # TODO: Fix the implementation of this
      new_node = net.contract_copy_node(copy)
    else:
      raise ValueError("Cannot use `contract_between_with_copies` with "
                       "node {} that has {} edges".format(copy, n))
  if new_node is None:
    return node1 @ node2
  return new_node @ new_node


def get_path(net: network.TensorNetwork, algorithm: Algorithm,
             sorted_nodes: List[network_components.Node],
             edge_map: Dict[network_components.Edge,
                            network_components.Edge] = None
             ) -> List[Tuple[int, int]]:
  """Calculates the contraction paths using `opt_einsum` methods.

  Args:
    net: TensorNetwork object to contract.
    algorithm: `opt_einsum` method to use for calculating the contraction path.

  Returns:
    The optimal contraction path as returned by `opt_einsum`.
  """
  # TODO: FIx docstring
  if edge_map:
    input_sets = [set((edge_map[edge] if edge in edge_map else edge
                       for edge in node.edges)) for node in sorted_nodes]
  else:
    input_sets = [set(node.edges) for node in sorted_nodes]
  output_set = net.get_all_edges() - net.get_all_nondangling()
  size_dict = {edge: edge.dimension for edge in net.get_all_edges()}
  return algorithm(input_sets, output_set, size_dict)