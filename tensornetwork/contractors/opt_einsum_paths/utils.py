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
    Dict[network_components.Edge, network_components.Edge]]:
  """Finds copy nodes and their normal node neighbors.

  Args:
    net: TensorNetwork that we want to contract.

  Returns:
    copy_neighbors: Dictionary that maps each copy node to a set that
      contains all its normal node neighbors.
    edge_map: Dictionary that maps each edge of a copy node to a specific
      representative edge of this copy node.
      This is used because in einsum string notation, the edges of a copy node
      are all equivalent and correspond to the same character.
  """
  edge_map = {} # Maps all non-dangling edges of a copy node to a specific
                # non-dangling edge of this copy node
  copy_neighbors = {} # Maps nodes to their copy node neighbors
  for copy in net.nodes_set:
    if isinstance(copy, network_components.CopyNode):
      copy_neighbors[copy] = set()
      representative_edge = None
      for edge in copy.edges:
        if not edge.is_dangling():
          # Update `edge_map`
          if representative_edge is None:
            representative_edge = edge
          edge_map[edge] = representative_edge
          # Update `neighbors_of_copy`
          node = edge.node1 if edge.node2 is copy else edge.node2
          copy_neighbors[copy].add(node)
  return copy_neighbors, edge_map


def find_copy_neighbors(net: network.TensorNetwork,
                        node: network_components.BaseNode
                        ) -> Set[network_components.CopyNode]:
  """Finds all the copy nodes connected to a node."""
  copies = set()
  for edge in node.edges:
    if not edge.is_dangling():
      neighbor = edge.node2 if edge.node1 is node else edge.node1
      if isinstance(neighbor, network_components.CopyNode):
        copies.add(neighbor)
  return copies


def disconnect_copy_edge(net: network.TensorNetwork,
                         edge: network_components.Edge,
                         node: network_components.BaseNode):
  # TODO: Docstring
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
  # TODO: Docstring
  # Find shared edges
  edges1 = set(edge for edge in copy.edges if node1 in {edge.node1, edge.node2})
  edges2 = set(edge for edge in copy.edges if node2 in {edge.node1, edge.node2})

  new_rank = len(edges1) + len(edges2) + 1
  new_copy = net.add_node(
      network_components.CopyNode(dimension=copy.dimension, rank=new_rank))
  for i, edge in enumerate(edges1):
    node_edge, copy_edge = disconnect_copy_edge(net, edge, node1)
    if new_copy[0].is_dangling():
      new_copy[0] ^ copy_edge
    else:
      copy.remove_edge(copy_edge)
    node_edge ^ new_copy[i + 1]
  for i, edge in enumerate(edges2):
    node_edge, copy_edge = disconnect_copy_edge(net, edge, node2)
    copy.remove_edge(copy_edge)
    node_edge ^ new_copy[i + len(edges1) + 1]
  return new_copy


def contract_between_with_copies(
    net: network.TensorNetwork,
    node1: network_components.BaseNode,
    node2: network_components.BaseNode,
    shared_copies: Set[network_components.CopyNode]
    ) -> network_components.BaseNode:
  # TODO: Docstring
  if node1 is node2:
    # No need to implement this since trace edges are handled seperately
    # in opt_einsum contractors
    raise NotImplementedError

  _VALID_SUBSCRIPTS = iter(
      'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

  for copy in set(shared_copies):
    if len(copy.edges) == 2:
      _, broken_edges = net.remove_node(copy)
      shared_copies.remove(copy)
      broken_edges[0] ^ broken_edges[1]
    elif len(copy.edges) != 3:
      raise ValueError

  copy_edges = {}
  edge_map = {} # for mapping edges to einsum characters
  copy_edge_map = {} # for updating edges in the end
  for copy in shared_copies:
    for edge in copy.edges:
      edge_nodes = {edge.node1, edge.node2}
      if node1 not in edge_nodes and node2 not in edge_nodes:
        copy_edge = edge
        break
    copy_edges.update({edge: copy_edge for edge in copy.edges})
    edge_map[copy_edge] = next(_VALID_SUBSCRIPTS)
    old_axis = edge.axis1 if edge.node1 is copy else edge.axis2
    copy_edge_map[copy_edge] = (copy, old_axis)
    net.remove_node(copy)

  node1_expr = []
  output_expr, output_edges = [], []
  for edge in node1.edges:
    if edge in copy_edges:
      char = edge_map[copy_edges[edge]]
      node1_expr.append(char)
      output_expr.append(char)
      output_edges.append((edge,) + copy_edge_map[copy_edges[edge]])
    else:
      char = next(_VALID_SUBSCRIPTS)
      node1_expr.append(char)
      if edge.node1 is node2 or edge.node2 is node2:
        edge_map[edge] = char
      else:
        output_expr.append(char)
        old_axis = edge.axis1 if edge.node1 is node1 else edge.axis2
        output_edges.append((edge, node1, old_axis))

  node2_expr = []
  for edge in node2.edges:
    if edge in edge_map:
      node2_expr.append(edge_map[edge])
    elif edge in copy_edges:
      node2_expr.append(edge_map[copy_edges[edge]])
    else:
      char = next(_VALID_SUBSCRIPTS)
      node2_expr.append(char)
      output_expr.append(char)
      old_axis = edge.axis1 if edge.node1 is node2 else edge.axis2
      output_edges.append((edge, node2, old_axis))

  input_expr = ",".join(["".join(node1_expr), "".join(node2_expr)])
  einsum_expr = "->".join([input_expr, "".join(output_expr)])
  new_tensor = net.backend.einsum(einsum_expr, node1.tensor, node2.tensor)
  new_node = net.add_node(new_tensor)
  # The uncontracted axes of node1 (node2) now correspond to the first (last)
  # axes of new_node.
  for new_axis, (edge, old_node, old_axis) in enumerate(output_edges):
    edge.update_axis(old_node=old_node,
                     old_axis=old_axis,
                     new_axis=new_axis,
                     new_node=new_node)
    new_node.add_edge(edge, new_axis)

  net.nodes_set.remove(node1)
  net.nodes_set.remove(node2)
  node1.disable()
  node2.disable()
  return new_node


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