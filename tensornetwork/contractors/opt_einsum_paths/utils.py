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
  for copy in set(shared_copies):
    if len(copy.edges) == 2:
      shared_copies.remove(copy)
      _, broken_edges = net.remove_node(copy)
      broken_edges[0] ^ broken_edges[1]
    elif len(copy.edges) != 3:
      raise NotImplementedError("Copy node {} has {} edges and cannot be "
                                "contracted.".format(copy, len(copy.edges)))
  if not shared_copies:
    return node1 @ node2

  _VALID_SUBSCRIPTS = iter(
      'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

  edge_char = {} # Dict: Edge -> einsum char
  node1_expr, output_expr = [], []
  # required to update edges: List[Tuple[Edge, old_node, old_axis]
  output_edges = []
  for edge in node1.edges:
    char = next(_VALID_SUBSCRIPTS)
    node1_expr.append(char)
    # Find neighbor
    if edge.node1 is node1:
      old_axis = edge.axis1
      neighbor = edge.node2
    else:
      assert edge.node2 is node1
      old_axis = edge.axis2
      neighbor = edge.node1

    # Map edge to einsum char
    if neighbor is node2:
      edge_char[edge] = char

    elif neighbor in shared_copies:
      assert len(neighbor.edges) == 3
      for copy_edge in set(neighbor.edges):
        if node1 in {copy_edge.node1, copy_edge.node2}:
          net.disconnect(copy_edge)
        elif node2 is copy_edge.node1:
          copy_edge, _ = net.disconnect(copy_edge)
          edge_char[copy_edge] = char
        elif node2 is copy_edge.node2:
          _, copy_edge = net.disconnect(copy_edge)
          edge_char[copy_edge] = char
        else:
          nodes = {copy_edge.node1, copy}
          assert node1 not in nodes
          assert node2 not in nodes
          output_expr.append(char)
          output_edges.append((copy_edge, neighbor, old_axis))

    else:
      output_expr.append(char)
      output_edges.append((edge, node1, old_axis))

  node2_expr = []
  for edge in node2.edges:
    if edge in edge_char:
      node2_expr.append(edge_char[edge])
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
    if old_node in shared_copies:
      shared_copies.remove(old_node)
      net.remove_node(old_node)

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