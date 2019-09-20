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

  edge_map = {edge: next(_VALID_SUBSCRIPTS)
              for edge in net.get_shared_edges(node1, node2)}
  copy_output = {}
  for copy in shared_copies:
    partners = copy.get_partners()
    assert node1 in partners
    assert node2 in partners
    assert len(partners) >= 2
    # each copy node corresponds to a specific einsum char
    char = next(_VALID_SUBSCRIPTS)
    if len(partners) == 2:
      # copy is only connected to node1 and node2 and can be removed
      _, broken_edges = net.remove_node(copy)
      # map all the broken edges to the same einsum char
      for edge in broken_edges.values():
        assert edge not in edge_map
        edge_map[edge] = char
    else:
      # disconnect all edges that connect the copy node with node1 and node2
      # and map the resulting dangling edges of node1 and node2 to the
      # same einsum char
      disc_copy_edges = set()
      for edge in copy.edges:
        edge_nodes = {edge.node1, edge.node2}
        if node1 in edge_nodes or node2 in edge_nodes:
          copy_edge, node_edge = net.disconnect(edge)
          if copy_edge.node1 is not copy:
            copy_edge, node_edge = node_edge, copy_edge
          disc_copy_edges.add(copy_edge)
          assert node_edge not in edge_map
          edge_map[node_edge] = char
      # now remove all the dangling edges we created in the copy node
      # and keep only one to connect to the output
      while len(disc_copy_edges) > 1:
        copy.remove_edge(disc_copy_edges.pop())
      copy_output[char] = disc_copy_edges.pop()

  # Create einsum expressions and keep track of output edges in
  # order to update the network after contraction
  input_expr = {node1: [], node2: []}
  output_expr, output_edges = [], []
  for node, expr in input_expr.items():
    for edge in node.edges:
      if edge in edge_map:
        char = edge_map[edge]
        expr.append(char)
        if char in copy_output:
          output_expr.append(char)
          output_edges.append((edge, None, char))
      else:
        char = next(_VALID_SUBSCRIPTS)
        expr.append(char)
        output_expr.append(char)
        old_axis = edge.axis1 if edge.node1 is node else edge.axis2
        output_edges.append((edge, node, old_axis))

  input_expr = ["".join(input_expr[node]) for node in [node1, node2]]
  input_expr = ",".join(input_expr)
  einsum_expr = "->".join([input_expr, "".join(output_expr)])
  new_tensor = net.backend.einsum(einsum_expr, node1.tensor, node2.tensor)
  new_node = net.add_node(new_tensor)
  # The uncontracted axes of node1 (node2) now correspond to the first (last)
  # axes of new_node.
  for new_axis, (edge, old_node, old_axis) in enumerate(output_edges):
    if old_node is None:
      # Now `old_axis` contains the `char` of the copy node so
      # we can use `copy_output` to find the old edge to connect with
      new_node[new_axis] ^ copy_output[old_axis]
    else:
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