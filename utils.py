import jraph  # graph neural networks in jax
import jax.numpy as jnp
from scipy.linalg import block_diag
import jax.tree_util as tree
from jraph._src import graph as gn_graph

# Adapted from https://github.com/google-deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb
# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
def _nearest_bigger_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than x for padding."""
  y = 2
  while y < x:
    y *= 2
  return y

def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Pads a batched `GraphsTuple` to the nearest power of two.
  For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
  would pad the `GraphsTuple` nodes and edges:
    7 nodes --> 8 nodes (2^3)
    5 edges --> 8 edges (2^3)
  And since padding is accomplished using `jraph.pad_with_graphs`, an extra
  graph and node is added:
    8 nodes --> 9 nodes
    3 graphs --> 4 graphs
  Args:
    graphs_tuple: a batched `GraphsTuple` (can be batch size 1).
  Returns:
    A graphs_tuple batched to the nearest power of two.
  """
  # Add 1 since we need at least one padding node for pad_with_graphs.
  pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
  pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
  # Add 1 since we need at least one padding graph for pad_with_graphs.
  # We do not pad to nearest power of two because the batch size is fixed.
  pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
  return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                               pad_graphs_to)

def graph_batch(graphs, np_=jnp):
    """Returns batched graph given a list of graphs and a numpy-like module."""
    # Calculates offsets for sender and receiver arrays, caused by concatenating
    # the nodes arrays.
    offsets = np_.cumsum(
        np_.array([0] + [np_.sum(g.n_node) for g in graphs[:-1]]))

    def _map_concat(nests):
        concat = lambda *args: np_.concatenate(args)
        return tree.tree_map(concat, *nests)

    x = _map_concat([g.globals['x'] for g in graphs])

    new_graphs = gn_graph.GraphsTuple(
        n_node=np_.concatenate([g.n_node for g in graphs]),
        n_edge=np_.concatenate([g.n_edge for g in graphs]),
        nodes=_map_concat([g.nodes for g in graphs]),
        edges=_map_concat([g.edges for g in graphs]),
        globals=dict(x=x),
        senders=np_.concatenate([g.senders + o for g, o in zip(graphs, offsets)]),
        receivers=np_.concatenate(
            [g.receivers + o for g, o in zip(graphs, offsets)]))

    hessians = [ g.globals['hessian'] for g in graphs ]
    y_label = block_diag(*hessians)
    n_atoms = new_graphs.n_node.sum()
    y_label = y_label.reshape((n_atoms, 3, n_atoms, 3))

    return new_graphs, y_label
