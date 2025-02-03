from ase.io import read
from ase.neighborlist import neighbor_list
import jraph
import numpy as np
from Nequip import Model
import orbax.checkpoint as ocp
from utils import graph_batch, pad_graph_to_nearest_power_of_two
from constant import hess_t, cm_hz
import itertools
import subprocess
import jax

# # Initialize the model
model = Model()
orbax_checkpointer = ocp.StandardCheckpointer()
params = orbax_checkpointer.restore('/path_to_checkpoint/flax_ckpt/hessian_checkpoint_x')['model_param']  #jax.jit(model.init)(random_key, dataset[0])

# If you want to run on CPU, uncomment the following line

# params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device=jax.devices("cpu")[0]), params)
# jax.config.update("jax_platform_name", "cpu")
# print(f"Running on platform: {jax.default_backend()}")

##############

def predictor(params, dataset):
    preds = model.apply(params, dataset)

    return preds

def predict_hessian_from_strucure(structure_file):

    cutoff = 5.0
    mol = read(structure_file)
    pos = mol.get_positions()
    x = mol.get_atomic_numbers()
    n_atoms = pos.shape[0]

    # this is a place holder for the true hessian matrix to build the graph
    hessian_matrix = np.zeros((n_atoms*3, n_atoms*3))

    edge_index_s, edge_index_t, edge_displacement=neighbor_list('ijD',mol, cutoff=cutoff)

    num_edge = edge_index_s.shape[0]
    num_node = pos.shape[0]

    graph = jraph.GraphsTuple(
        # positions are per-node features:
        nodes=dict(positions=pos),
        edges=dict(shifts=edge_displacement),
        # energy and cell are per-graph features:
        globals=dict(hessian=hessian_matrix, x=x),
        # The rest of the fields describe the connectivity and size of the graph.
        senders=edge_index_s,
        receivers=edge_index_t,
        n_node=np.array([num_node]),
        n_edge=np.array([num_edge]),
    )

    _batch, _hessian_true = graph_batch([graph])
    batch = pad_graph_to_nearest_power_of_two(_batch)
    hessian_true = _hessian_true.reshape(_hessian_true.shape[0]*3, _hessian_true.shape[2]*3)
    hessian_pred = predictor(params, batch)
    hessian_pred = hessian_pred.reshape(hessian_pred.shape[0]*3, hessian_pred.shape[2]*3)
    hessian_pred = hessian_pred[~np.all(hessian_pred == 0, axis=1)]
    hessian_pred = hessian_pred[:, ~np.all(hessian_pred == 0, axis=0)]

    return hessian_pred, mol

def is_linear(mol, tolerance=1.0):
    """Check if the molecule is linear"""
    index_atoms = list(range(len(mol.get_positions())))
    index_list = np.asarray(list(itertools.combinations(index_atoms, 3)))
    angles = mol.get_angles(index_list)
    close_to_0 = np.isclose(angles, 0, atol=tolerance)
    close_to_180 = np.isclose(angles, 180, atol=tolerance)
    linear = np.logical_or(close_to_0, close_to_180)
    return linear.all()

def cal_freq_modes(mol, hessian, normal=False, linear_tolerance=1.0, scale=0.965): # scale is for the comparison to the experiment
    masses = mol.get_masses()
    wmasses=np.power(masses,-0.5).repeat(3, axis=0)
    wmat=wmasses[:,None]*wmasses[None,:]
    hessian = hessian*wmat
    eva, evec=np.linalg.eigh(hessian)
    eva = eva * hess_t
    freq=np.power(eva,0.5) / (2 * np.pi)
    freq=freq/cm_hz
    p=-evec.T*wmasses
    normals = np.expand_dims(np.linalg.norm(p, axis=1), 1)
    if normal:
        p=p/normals
    p=p.reshape(len(freq),-1,3)
    if scale is not None:
        freq=freq*scale
    linear = is_linear(mol, tolerance=linear_tolerance)
    if linear:
        return freq[5:],p[5:]
    else:
        return freq[6:],p[6:]

def gen_oclimax_xyz(filename, mol, freq, modes):
    f = open(filename, "w")
    symbols = mol.get_chemical_symbols()
    positions = mol.get_positions()
    n_atoms = len(symbols)
    n_modes = len(freq)
    for i in range(n_modes):
        line1 = f"       {n_atoms}\n"
        line2 = f" Eigenmode         {i+1}  {freq[i]}   cm-1\n"
        lines = [line1, line2]
        for j in range(n_atoms):
            lines.append(f"{symbols[j]}  {positions[j][0]:10.8f}  {positions[j][1]:10.8f}  {positions[j][2]:10.8f}  {modes[i][j][0]:10.8f} {modes[i][j][1]:10.8f} {modes[i][j][2]:10.8f}\n")
        f.writelines(lines)
    f.close()

if __name__ == "__main__":

    raw_file = "sucrose.xyz"
    hessian, mol = predict_hessian_from_strucure(raw_file)
    pred_freq, pred_modes = cal_freq_modes(mol, hessian)
    gen_oclimax_xyz("./sucrose_oclimax.xyz", mol, pred_freq, pred_modes)
    subprocess.run(["./oclimax_convert", "-xyz", "./sucrose_oclimax.xyz"], text=True)