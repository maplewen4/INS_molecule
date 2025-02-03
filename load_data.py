import numpy as np
import torch
from torch_geometric.data import Dataset
import os.path as osp
from ase.io import read
from ase.neighborlist import neighbor_list
from tqdm import tqdm
import jraph
import warnings
warnings.filterwarnings("ignore")

class MoleculeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        # Return the names of the files to be found in the `raw_dir`
        f = open("data_list.txt", 'r')
        data_list = f.read().splitlines()
        f.close()
        return data_list

    @property
    def processed_file_names(self):
        len_data = len(self.raw_file_names)
        file_list = [f'data_{i}.pt' for i in range(1, len_data + 1)]
        return file_list

    def download(self):
        # Download raw data to `self.raw_dir`
        pass

    def process(self):

        for i, raw_file in tqdm(enumerate(self.raw_file_names), total = len(self.raw_file_names)):
            data = make_data_point(raw_file)
        
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{i+1}.pt'))
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx+1}.pt'))
        return data


def make_data_point(raw_file, cutoff=5.0):
    structrue_filename = "raw/" + raw_file + ".xyz"
    mol = read(structrue_filename)
    pos = mol.get_positions()
    x = mol.get_atomic_numbers()
    hessian_filename = "raw/hessian_" + raw_file + ".npy"
    hessian_matrix = np.load(hessian_filename)
    hessian_matrix = hessian_matrix * 27.2114079527 / 0.529177249 / 0.529177249 # convert atomic unit to eV/Angstrom**2

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

    return graph



if __name__ == '__main__':

    dataset = MoleculeDataset('./')
    print(dataset[0])
    dataset.shuffle()

