import jax
import jax.numpy as jnp
import numpy as np
import optax  # optimizers for jax
from Nequip import Model
from tqdm import tqdm
import orbax.checkpoint as ocp
from etils import epath
from load_data import MoleculeDataset

from utils import graph_batch, pad_graph_to_nearest_power_of_two


dataset = MoleculeDataset('./')

train_ratio = 0.90
val_ratio = 0.05
test_ratio = 0.05

# Calculate split sizes
total_size = len(dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size

# Split dataset
train_datasets = dataset[:train_size]
val_datasets = dataset[train_size:train_size + val_size]
test_datasets = dataset[train_size + val_size:]

random_key = jax.random.PRNGKey(0)  # change it to get different initializations

# # Initialize the model
model = Model()
params = jax.jit(model.init)(random_key, dataset[0])

# Use the following if you want to and finetune a model
'''
orbax_checkpointer = ocp.StandardCheckpointer()
params = orbax_checkpointer.restore('/path_to_checkpoint/flax_ckpt/hessian_checkpoint_x')['model_param']
'''

# Initialize the optimizer
opt = optax.adam(3e-4)
opt_state = opt.init(params)


def loss_fn(preds, targets):
    assert preds.shape == targets.shape
    return jnp.mean(jnp.square(preds - targets))

@jax.jit
def train_step(opt_state, params, dataset, hessian_label):

    # Compute the loss as a function of the parameters
    def fun(w):
        preds = model.apply(w, dataset)
        targets = hessian_label

        assert preds.shape == targets.shape

        return loss_fn(preds, targets)

    # And take its gradient
    loss, grad = jax.value_and_grad(fun)(params)

    # Update the parameters and the optimizer state
    updates, opt_state = opt.update(grad, opt_state)
    params = optax.apply_updates(params, updates)

    return opt_state, params, loss

@jax.jit
def validate_step(params, dataset, hessian_label):

    # Compute the loss as a function of the parameters
    def fun(w):
        preds = model.apply(w, dataset)
        targets = hessian_label

        assert preds.shape == targets.shape

        return loss_fn(preds, targets)

    # And take its gradient
    loss = fun(params)

    return loss

def split_list_into_sublists(original_list, chunk_size):
    return [original_list[i:i + chunk_size] for i in range(0, len(original_list), chunk_size)]

batch_size = 2
n_epoch = 100

# Path to save the model
ckpt_dir = epath.Path('/path_to_checkpoint/flax_ckpt/')

train_dataloader = split_list_into_sublists(train_datasets, batch_size)
val_dataloader = split_list_into_sublists(val_datasets, batch_size)

x_epoch = np.zeros(0)
train_loss_epoch = np.zeros(0)
val_loss_epoch = np.zeros(0)
best_loss = jnp.inf
for epoch in range(n_epoch):
    running_loss = 0.0
    running_val_loss = 0.0
    for j, batch_temp in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        _batch, _hessian_true = graph_batch(batch_temp)
        batch = pad_graph_to_nearest_power_of_two(_batch)
        n_node_before = _batch.nodes['positions'].shape[0]
        n_node_after = batch.nodes['positions'].shape[0]
        hessian_true = jnp.pad(_hessian_true, ((0, n_node_after - n_node_before), (0, 0), (0, n_node_after - n_node_before), (0, 0)))
        opt_state, params, loss = train_step(opt_state, params, batch, hessian_true)
        running_loss += loss
    running_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Training Loss: {running_loss}")

    for j, batch_temp in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        _batch, _hessian_true = graph_batch(batch_temp)
        batch = pad_graph_to_nearest_power_of_two(_batch)
        n_node_before = _batch.nodes['positions'].shape[0]
        n_node_after = batch.nodes['positions'].shape[0]
        hessian_true = jnp.pad(_hessian_true, ((0, n_node_after - n_node_before), (0, 0), (0, n_node_after - n_node_before), (0, 0)))
        val_loss = validate_step(params, batch, hessian_true)
        running_val_loss += val_loss
    running_val_loss = running_val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1} Validation Loss: {running_val_loss}")

    x_epoch = np.append(x_epoch, epoch+1)
    train_loss_epoch = np.append(train_loss_epoch,running_loss)
    val_loss_epoch = np.append(val_loss_epoch, running_val_loss)

    # Save the model if the validation loss is lower than the best loss
    if running_val_loss < best_loss:
        best_loss = running_val_loss
        ckpt = {'model_param': params}
        orbax_checkpointer = ocp.StandardCheckpointer()
        orbax_checkpointer.save(ckpt_dir / 'hessian_checkpoint_x', ckpt, force=True)

    # Save the learning curve data
    learning_curve = np.stack((x_epoch, train_loss_epoch, val_loss_epoch))
    np.save('lc_hessian_1.npy', learning_curve) # lc = learning curve
