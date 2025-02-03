from nequip_jax import NEQUIPLayerFlax  # e3nn implementation of NEQUIP
import e3nn_jax as e3nn
import flax  # neural network modules for jax
import jax.numpy as jnp
import jax

class Model(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, graphs, cutoff = 5.0):
        senders = graphs.senders
        receivers = graphs.receivers
        atomic_numbers = graphs.globals["x"]
        species = jnp.asarray(graphs.globals["x"], dtype=jnp.int32)

        positions = graphs.nodes["positions"]
        
        def cal_energy(pos):
            features = e3nn.IrrepsArray("9x0e", jax.nn.one_hot(atomic_numbers, 9))
            positions_receivers = pos[receivers]
            positions_senders = pos[senders]
            vectors = e3nn.IrrepsArray("1o", positions_receivers - positions_senders)

            vectors = vectors / cutoff

            # Apply 3 Nequip layers with different internal representations
            for irreps in [
                 "32x0e + 32x1o + 32x2e",
                 "32x0e + 32x1o + 32x2e",
                 "32x0e + 32x1o + 32x2e",
            ]:
                layer = NEQUIPLayerFlax(
                    avg_num_neighbors=14.26,  # average number of neighbors to normalize by
                    output_irreps=irreps,
                )
                features = layer(vectors, features, species, senders, receivers)

            # Self-Interaction layers
            features = e3nn.flax.Linear("8x0e")(features)
            features = e3nn.flax.Linear("0e")(features)

            # Global Pooling
            return e3nn.sum(features).array

        hessian = jax.hessian(cal_energy)(positions).squeeze(0)

        return hessian
