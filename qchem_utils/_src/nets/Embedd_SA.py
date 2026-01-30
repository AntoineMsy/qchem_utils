from typing import Any
import jax.numpy as jnp
import jax
import netket as nk
import flax.linen as nn

def extract_patches_1d(x, graph=None, b=1):
    """
    x: (2, L)
    returns: (L_eff, 2*b)
    """
    L = x.shape[1]
    L_eff = L // b

    x = x.reshape(2, L_eff, b)        # (2, L_eff, b)
    x = jnp.transpose(x, (1, 0, 2))   # (L_eff, 2, b)
    x = x.reshape(L_eff, 2 * b)       # (L_eff, 2b)
    return x

def extract_patches_1d_separate_spin(x, graph=None, b=1):
    """
    x: (2, L)
    returns: (2*L_eff, b)
    """
    L = x.shape[1]
    L_eff = L // b

    x = x.reshape(2, L_eff, b)        # (2, L_eff, b)
    x = jnp.reshape(x, (2 * L_eff, b))  # (2*L_eff, b)

    return x


class Embed(nn.Module):
    d_model: int
    b: int
    n_patches: int  # given b and n_patches we can extract the system size: n_patches * (b**2) = L**2
    separate_spin: bool = False
    graph: nk.graph = None
    is_equivariant: bool = False
    param_dtype: Any = jnp.float64
    initializer: Any = nn.initializers.lecun_normal()

    def setup(self):
        self.extract_patches = extract_patches_1d

        self.embed = nn.Dense(
            self.d_model,
            kernel_init=self.initializer,
            param_dtype=self.param_dtype,
            bias_init=jax.nn.initializers.zeros,
        )
        
        # Standard learnable positional embedding
        # Shape: (1, n_patches, d_model) to broadcast over batch dimension
        self.pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, self.n_patches, self.d_model),
            self.param_dtype,
        )

    @nn.compact
    def __call__(self, x):

        L = self.graph.extent[0]
        # Ly = self.graph.extent[1]
        x = x.reshape(-1, 2, L)

        if (self.is_equivariant is True) and (self.graph is not None) and (self.b > 1):
            translations = []
            if self.graph.pbc[0]:
                translations.append(jnp.roll(x, 1, axis=-2))  # x translation
            if self.graph.pbc[1]:
                translations.append(jnp.roll(x, 1, axis=-1))  # y translation
            if self.graph.pbc[0] and self.graph.pbc[1]:
                translations.append(jnp.roll(x, 1, axis=(-2, -1)))  # x+y translation

            x = jnp.concatenate([x] + translations, axis=0)

        if self.separate_spin:
            x = jax.vmap(extract_patches_1d_separate_spin, in_axes=(0, None, None))(x, self.graph, self.b)
    
        else:
            # Apply the vectorized function to the input array
            x = jax.vmap(extract_patches_1d, in_axes=(0, None, None))(x, self.graph, self.b)
            
        # Perform the linear patch embedding
        x = self.embed(x)
        
        # Add the learnable positional embedding
        x = x + self.pos_embed
        
        return x