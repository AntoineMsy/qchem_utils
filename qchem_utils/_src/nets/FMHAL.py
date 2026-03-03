import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
from einops import rearrange
from jax._src import dtypes
from typing import Any
from functools import partial

def custom_uniform(scale = 1e-2, dtype = jnp.float_):
    def init(key, shape, dtype = dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return (2.0*random.uniform(key, shape, dtype) - 1.0) * scale
    return init

def custom_id(dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        sidex = shape[-2]
        sidey = shape[-1]
        n_patches = sidex * sidey
        vector = jnp.zeros(n_patches, dtype=dtype)
        vector = vector.at[0].set(1.0) 
        vector = vector.reshape(sidex, sidey)
        #vector = jnp.array([1, 0, 0, 0], dtype=dtype)
        return jnp.tile(vector, (shape[0], 1, 1))
    return init

def construct_trans_inv_full_j(base_J, graph):
    """
    Constructs a full 2D translationally invariant attention matrix
    from base_J of shape (num_heads, Lx, Ly), where for each head h
      full_J[h, p, q] = base_J[h, (r_p - r_q) mod Lx, (c_p - c_q) mod Ly].
    
    This requires that patches are arranged on a grid of size (Lx, Ly)
    in standard row-major order.
    
    Args:
        base_J: jnp.ndarray, shape (num_heads, Lx, Ly)
    
    Returns:
        full_J: jnp.ndarray, shape (num_heads, Lx*Ly, Lx*Ly)
    """
    heads, Lx, Ly = base_J.shape
    n_patch = Lx * Ly
    # Build grid coordinates in standard row-major order.
    grid_r, grid_c = jnp.meshgrid(jnp.arange(Lx), jnp.arange(Ly), indexing="ij")
    grid_r = grid_r.reshape(-1)  # shape (n_patch,)
    grid_c = grid_c.reshape(-1)  # shape (n_patch,)

    if graph.pbc[0] and graph.pbc[1]:
        # Compute relative differences for every pair (p, q) with wrapping.
        rel_r = (grid_r[:, None] - grid_r[None, :]) % Lx  # shape (n_patch, n_patch)
        rel_c = (grid_c[:, None] - grid_c[None, :]) % Ly  # shape (n_patch, n_patch)
        # Expand to include head dimension.
        rel_r = jnp.broadcast_to(rel_r, (heads, n_patch, n_patch))
        rel_c = jnp.broadcast_to(rel_c, (heads, n_patch, n_patch))
        # To index base_J we compute a linear index: index = r * Ly + c.
        indices = rel_r * Ly + rel_c  # shape (heads, n_patch, n_patch)

    elif graph.pbc[0]:
        # Invariance along x: use modulo on row difference.
        # For a pair (p, q): use relative row: (r_p - r_q) mod Lx,
        # and use absolute col from patch q.
        rel_r = (grid_r[:, None] - grid_r[None, :]) % Lx         # (n_patch, n_patch)
        col_idx = jnp.broadcast_to(grid_c[None, :], (n_patch, n_patch))  # (n_patch, n_patch)
        # Broadcast to heads.
        rel_r = jnp.broadcast_to(rel_r, (heads, n_patch, n_patch))
        col_idx = jnp.broadcast_to(col_idx, (heads, n_patch, n_patch))
        # Compute linear index into base_J: index = rel_r * Ly + col_idx.
        indices = rel_r * Ly + col_idx  # (heads, n_patch, n_patch)

    elif graph.pbc[1]:
        # Invariance along y: use modulo on column difference.
        # For a pair (p, q): use relative col: (c_p - c_q) mod Ly,
        # and use absolute row from patch p.
        rel_c = (grid_c[:, None] - grid_c[None, :]) % Ly         # (n_patch, n_patch)
        row_idx = jnp.broadcast_to(grid_r[:, None], (n_patch, n_patch))  # (n_patch, n_patch)
        # Broadcast to heads.
        rel_c = jnp.broadcast_to(rel_c, (heads, n_patch, n_patch))
        row_idx = jnp.broadcast_to(row_idx, (heads, n_patch, n_patch))
        indices = row_idx * Ly + rel_c  # (heads, n_patch, n_patch)
    else:
        raise ValueError("Invariance must be either 'x' or 'y' or 'xy'")
    
    # Flatten base_J spatial dimensions: shape (heads, n_patch)
    base_J_flat = base_J.reshape(heads, n_patch)
    # Gather attention scores using the computed indices.
    full_J = jnp.take_along_axis(base_J_flat[:, None, :], indices, axis=-1)
    return full_J


class FactoredAttention(nn.Module):
    n_patches: int
    d_model: int
    heads: int
    b: int
    graph: nk.graph = None
    is_equivariant: bool = False
    use_uniform_init: bool = False
    param_dtype: Any = jnp.float64
    initializer: Any = nn.initializers.lecun_normal()

    def setup(self):

        if self.use_uniform_init:
            init = custom_uniform(scale=(3.0/self.n_patches)**0.5)
        else: 
            init = custom_id()
        
        if (self.is_equivariant is True) and (self.graph is not None):

            self.N_translations = 1 + int(self.graph.pbc[0]) + int(self.graph.pbc[1]) + int(self.graph.pbc[0] and self.graph.pbc[1])
            sidex = self.graph.extent[0] // self.b
            sidey = self.graph.extent[1] // self.b

            # base_J defines the attentional kernel as a function of 2D relative position.
            # Its shape is (num_heads, Lx, Ly)
            self.base_J = self.param("J", 
                                    init,
                                    (self.heads, sidex, sidey))
            
            self.full_J = construct_trans_inv_full_j(self.base_J, self.graph)

        elif self.is_equivariant is False:
            self.base_J = self.param("J", 
                                    init,
                                    (self.heads, self.n_patches, self.n_patches))
            self.full_J = self.base_J  # Non translational invariant.

        else:
            raise ValueError("You must provide either a graph for translational invariance or you must set is_translational_invariant to False")

        # Projection for Value vectors. Input x shape: (Nt, n_patches, d_in)
        # The projection maps to latent_dim; later we split into heads.
        self.v = nn.Dense(
            self.d_model,
            kernel_init=self.initializer,
            param_dtype=self.param_dtype,
            bias_init=jax.nn.initializers.zeros,
        )
        # Final output projection.
        self.W = nn.Dense(
            self.d_model,
            kernel_init=self.initializer,
            param_dtype=self.param_dtype,
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):
        """
        Args:
            x: input tensor of shape (Nt, n_patches, d_in)
        
        Returns:
            out: output tensor of shape (Nt, n_patches, latent_dim)
        """

        # Project values: (Nt, n_patches, latent_dim)
        v = self.v(x)

        # Reshape (Nt, n_patches, num_heads, d_head) -> (Nt, num_heads, n_patches, d_head)
        v = rearrange(v, "Nt np (h d) -> Nt np h d", h=self.heads)
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Construct full attention matrix from base_J.
        # full_J has shape (N_translations, num_heads, n_patches, n_patches)

        # Apply attention per head. (Einstein summation multiplies J and v along patch dimension.)
        attn_out = jnp.einsum("hij,thjd->thid", self.full_J, v)
        # attn_out is shape (Nt, num_heads, n_patches, d_head)

        # Concatenate heads: first transpose to (Nt, n_patches, num_heads, d_head) then reshape.
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        attn_out = rearrange(attn_out, "Nt np h d -> Nt np (h d)")

        out = self.W(attn_out)
        return out, self.base_J

class EncoderBlock(nn.Module):
    n_patches: int
    d_model: int
    heads: int
    b: int
    graph: nk.graph = None
    is_equivariant: bool = False
    use_uniform_init: bool = False
    param_dtype: Any = jnp.float64
    initializer: Any = nn.initializers.lecun_normal()

    def setup(self):
        self.attn = FactoredAttention(n_patches=self.n_patches, 
                                        d_model=self.d_model, 
                                        heads=self.heads, 
                                        b=self.b,
                                        graph=self.graph, 
                                        is_equivariant=self.is_equivariant,
                                        use_uniform_init = self.use_uniform_init,
                                        param_dtype=self.param_dtype, 
                                        initializer=self.initializer
                                        )
        # Layer normalization
        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)
        # Feed forward layer
        self.ff = nn.Sequential(
            [
                nn.Dense(
                    2 * self.d_model,
                    kernel_init=self.initializer,
                    param_dtype=self.param_dtype,
                ),
                nn.gelu,
                nn.Dense(
                    self.d_model,
                    kernel_init=self.initializer,
                    param_dtype=self.param_dtype,
                ),
            ]
        )

    def __call__(self, x):

        x_att, att = self.attn(self.layer_norm_1(x))
        x = x + x_att
        x = x + self.ff(self.layer_norm_2(x))
        return x, att


class Encoder_FMHAL(nn.Module):
    n_patches: int
    n_layers: int
    d_model: int
    heads: int
    b: int
    graph: nk.graph = None
    is_equivariant: bool = False
    use_uniform_init: bool = False
    param_dtype: Any = jnp.float64
    initializer: Any = nn.initializers.lecun_normal()

    def setup(self):
        self.layers = [
            EncoderBlock(n_patches=self.n_patches, 
                         d_model=self.d_model, 
                         b=self.b,
                         heads=self.heads, 
                         graph=self.graph, 
                         is_equivariant=self.is_equivariant,
                         use_uniform_init = self.use_uniform_init,
                         param_dtype=self.param_dtype, 
                         initializer=self.initializer
                         )

            for _ in range(self.n_layers)
        ]

    def __call__(self, x):

        for _, l in enumerate(self.layers):
            x = l(x)[0]

        return x

    def get_attention(self, x):
        # A function to return the attention maps within the model for a single application
        # Used for visualization purpose later
        attention_maps = []
        for l in self.layers:
            _, attn_map = l(x)
            attention_maps.append(attn_map)
            x, _ = l(x)
        return attention_maps
