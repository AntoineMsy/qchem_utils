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

def construct_trans_inv_full_j(base_J, graph):
    """
    Constructs a full 2D translationally invariant attention bias matrix
    from base_J of shape (num_heads, Lx, Ly), where for each head h
      full_J[h, p, q] = base_J[h, (r_p - r_q) mod Lx, (c_p - c_q) mod Ly].
    
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
        rel_r = (grid_r[:, None] - grid_r[None, :]) % Lx         # (n_patch, n_patch)
        col_idx = jnp.broadcast_to(grid_c[None, :], (n_patch, n_patch))  # (n_patch, n_patch)
        # Broadcast to heads.
        rel_r = jnp.broadcast_to(rel_r, (heads, n_patch, n_patch))
        col_idx = jnp.broadcast_to(col_idx, (heads, n_patch, n_patch))
        indices = rel_r * Ly + col_idx  # (heads, n_patch, n_patch)

    elif graph.pbc[1]:
        # Invariance along y: use modulo on column difference.
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


class SelfAttention(nn.Module):
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
        # Head dimension
        self.d_head = self.d_model // self.heads
        if self.d_model % self.heads != 0:
            raise ValueError("d_model must be divisible by heads")

        # Projections for Query, Key, and Value
        self.q = nn.Dense(self.d_model, kernel_init=self.initializer, param_dtype=self.param_dtype, bias_init=nn.initializers.zeros)
        self.k = nn.Dense(self.d_model, kernel_init=self.initializer, param_dtype=self.param_dtype, bias_init=nn.initializers.zeros)
        self.v = nn.Dense(self.d_model, kernel_init=self.initializer, param_dtype=self.param_dtype, bias_init=nn.initializers.zeros)
        
        # Final output projection
        self.o = nn.Dense(self.d_model, kernel_init=self.initializer, param_dtype=self.param_dtype, bias_init=nn.initializers.zeros)

        # Handling Translational Invariance via Relative Positional Bias
        if (self.is_equivariant is True) and (self.graph is not None):
            sidex = self.graph.extent[0] // self.b
            sidey = self.graph.extent[1] // self.b
            
            # Learnable relative bias parameters (similar to base_J in factored attention, but added to logits)
            self.relative_bias_params = self.param("rel_bias", 
                                    nn.initializers.zeros,
                                    (self.heads, sidex, sidey))
            
            self.full_bias = construct_trans_inv_full_j(self.relative_bias_params, self.graph)

        elif self.is_equivariant is False:
            self.full_bias = None
        else:
            raise ValueError("You must provide either a graph for translational invariance or you must set is_equivariant to False")

    def __call__(self, x):
        """
        Args:
            x: input tensor of shape (Nt, n_patches, d_in)
        
        Returns:
            out: output tensor of shape (Nt, n_patches, latent_dim)
            attn_weights: attention probability matrix (Nt, heads, n_patches, n_patches)
        """
        Nt, np, _ = x.shape

        # Calculate Q, K, V
        q = self.q(x) # (Nt, np, d_model)
        k = self.k(x) # (Nt, np, d_model)
        v = self.v(x) # (Nt, np, d_model)

        # Reshape into heads: (Nt, np, heads, d_head) -> (Nt, heads, np, d_head)
        q = rearrange(q, "Nt np (h d) -> Nt h np d", h=self.heads)
        k = rearrange(k, "Nt np (h d) -> Nt h np d", h=self.heads)
        v = rearrange(v, "Nt np (h d) -> Nt h np d", h=self.heads)

        # Scaled Dot-Product Attention
        # (Nt, h, np, d) @ (Nt, h, d, np) -> (Nt, h, np, np)
        scale = 1.0 / jnp.sqrt(self.d_head).astype(self.param_dtype)
        attn_logits = jnp.einsum("thid,thjd->thij", q, k) * scale

        # Add Relative Positional Bias if equivariant
        if self.is_equivariant and self.full_bias is not None:
            # full_bias shape: (heads, np, np) -> broadcast to (Nt, heads, np, np)
            attn_logits = attn_logits + self.full_bias[None, ...]

        # Softmax to get probabilities
        attn_weights = nn.softmax(attn_logits, axis=-1)

        # Apply attention to values
        # (Nt, h, np, np) @ (Nt, h, np, d) -> (Nt, h, np, d)
        out = jnp.einsum("thij,thjd->thid", attn_weights, v)

        # Recombine heads
        out = rearrange(out, "Nt h np d -> Nt np (h d)")

        # Output projection
        out = self.o(out)
        
        return out, attn_weights

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
        # Switched to SelfAttention
        self.attn = SelfAttention(n_patches=self.n_patches, 
                                  d_model=self.d_model, 
                                  heads=self.heads, 
                                  b=self.b,
                                  graph=self.graph, 
                                  is_equivariant=self.is_equivariant,
                                  use_uniform_init=self.use_uniform_init,
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
        # SelfAttention returns (output, attention_weights)
        x_att, att = self.attn(self.layer_norm_1(x))
        x = x + x_att
        x = x + self.ff(self.layer_norm_2(x))
        return x, att


class Encoder_SA(nn.Module):
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
            # Capture the dynamic attention weights
            _, attn_map = l(x)
            attention_maps.append(attn_map)
            x, _ = l(x)
        return attention_maps