import flax.linen as nn
from typing import Any
import jax.numpy as jnp
from functools import partial
import netket as nk

from qchem_utils._src.nets.Embedd import Embed
from qchem_utils._src.nets.FMHAL import Encoder_FMHAL
from qchem_utils._src.nets.Out_layer import OuputHead, OuputHead_Luca

def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)

class ViT_trans_equi(nn.Module):
    n_layers: int
    d_model: int
    d_output: int
    d_latent: int
    heads: int
    b: int
    n_patches: int
    graph: nk.graph = None
    is_equivariant: bool = False
    make_it_invariant: bool = False
    use_uniform_init: bool = False
    separate_spin: bool = False
    out_activation: Any = log_cosh
    complex: bool = False
    is_Luca: bool = False
    param_dtype: Any = jnp.float64
    initializer: Any = nn.initializers.xavier_uniform()
    """
    Should be factored or NLP
    """

    def setup(self):
        self.embedding = Embed(
            d_model=self.d_model,
            b=self.b,
            n_patches=self.n_patches,
            graph=self.graph,
            separate_spin=self.separate_spin,
            is_equivariant=self.is_equivariant,
            param_dtype=self.param_dtype,
            initializer=self.initializer,
        )

        
        self.encoder = Encoder_FMHAL(
            n_patches=self.n_patches,
            b=self.b,
            graph=self.graph,
            n_layers=self.n_layers, 
            d_model=self.d_model, 
            use_uniform_init=self.use_uniform_init,
            heads=self.heads,
            is_equivariant=self.is_equivariant,
            initializer=self.initializer,
        )

        if self.is_Luca:
            self.out = OuputHead_Luca(
                d_model=self.d_model,
                d_latent=self.d_latent,
                out_activation=self.out_activation,
                is_equivariant=self.is_equivariant,
                param_dtype=self.param_dtype,
                initializer=self.initializer,
            )

        else:
            self.out = OuputHead(
                d_model=self.d_model,
                d_latent=self.d_latent,
                d_output=self.d_output,
                out_activation=self.out_activation,
                make_it_invariant=self.make_it_invariant,
                is_equivariant=self.is_equivariant,
                param_dtype=self.param_dtype,
                is_complex=self.complex,
                initializer=self.initializer,
            )

    @nn.compact
    def __call__(self, x):

        d_shape_in = x.shape[-1] # batch shape
        batch_shape_in = x.shape[:-1]
        x = x.reshape(-1, d_shape_in)

        if self.is_Luca:
            @partial(jnp.vectorize, signature="(x)->()")
            def compute_wavefunc(x):
                x = self.embedding(x)
                x = self.encoder(x)
                return self.out(x)
            
        else:   
            @partial(jnp.vectorize, signature="(x)->(n)")
            def compute_wavefunc(x):
                x = self.embedding(x)
                x = self.encoder(x)
                return self.out(x)
            
        out = compute_wavefunc(x)
        out = out.reshape(*batch_shape_in,-1)
            
        return out
