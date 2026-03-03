import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
class OuputHead(nn.Module):
    d_model: int
    d_latent: int
    d_output: int 
    out_activation: Any
    make_it_invariant: bool = False
    is_equivariant: bool = False
    param_dtype: Any = jnp.float64
    is_complex: bool = False
    initializer: Any = nn.initializers.lecun_normal()

    def setup(self):

        self.layer_norm = nn.LayerNorm(
            param_dtype=self.param_dtype
            )
        
        self.out_layer_norm = nn.LayerNorm(
            param_dtype=self.param_dtype
            )
        
        self.dense_expand = nn.Dense(
            self.d_latent,
            param_dtype=self.param_dtype, 
            kernel_init=self.initializer,
            bias_init=jax.nn.initializers.zeros,
        )

        if self.is_complex:
            self.norm_amp = nn.LayerNorm(
                param_dtype=self.param_dtype,
                )
            self.norm_sign = nn.LayerNorm(
                param_dtype=self.param_dtype
                )

            self.output_layer_amp = nn.Dense(
                self.d_output,
                param_dtype=self.param_dtype, 
                kernel_init=self.initializer, 
                bias_init=jax.nn.initializers.zeros
                )
            
            self.output_layer_sign = nn.Dense(
                self.d_output,
                param_dtype=self.param_dtype, 
                kernel_init=self.initializer, 
                bias_init=jax.nn.initializers.zeros
                )
            
        else:
            self.output_layer = nn.Dense(
                self.d_output,
                param_dtype=self.param_dtype,
                kernel_init=self.initializer,
                bias_init=jax.nn.initializers.zeros,
            )

    def __call__(self, x):

        if self.make_it_invariant:

            x = self.layer_norm(x.sum(axis=-2)) # (N_T, N_P, d) -> (N_T, d)
            x = self.dense_expand(x) # (N_T, d) -> (N_T, N_lat)
            x = nn.gelu(x)
            x = self.out_layer_norm(x.sum(axis=-2)) # Pool over the translation dimension (N_T, N_lat) -> (N_lat)
            x = x.flatten()

            if self.is_complex:

                amp = self.norm_amp(self.output_layer_amp(x))
                sign = self.norm_sign(self.output_layer_sign(x))
                out = amp + 1j*sign # Pensa se mettere un layer norm

            else: 

                out = self.output_layer(x)

        elif self.is_equivariant:

            x = x.reshape(-1, self.d_model) # (N_P, N_T, d) -> (N_o, d)
            x = self.dense_expand(x) # (N_o, d) -> (N_o, N_lat)
            x = nn.gelu(x)
            out = self.out_layer_norm(self.output_layer(x)).flatten() # (N_o, N_lat) -> (N_o, N_fermions)
        else:

            x = self.dense_expand(x)
            x = nn.gelu(x)
            x = self.layer_norm(x.sum(axis=-2)) # (N_P, d) -> (d)
            if self.is_complex:
                amp = self.norm_amp(self.output_layer_amp(x.flatten()))
                sign = self.norm_sign(self.output_layer_sign(x.flatten()))
                out = amp + 1j*sign # Pensa se mettere un layer norm
            else:
                out = self.out_layer_norm(self.output_layer(x.flatten()))
            
        return self.out_activation(out)

        # if self.make_it_invariant:

        #     x = self.layer_norm(x.sum(axis=-2)) # (N_T, N_P, d) -> (N_T, d)
        #     x = self.dense_expand(x) # (N_T, d) -> (N_T, N_lat)
        #     x = nn.gelu(x)
        #     x = self.out_layer_norm(x.sum(axis=-2)) # Pool over the translation dimension (N_T, N_lat) -> (N_lat)
        #     x = x.flatten()

        #     if self.is_complex:

        #         amp = self.norm_amp(self.output_layer_amp(x))
        #         sign = self.norm_sign(self.output_layer_sign(x))
        #         out = amp + 1j*sign # Pensa se mettere un layer norm

        #     else: 

        #         out = self.output_layer(x)

        # elif self.is_equivariant:

        #     x = self.dense_expand(x) # (N_P, d) -> (N_P, N_lat)
        #     x = nn.gelu(x)
        #     out = self.out_layer_norm(self.output_layer(x)).flatten() # (N_P, N_lat) -> (N_P, N_fermions)

        # else:

        #     x = self.layer_norm(x.sum(axis=-2)) # (N_P, d) -> (d)
        #     out = self.out_layer_norm(self.output_layer(x.flatten()))
            
        # return self.out_activation(out)
class OuputHead_Luca(nn.Module):
    d_model: int
    d_latent: int
    out_activation: Any
    is_complex: bool = True
    is_equivariant: bool = False
    param_dtype: Any = jnp.float64
    initializer: Any = nn.initializers.lecun_normal()

    def setup(self):

        self.out_layer_norm = nn.LayerNorm(
            param_dtype=self.param_dtype
            )

        self.dense_expand = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype, 
            kernel_init=self.initializer,
            bias_init=jax.nn.initializers.zeros,
        )

        self.norm_amp = nn.LayerNorm(
            use_scale=True, 
            use_bias=True, 
            param_dtype=self.param_dtype,
            )
        self.norm_sign = nn.LayerNorm(
            use_scale=True, 
            use_bias=True, 
            param_dtype=self.param_dtype
            )
        
        self.output_layer_amp = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype, 
            kernel_init=self.initializer, 
            bias_init=jax.nn.initializers.zeros
            )
        
        self.output_layer_sign = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype, 
            kernel_init=self.initializer, 
            bias_init=jax.nn.initializers.zeros
            )

        self.output_layer = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=self.initializer,
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):

        if self.is_complex:

            if (self.is_equivariant) and (self.make_it_invariant):

                x = self.dense_expand(x) # (N_batch, N_T, N_P, N_features * 2)
                x = nn.gelu(x)
                x = x.sum(axis=(-3,-2)) # Pool over the spatial dimensions
                x = x.flatten()

                amp = self.norm_amp(self.output_layer_amp(x))
                sign = self.norm_sign(self.output_layer_sign(x))
                out = amp + 1j*sign

            else: 
                x = self.out_layer_norm(x.sum(axis=-2))
                amp = self.norm_amp(self.output_layer_amp(x.flatten()))
                sign = self.norm_sign(self.output_layer_sign(x.flatten()))
                out = amp + 1j*sign

        else:
            raise ValueError("Luca's output head is only implemented for complex output")

        return jnp.sum(self.out_activation(out), axis=-1)
