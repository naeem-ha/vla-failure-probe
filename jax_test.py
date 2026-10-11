import jax, jax.numpy as jnp

x = jnp.ones((4096, 4096), dtype=jnp.bfloat16)
y = jax.jit(lambda a: (a @ a).sum())(x)
print("JAX_MATMUL_OK", float(y), jax.devices()[0].device_kind)
