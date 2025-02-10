from MDC.jacks import jax

def hank(X, lags):
    """
    Less user friendly but MUCH faster
     - lags as explicit array i.e [0,1,2,3,..]
     - use vmap to go accross multiple input/output dimensions
         - i.e. @partial(jax.vmap, in_axes=(1, 0), out_axes=(1, None))
         - all lags accross every dim: jax.vmap(hank.hank, in_axes=(1, None), out_axes=(0, None)) 
         
    """
    off = X.shape[0] - lags.shape[0]
    _, H = jax.lax.scan(
        lambda i, _: (i + 1, X[i + 1][::-1]), lags - 2, None, length=off
    )
    return H, -off
