# %%
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor, solve_triangular


# ## helpers


def pd2(a, b):
    return ((a[:, None] - b) ** 2).sum(-1)


def thexp(theta):
    return {k: jnp.exp(v) for k, v in theta.items()}


# %% kernels


def Lin(a, b, theta):
    return theta["sf_lin"] * (a[:, None] * b).sum(-1)


def SE(a, b, theta):
    return theta["sf_se"] * jnp.exp(-0.5 * pd2(a, b) / theta["ll"] ** 2)


def Mat23(a, b, theta):
    d = pd2(a, b) ** 0.5
    t1 = jnp.sqrt(3) * d / theta["ll"] ** 2
    return theta["sf_m32"] * (1 + t1) * jnp.exp(-t1)


def Mat52(a, b, theta):
    d = pd2(a, b) ** 0.5
    t1 = jnp.sqrt(5) * d / theta["ll"] ** 2
    return theta["sf_m52"] * (1 + t1 + (t1**2) / 3) * jnp.exp(-t1)


# %% GP helpers


def GP(X, Y, kernel, mu=lambda x: jnp.zeros((x.shape[0], 1))):
    mux = mu(X)

    @jax.jit
    def train(theta):
        theta = thexp(theta)
        L, _ = cho_factor(
            kernel(X, X, theta) + theta["sn"] * jnp.eye(X.shape[0]),
            overwrite_a=0,
            lower=1,
        )
        alpha = cho_solve((L, 1), Y - mux)
        return {"L": L, "alpha": alpha, "theta": theta}

    @jax.jit
    def predict_f(Xp, state, vp=None):
        Kstar = kernel(X, Xp, state["theta"])
        return mu(Xp) + Kstar.T @ state["alpha"]

    @jax.jit
    def predict(Xp, state, vp=None):
        Kstar = kernel(X, Xp, state["theta"])
        yp = Kstar.T @ state["alpha"]
        v = cho_solve((state["L"], 1), Kstar)
        vp = kernel(Xp, Xp, state["theta"]) - Kstar.T @ v
        return mu(Xp) + yp, vp

    @jax.jit
    def nlml(theta):
        state = train(theta)
        t1 = 0.5 * ((Y - mux).T @ state["alpha"]).reshape(())
        t2 = jnp.log(jnp.diag(state["L"])).sum()
        t3 = 0.5 * Y.shape[0] * jnp.log(2 * jnp.pi)
        return t1 + t2 + t3

    return train, predict_f, predict, nlml

# %% Sparse approximation (FITC)

def FITC(X, Y, U=None, kernel=SE):
    jitter = 1e-12
    Uglobal=U

    @jax.jit
    def getU(theta):
        if 'U' in theta:
            U = theta['U']
        else: 
            U = Uglobal
        return U
    
    @jax.jit
    def train(theta):
        U = getU(theta) # do not exp U
        theta = thexp(theta)
        Kuu = kernel(U, U, theta)
        Luu = jnp.linalg.cholesky(Kuu + jitter * jnp.eye(Kuu.shape[0]))
        Kfu = kernel(X, U, theta)
        Lff = solve_triangular(Luu, Kfu.T, lower=1).T
        Lam_vec = theta["sf_se"] - (Lff**2).sum(1) + theta["sn"]
        LL = jnp.hstack([Luu, Kfu.T * jnp.sqrt(1 / Lam_vec[None, :])])
        R = jnp.linalg.qr(LL.T, mode="r")
        RI = solve_triangular(R, jnp.identity(R.shape[0]))
        alpha = (RI @ RI.T) @ Kfu.T @ (1 / Lam_vec * jnp.squeeze(Y))

        return {
            "theta": theta,
            "alpha": alpha,
            "lam": Lam_vec,
            "Luu": Luu,
            "Kfu": Kfu,
            "R": R,
            "RI": RI,
        }

    @jax.jit
    def predict_f(Xp, state):
        U = getU(state['theta']) 
        K_star_u = kernel(Xp, U, state["theta"])
        return K_star_u @ state["alpha"]

    @jax.jit
    def predict(Xp, state):
        U = getU(state['theta'])    
        K_star_u = kernel(Xp, U, state["theta"])
        K_star_star = kernel(Xp, Xp, state["theta"])
        Q_star_star = solve_triangular(state["Luu"], K_star_u.T, lower=1)
        KR = K_star_u @ state["RI"]
        return K_star_u @ state["alpha"], K_star_star + Q_star_star - KR @ KR.T

    @jax.jit
    def nlml(theta):
        state = train(theta)
        # quinonero-candela05a.dvi eqn30
        lam = state["lam"]
        Luu = state["Luu"]
        Kfu = state["Kfu"]
        R = state["R"]
        RI = state["RI"]

        # square term
        yh = (Y.T * lam[None] ** 0.5).T
        t1 = 0.5 * (Y.T @ yh - ((yh.T @ Kfu @ RI) ** 2).sum())

        # determinant term
        t2 = (
            0.5 * jnp.log(lam).sum()
            - jnp.log(jnp.diag(Luu)).sum()
            + jnp.log(jnp.abs(jnp.diag(R))).sum()
        )

        # constant term
        t3 = 0.5 * Y.shape[0] * jnp.log(2 * jnp.pi)

        return (t1 + t2 + t3).reshape(())

    return train, predict_f, predict, nlml
