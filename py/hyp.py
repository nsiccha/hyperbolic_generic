# from .base import *
from .base import *

sig = Tensor('sigma', 1, 0)
A = Tensor('A', 2, 1)
Flux = Tensor('F', 2, 0)
xi = Symbol('xi')
S = Tensor(r'\mathcal{S}', 1, 0)
Sd = S[d-1]
q = Tensor('q', 1, 0)
a_, b_, c_ = Wild.symbols('a b c')

qpde = Equation(
    'system of quasilinear partial differential equations',
    Add(A[0,i,j]*u[j, t](x), A[mu, i, j](u(x)) * u[j, mu](x)),
    sig[i](u(x))
)

hyp = Definition(
    'hyperbolic',
    A[mu, i, j] * xi[mu]
)
eta = Symbol('eta')
entropy = Definition(
    'entropy function',
    Contains(eta, Cinf(U, Reals))
)
eeb = Definition(
    'additional balance law',
    # Geq(
    Eq(
        PDiff(eta, u[i]) * qpde.lhs,
        eta[t](u(x)) + q[mu, mu](u(x))
    ),
    0
    # )
)
compc = Equation(
    'compatibility condition',
    PDiff(q[mu], u[j]),
    PDiff(eta, u[i]) * A[mu, i, j]
)

qflux = Definition(
    'entropy-flux function',
    Contains(q, Cinf(U, Reals**d))
)
fluxes = Definition(
    'flux functions',
    Contains(Flux[i], Cinf(U, Reals**d)),
    Eq(PDiff(Flux[mu, i], u[j]), A[mu, i, j])
)
fluxes_existence = Equation(
    '',
    ASymm(PDiff(A[mu, i, j], u[k]), j, k),
    0
)
qflux_existence = Equation(
    '',
    ASymm(PDiff(compc.rhs, u[k]), j, k),
    0
)
Eta = Equation(
    'entropy functional',
    Symbol('Eta'),
    sp.Integral(eta(u(x)), x)
)
v, w = Symbol.symbols('v w')
skewL = Equation(
    'skew-symmetric bilinear form',
    L(v, w),
    # Eq(L(u, v), -L(v, u)),
    Dot(v[i], L_op.rhs * w[j]),
    Dot(v[i], b[mu,i,j,k] * u[k,mu]*w[j]) +
    Dot(v[i], g[mu,i,j] * w[j, mu])
)
deta = PDiff(eta, u)
skewpde = Equation(
    'evolution equation',
    Eq(u[i,t], Sup(L * deta, i)),
    (
        g[mu,i,j] * eta[j, k] +
        b[mu,i,j,k] * eta[j]
    ) * u[k,mu]
)
skewcond = Equation(
    '',
    2 * L(v, w),
    2*Dot(v[i], (
        b[mu,i,j,k]
    )*u[k,mu]*w[j] + g[mu,i,j] * w[j, mu]),
    L(v, w) - L(w, v),
    2*ASymm(Dot(v[i], b[mu,i,j,k]*u[k,mu]*w[j]), v, w) +
    ASymm(Dot(v[i], g[mu,i,j]*w[j, mu]), v, w).doit(),
    2*Dot(v[i], ASymm(b[mu,i,j,k], i, j)*u[k,mu]*w[j]) +
    Dot(v[j], PDiff(g[mu,i,j] * w[i], x[mu])) +
    Dot(v[i], g[mu,i,j] * w[j, mu]),
    2*Dot(v[i], ASymm(b[mu,i,j,k], i, j)*u[k,mu]*w[j]) +
    Dot(v[j], PDiff(g[mu,i,j], u[k]) * u[k, mu] * w[i]) +
    2*Dot(v[i], Symm(g[mu,i,j], i, j) * w[j, mu]),
    Dot(v[i], Underbrace(
        Par(2*ASymm(b[mu,i,j,k], i, j) +
        PDiff(g[mu,j,i], u[k])),
        2*b[mu,i,j,k]
    )*u[k,mu]*w[j] + Underbrace(
        2*Symm(g[mu,i,j], i, j),
        2*g[mu,i,j]
    ) * w[j, mu]),

    # Dot(v[i], L_op.rhs * w[j]) - Dot(w[i], L_op.rhs * v[j]),
    # skewL.args[-1] -
    # Dot(w[i], b[mu,i,j,k] * u[k,mu]*v[j]) -
    # Dot(w[i], g[mu,i,j] * v[j, mu]),
    # Dot(v[i], 2*ASymm(b[mu,i,j,k], i, j)*u[k,mu]*w[j]) +
    # Dot(v[i], g[mu,i,j] * w[j, mu]) +
    # Dot(v[j], PDiff(g[mu,i,j] * w[i], x[mu])),
    # Dot(v[i], ASymm(b[mu,i,j,k], i, j)*u[k,mu]*w[j]) +
    # Dot(v[i], 2*Symm(g[mu,i,j], i, j) * w[j, mu]) +
    # Dot(v[j], PDiff(g[mu,i,j], u[k]) * u[k, mu] * w[i]),
    # Dot(v[i], (
    #     2*ASymm(b[mu,i,j,k], i, j) +
    #     PDiff(g[mu,j,i], u[k])
    # )*u[k,mu]*w[j]) +
    # Dot(v[i], 2*Symm(g[mu,i,j], i, j) * w[j, mu])

    # skewL.args[-1],# - Rep({u: v, v: u})(skewL.args[-1]),
    # Dot(u[i], Symm(b[mu,i,j,k], i,j)*u[k,mu]*v[j])
    # Dot(v[i], L_op.rhs * u[j]),
    # Dot(u[i], L_op.rhs * v[j]) +
    # Dot(v[j], Rep({i: j, j: i})(L_op.rhs) * u[i]),
    # Dot(u[i], (L_op.rhs + Rep({i: j, j: i})(L_op.rhs)) * v[j])


)
skeweeb = Equation(
    'additional local conservation law',
    eta[t](u(x)),
    deta[i] * skewpde.rhs
    # deta[i] * L_op.rhs * deta[j],
    # deta[i] * (
    #     b[mu,i,j,k] * deta[j] +
    #     g[mu,i,j] * PDiff(eta, u[j], u[k])
    # ) * u[k, mu]
)
skewq = Equation(
    'gradient of the entropy-flux function',
    PDiff(q[mu], u[k]),
    deta[i] * (
        b[mu,i,j,k] * deta[j] +
        g[mu,i,j] * PDiff(eta, u[j], u[k])
    )
)
skewqex1 = Equation(
    '',
    ASymm(PDiff(skewq.lhs, u[l]), k, l),
    ASymm(PDiff(skewq.rhs, u[l]), k, l),
    ASymm(
        Underbrace(
            eta[i,l]*b[mu,i,j,k]*eta[j],
            eta[i]*b[mu,j,i,k]*eta[j,l]
        ) +
        Underbrace(
            eta[i,l]*g[mu,i,j]*eta[j,k],
            ASymm(Par(), k, l) == 0
        ) + dots,
        k,l
    ),
    ASymm(
        dots + eta[i] * (
            PDiff(b[mu,i,j,k]*eta[j], u[l]) +
            Underbrace(
                PDiff(g[mu,i,j]*eta[j,k], u[l]),
                ASymm(Par(), k, l) == ASymm(g[mu,i,j,l]*eta[j,k], k, l)
            )
        ),
        k,l
    ),
    ASymm(
        eta[i] * (
            b[mu,j,i,k]*eta[j,l] +
            PDiff(b[mu,i,j,k]*eta[j], u[l]) +
            g[mu,i,j,l]*eta[j,k]
        ),
        k, l
    ),
    ASymm(
        eta[i] * (
            Underbrace(
                2*Symm(b[mu,j,i,k], i, j),
                g[mu,i,j,k]
            )*eta[j,l] +
            PDiff(b[mu,i,j,k], u[l])*eta[j] +
            g[mu,i,j,l] * eta[j,k]
        ),
        k, l
    ),
    ASymm(
        Underbrace(
            eta[i] * g[mu,i,j,k] * eta[j,l] +
            eta[i] * g[mu,i,j,l] * eta[j,k],
            ASymm(Par(), k, l) == 0
        ) +
        Underbrace(
            eta[i] * PDiff(b[mu,i,j,k], u[l]) * eta[j],
            Eq(
                blank,
                eta[i] * Symm(PDiff(b[mu,i,j,k], u[l]), i, j) * eta[j],
                Frac(1,2)*eta[i] * PDiff(g[mu,i,j,k], u[l]) * eta[j]
            )
        ),
        k,l
    ),
    ASymm(
        Frac(1,2)*eta[i] * PDiff(g[mu,i,j], u[k], u[l]) * eta[j],
        k,l
    ),
    0
)
skewqex = Equation(
    '',
    ASymm(PDiff(skewq.lhs, u[l]), k, l),
    ASymm(PDiff(skewq.rhs, u[l]), k, l),
    # REPLACE(
    #     ASymm(PDiff(skewq.rhs, u[l]), k, l),
    #     PDiff.product_rule,
    #     PDiff.summation_rule,
    #     PDiff.product_rule,
    #     {a_*(b_+c_): a_*b_ + a_*c_},
    #     # (ASymm('a_', 'b_', 'c_'), lambda a, b, c: a - a.subs({b: c, c: b}))
    # ).steps[-1],
    eta[i] * (
        ASymm(b[mu,i,j,k] * eta[j,l], k, l) +
        eta[j] * ASymm(
            Symm(
                PDiff(b[mu,i,j,k], u[l]),
                i, j
            ),
            k, l
        ) +
        ASymm(g[mu,i,j,l] * eta[j,k], k, l) +
        ASymm(eta[j,l] * b[mu,j,i,k], k, l)
    ),
    eta[i] * (
        ASymm(
            eta[j, l] * 2 * Symm(b[mu,i,j,k], i, j),
            k, l
        ) +
        ASymm(g[mu,i,j,l] * eta[j,k], k, l) +
        eta[j]/2 * ASymm(
            PDiff(g[mu,i,j], u[k], u[l]),
            k, l
        )
    ),
    eta[i] * (
        ASymm(g[mu,i,j,k] * eta[j,l], k, l) +
        ASymm(g[mu,i,j,l] * eta[j,k], k, l)
    ),
    0
    # Rep(PDiff.product_rule)(PDiff(skewq.rhs, u[l]))
)
skewa = Equation(
    '',
    A[mu, i, k],
    (
        g[mu,i,j] * eta[j, k] +
        b[mu,i,j,k] * eta[j]
    )
)
skewfex = Equation(
    '',
    ASymm(PDiff(skewa.lhs, u[l]), k, l),
    *REPLACE(
        ASymm(PDiff(skewa.rhs, u[l]), k, l),
        PDiff.summation_rule,
        PDiff.product_rule,
    ).steps,
    ASymm(
        b[mu,j,i,l] * eta[j,k] +
        Underbrace(
            b[mu,i,j,l] * eta[j,k] +
            b[mu,i,j,k] * eta[j,l],
            ASymm(Par(), k, l) == 0
        ) +
        eta[j] * PDiff(b[mu,i,j,k], u[l]),
        k, l
    ),
    ASymm(
        PDiff(b[mu,j,i,l] * eta[j], u[k]) -
        PDiff(b[mu,j,i,l], u[k]) * eta[j] +
        eta[j] * PDiff(b[mu,i,j,k], u[l]),
        k, l
    ),
    ASymm(
        PDiff(b[mu,j,i,l] * eta[j], u[k]) +
        eta[j] * PDiff(g[mu,i,j], u[k], u[l]),
        k, l
    ),
    ASymm(
        PDiff(b[mu,j,i,l] * eta[j], u[k]),
        k, l
    )
    # ASymm()
    # PDiff(
    #     b[mu,j,i,k] * eta[j],
    #     u[l]
    # ) -
)
skewhyp = Equation(
    '',
    skewfex.args[-1],
    0
)

xi = Definition('direction', xi, Contains(xi, Sd))

qpde_def = Def(rf"""We consider _systems of quasilinear first-order partial differential equations_ {qpde.bl}.""", 'system of quasilinear first-order PDEs')

hpde_def = Def(rf"""A {qpde_def.long_ref} is called _hyperbolic_ if the eigenvalue problem {Eq(
    Bra(A(xi)-lam * A[0])*v, 0
).bl} has real eigenvalues and $n$ linearly independent eigenvectors for every {xi.lexpr}, where {Comma(
    Maps(A, Sd, Tensors(1,1)*tspace),
    A(xi) == A[mu]*xi[mu]
).bl}. If {A[0] == Id} this translates to {A(xi)} being diagonalizable over the real numbers for every {xi.lm}.""", 'hyperbolic')

pd = Symbol('\partial')
F = Tensor('F', 2, 0)
Fs = Definition('flux functions', Contains(F[mu], Gamma(Vectors(tspace))))
bpde_def = Def(rf"""A {qpde_def.long_ref} is called a system of _balance laws_ if there exist smooth {Fs.lm} such that {Eq(
    PDiff(F[mu,i], u[j]), A[mu,i,j]
).bl} for $\mu=0,\dots,d$. If {sig == 0} it is called a system of _conservation laws_.""", 'system of balance laws')

spde_def = Def(rf"""A {qpde_def.long_ref} is called _symmetric hyperbolic_ if {A[0]} is symmetric positive definite and all {A[mu]} are symmetric. A system that can be brought into this form is called _symmetrizable_.""", 'symmetrizable')
symmetrizable = Ref(spde_def, 'symmetrizable')
symmetrizer = Ref(spde_def, 'symmetrizer')


eta, q = Symbol.symbols('\eta q')
eta = Definition('entropy density', eta)
eef = Definition(
    'entropy-entropy flux pair',
    Tuple(eta, q)
)
eef_def = Def(rf"""A pair {eef} is called an _entropy-entropy flux pair_ to the {qpde_def.long_ref} if the {eeb.m} is implied by {qpde.lref}.""", eef.lm)

eef_lemma = Lemma(rf"""A {qpde_def.long_ref} admits an {eef_def.long_ref} if the {compc.m} is satisfied.""")

compc2 = Equation(
    'integrability condition',
    ASymm(
        PDiff(compc.rhs, u[k]),
        j, k
    ),
    0
)
eef_lemma2 = Lemma(rf"""Given an {eta.lm}, a {qpde_def.long_ref} admits an {eef_def.long_ref} if and only if the {compc2.m} is satisfied.""", rf"""Follows from {eef_lemma.lref} combined with the requirement {Eq(
    ASymm(
        PDiff(q[mu], u[j], u[k]),
        j, k
    ),
    ASymm(
        PDiff(compc.rhs, u[k]),
        j, k
    ),
    0
).ibl} and from the assertion that the {tspace.lm} is simply connected (see e.g. [Zori2016], section 14.3).""")

spde_lemma = Lemma(rf"""A {bpde_def.long_ref} admitting an {eef_def.long_ref} with strictly convex {eta.lm} is {spde_def.long_ref}.""", 'See e.g. [Frie1971].')

cpde_lemma = Lemma(rf"""A {qpde_def.long_ref} can be written as a {bpde_def.long_ref} in original coordinates $u$ if {fluxes_existence.bl} holds. WILL ADD CONTENT FROM [bogo2011]. """, rf"""
Follows from the requirement {Eq(
    ASymm(
        PDiff(F[mu, i], u[j], u[k]),
        j, k
    ),
    fluxes_existence.lhs,
    0
).ibl} and from the assertion that the {tspace.lm} is simply connected.""")
# If ADD CRITERIA, coordinates {u.hat(v)} exist such that the system can be written in balance law form in these new coordinates.


text = rf"""# Hyperbolic partial differential equations
The main references are

* [Benz2006] Standard hyp. PDE Book
* [Godu1962] Ex. solutions
* [Rugg2006] Ex. solutions
* [Frie1971] Convex entropy extension
* [Tadm1984] Skew-selfadjoint hyp. PDEs
* [Pesh2018] GENERIC <-> SHTC
* [Pave2020] Hamiltonian continuum mechanics

Throughout, unless otherwise stated, we will use a {xspace.lm}, a simply connected {tspace.lm} and assume {Contains(u(t), uspace)} for all $t$ considered. Furthermore, indices in products which appear once above and once below will be summed. Greek indices like {Comma(mu,nu)} run from $1$ to $d$ and latin indices like {Comma(i,j,k)} run from $1$ to $n$, unless otherwise noted.

{qpde_def}

We are interested in the following subclasses:

{hpde_def}

__Note:__ Unless otherwise noted, we will assume {A[0] == Id}.

{bpde_def}

{spde_def}

{eef_def}

{eef_lemma}

{eef_lemma2}

{spde_lemma}

{cpde_lemma}

It has been shown that if the entropy and the fluxes are _homogeneous_,
symmetrizability, admitting an {eef.label} and having a so called _skew-
selfadjoint form_ are all equivalent (see [Tadm1984], theorem 2.1 for details).
WILL ADD DEFINITION, LEMMA AND DISCUSSION.

A weaker requirement than homogeneity and having a skew-selfadjoint form is the following:"""

asymm_eq = Definition(
    '',
    And(
        Eq(2*Symm(b[mu,i,j,k], i, j), PDiff(g[mu,i,j], u[k])),
        Eq(ASymm(g[mu,i,j], i, j), 0)
    )
)

hyp_lemma = Lemma(rf"""Consider an {Eta.im} and a {skewL.slice[:-1].m}.
Then the {skewpde.m} admits an {eef_def.long_ref}.
If {skewhyp.bl}, it is a {bpde_def.long_ref}.
If the {eta.lm} is strictly convex, the system is {spde_def.long_ref}.""", rf"""First we note that from the skew-symmetry of $L$ it follows that
{skewcond.bl}. Hence, we get {asymm_eq.bl}.

__To check the existence of an entropy-flux function:__
From {skewpde.lref} we get {skeweeb.bl} and thus identify the {skewq.m} and can check the conditions of {eef_lemma2.lref}:
{skewqex1.bl}.

__To check the existence of flux functions:__ From the evolution equation we identify {skewa.l}. We check
{skewfex.bl}
which completes the proof.""")

# hyp_proof = Proof()

# text += hyp_lemma.l# + hyp_proof.l

text += rf"""
{hyp_lemma}

## Symmetric hyperbolic thermodynamically compatible systems
## Skew-selfadjoint hyperbolic systems of balance laws
"""
# text = """
# # Hyperbolic partial differential equations
# The main references are
#
# * [Benz2006] Standard hyp. PDE Book
# * [Godu1962] Ex. solutions
# * [Rugg2006] Ex. solutions
# * [Frie1971] Convex entropy extension
# * [Tadm1984] Skew-selfadjoint hyp. PDEs
# * [Pesh2018] GENERIC <-> SHTC
# * [Pave2020] Hamiltonian continuum mechanics
#
# ## Hyperbolic systems of balance laws
# ## Friedrichs symmetrizable systems of balance laws
# ## Symmetric hyperbolic thermodynamically compatible systems
# ## Skew-selfadjoint hyperbolic systems of balance laws
# """
