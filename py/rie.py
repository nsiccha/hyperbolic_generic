from .base import *

i, j = Symbol.symbols(r'i j')
ih, jh = i.hat, j.hat

g = Tensor('g', 2, 0)
gi = g.inv

pd = Tensor('\partial', 0, 1)

Gamma = Tensor('Gamma', 1, 2)
christoffel_eq = Equation(
    'Christoffel symbols of the second kind',
    Gamma[k,i,j],
    Frac(1,2)*g[k,l]*(
        pd[i] @ gi[j,l] +
        pd[j] @ gi[i,l] -
        pd[l] @ gi[i,j]
    )
)

R = Tensor('R', 1, 3)
R_eq = Equation(
    'Riemannian curvature endomorphism',
    R[l,i,j,k],
    ASymm(
        pd[j] @ Gamma[l,i,k] + Gamma[m,i,k]*Gamma[l,j,m],
        i,j
    )
)
Rm = Tensor('Rm', 0, 4)
Rm_eq = Equation(
    'Riemannian curvature tensor',
    Rm[i,j,k,l],
    R[n,i,j,k]*gi[l,n],
)
flat_eq = Equation(
    'flatness criterion',
    Rm[i,j,k,l],
    0
)
# trafo_eq = Equation(
#     'transformation'
# )

trafo_g_eq = Equation(
    'transformed metric',
    gi[ih, jh],
    u[i, ih] * gi[i,j] * u[j, jh]
)

h = Tensor('h', 0, 2)
trafo_hessian_eq = Equation(
    'transformed hessian',
    h[ih, jh],
    u[i, ih] * h[i,j] * u[j, jh] + h[i] * u[i, ih, jh]
)


text = rf"""
# Differential geometry
## Manifolds
## Tensor fields
## The Lie derivative
## Metrics
## Connections
## Covariant derivatives and geodesics
## Curvature
## Torsion
## The Levi-Civita connection

For now, we will use this as a dump for formulae and definitions:

{christoffel_eq.m}

{R_eq.m}

{Rm_eq.m}

{flat_eq.m}

{trafo_g_eq.m}

{trafo_hessian_eq.m}
"""

rie = Def(rf'A metric {g(u)} is called _Riemannian_ if it is everywhere positive definite.', 'Riemannian')
prie = Def(rf'A metric {g(u)} is called _pseudo-Riemannian_ if it is everywhere non-degenerate.', 'pseudo-Riemannian')

text += rf"""
Following definitions just a test for back referencing.

{rie}

{prie}
"""

T, D, p, N, X, Y, pi, sigma, E, U, Id, Phi = Symbol.symbols(r'T D p N X Y \pi \sigma E U \mathrm{Id} \Phi')

p = Definition('point', p, Contains(p, M))
q = Definition('point', 'q', Contains(q, U))
D = Definition('smooth distribution', 'D')
M = Definition('smooth manifold', M)
UM = Definition('open subset', U, SubsetEq(U, M))
TM = Definition('tangent bundle', T*M)
lie = Definition('Lie bracket', Bra(Comma(X, Y)))
f = Definition('smooth function', f, Contains(f, Cinf(M)))
#
E = Definition('topological space', E)
kbundle = Definition(rf'vector bundle of rank $k$ over {M}')
bundle = Definition('vector bundle', E, Maps(pi, E, M))
cmap = Definition('continuous map', pi, Maps(pi, M, N))
cmapi = Definition(cmap.label, sigma, Maps(sigma, N, M))
homeo = Definition('homeomorphism', Phi, Maps(Phi, Inv(pi)(U), Times(U, Reals**k)))

section_def = Def(rf"""Consider a {cmap.lm}. A _section_ of {cmap} is a continuous right inverse for {cmap}, i.e. a {cmapi.lm} such that {After(cmap, cmapi)==Id[N]}.  A _local_ section need only be defined on some {UM.lm}.""")

bundle_def = Def(rf"""Let {M} be a topological space. A (real) {kbundle.label} is a {E.lm} together with a surjective {cmap.label} {bundle.lexpr} satisying the following conditions:

* For each {p.lm}, the fiber {E[p]==Inv(pi)(p)} over {p} is endowed with the structure of a $k$-dimensional real vector space.
* For each {p.lm}, there exist a neighborhood {UM} of {p} in {M} and a {homeo.lm} satisfying
    * {After(pi[U], homeo)==pi} and
    * for each {q.lexpr}, the restriction of {homeo} to {E[q]} is a vector space isomorphism from {E[q]} to {Times(MySet(q), Reals**k)}.

""")

bundle_section_def = Def(rf"""Consider a {bundle.lm}. A _section_ of {bundle} is a section of the map {pi.l}. A _local_ section need only be defined on some {UM.lm}, while a _global_ section is defined on all of {M}.""")

lie_def = Def(rf"""The _Lie bracket_ of two smooth vector fields {X.l} and {Y.l} is defined by {Eq(lie(f), X(Y(f))-Y(X(f))).bl} for every {f.lm}.""")

distribution_def = Def(rf"""A {D.lm} on a {M.lm} of rank $k$ is a rank-$k$ smooth subbundle of the {TM.lm}.""")

involutive_def = Def(rf"""A {D.lm} on {M.l} is said to be _involutive_ if given any pair of smooth local sections of D, their Lie bracket is also a local section of D.""")

integrable_manifold_def = Def(rf"""A nonempty immersed submanifold {SubsetEq(N,M)} is called an _integral manifold_ of {D.l} if {T[p]*N==D[p]} at each point {Contains(p, N)}.""")

integrable_def = Def(rf"""A smooth distribution {D.l} on {M.l} is said to be _integrable_ if each point of {M.l} is contained in an integral manifold of {D.l}.""")

cintegrable_def = Def(rf"""A {D.lm} is said to be _completely integrable_ if there exists a flat chart for {D} in a neighborhood of each {p.lm}.""")

integrable_prop = Proposition(rf"""Every involutive distribution is completely integrable and vice versa.""", 'See e.g. [Lee2012], theorem 19.12.')

foliation = Definition('foliation', r'\mathcal{F}')
kfoliation = Definition(f'foliation of dimension k on {M}', foliation)
phi = Symbol('\phi')
chart = Definition('smooth chart', Tuple(U, phi))
x = Tensor('x', 1, 0)
c = Tensor('c', 1, 0)

flat_chart_def = Def(rf"""Consider a {M.lm} and let {foliation} be any collection of $k$-dimensional submanifolds of M. A {chart.lm} is said to be _flat for {foliation}_ if {phi(U)} is a cube in {Reals**d} and each submanifold in {foliation} intersects {U} in either the empty set or a countable union of $k$-dimensional slices of the form {x[i]==c[i]} for {i==Comma(k+1,dots,n)}.""")

foliation_def = Def(rf"""A collection {foliation} of disjoint, connected, nonempty, immersed $k$-dimensional submanifolds of {M}, whose union is {M}, such that in a neighborhood of each {p.lm} there exists a flat chart for {foliation} is called _{kfoliation.label}_.""")

text += rf"""
All following definitions collected from [Lee2012].

{section_def}

{bundle_def}

{bundle_section_def}

{lie_def}

{distribution_def}

{involutive_def}

{integrable_manifold_def}

{integrable_def}

{flat_chart_def}

{cintegrable_def}

{integrable_prop}

{foliation_def}
"""
