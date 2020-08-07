from IPython.display import display, Markdown
import sympy as sp
import numpy as np
import itertools
# from bib import *
import functools
import operator
import dill
from cached_property import cached_property

def tprint(arg):
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'[{current_time}]: {arg}')

class Printer:
    verbose = False
    def tprint(self, arg):
        if self.verbose:
            tprint(arg)

class Cache(Printer):
    def __init__(self, path, **kwargs):
        import pickle, os
        self.path = path + '/cache'
        self.cache = None
    def load(self):
        import pickle, os
        if os.path.exists(self.path):
            self.tprint(f'Loading cache ({self.hsize})')
        # try:
            try:
                self.cache = pickle.load(open(self.path, "rb"))
                self.tprint(f'Loaded cache')
            except:
                self.cache = dict()
                tprint(f'Failed to load cache')
        # except:
        else:
            self.tprint('Initializing cache')
            self.cache = dict()
    @property
    def size(self):
        import pickle, os
        if os.path.exists(self.path):
            return os.path.getsize(self.path)
        return 0

    @property
    def hsize(self):
        # https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
        num = self.size
        suffix = 'B'
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    def save(self):
        import pickle
        self.tprint(f'Saving cache ({self.hsize})')
        try:
            pickle.dump(self.cache, open(self.path, "wb"))
            self.tprint(f'Saved cache ({self.hsize})')
        except:
            self.tprint(f'Failed saving cache ({self.hsize})')


    def memoize(self): return self
    def __call__(self, func):
        def _func(*args):
            key = func.__qualname__, args
            if self.cache is None:
                self.load()
            if key not in self.cache:
                self.tprint(f'Did not find {key}')
                self.cache[key] = func(*args)
                self.save()
            else:
                self.tprint(f'Did find {key}')
            rv = self.cache[key]
            # tprint('Retrieved value')
            return rv
        _func.__name__ = func.__name__
        return _func

# from diskcache import Cache
cache = Cache('./wtmp', size_limit=10e9)


# exec(open('./bib.py').read())
from collections.abc import Iterable   # drop `.abc` with Python 2.7 or lower

def iterable(obj):
    return isinstance(obj, Iterable)

def doit(obj): return getattr(obj, 'doit', lambda *args: obj)()

class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)

def str_to_slice(x):
    xs = [
        None if x == '' else int(x) for x in x.split(':')
    ]
    return slice(*xs)


class List(list):
    def __str__(self): return ', '.join(map(str, self))
    def __getitem__(self, key):
        if isinstance(key, str):
            key = str_to_slice(key)
        if isinstance(key, slice):
            return List(list.__getitem__(self, key))
        return list.__getitem__(self, key)

def Map(*args, **kwargs): return List(map(*args, **kwargs))

class Slicer:
    def __init__(self, expr):
        self.expr = expr
    def __getitem__(self, key):
        if isinstance(key, list):
            return self.expr.func(*(self.expr.args[i] for i in key))
        return self.expr.func(*self.expr.args[key])

class BasicExpr(sp.Expr, Printer):
    # precedence = 0
    math_env = 'align'
    @property
    def is_scalar(self):
        return all(getattr(arg, 'is_scalar', True) for arg in self.args)
    # @classmethod
    @classproperty
    def tag(cls): return cls.__name__.lower()

    @property
    def ref_label(self):
        label = getattr(self, 'label', '')
        if label == '':
            label = str(hash(self))
        label = label.lower().replace(' ', '-')
        return f'{self.tag}:{label}'
    @property
    def ref(self): return rf'\ref{{{self.ref_label}}}'
    @property
    def lref(self): return rf'{self.tag} {self.ref}'

    @property
    def long_ref(self): return rf'\hyperref[{self.ref_label}]{{{self.long_label}}}'

    show_label = True
    def env(self, tag, content):
        label = getattr(self, 'ref_label', '')
        if label != '':
            label = rf'\label{{{label}}}' + '\n'
        if not self.show_label:
            label = ''
        return rf"""
\begin{{{tag}}}{label}{content}\end{{{tag}}}
"""
    def align(self, math): return self.env(self.math_env, math)
#         label = getattr(self, 'ref_label', '')
#         if label != '':
#             label = rf'\label{{{label}}}' + '\n'
#         return rf"""
# \begin{{{self.math_env}}}{label}{math}\end{{{self.math_env}}}
# """
    is_commutative = True#False
    @property
    def m(self): return self._repr_markdown_()
    @property
    def il(self): return sp.latex(self)
    @property
    def al(self): return self.il
    # def il(self): return self.l[len('$\displaystyle'):-1]
    @property
    def l(self): return rf'${self.il}$'
    # def l(self): return self._repr_latex_()
    @property
    # @cache.memoize()
    def bl(self): return self.align(self.al)
    @property
    def f(self):
        return Contains(self, self.domain)

    def __call__(self, *args):
        return Call(self, *args)
    def __getitem__(self, arg):
        if isinstance(arg, list):
            return SupSub(self, arg[0], arg[1])
        if not isinstance(arg, tuple):
            arg = arg,
        return Sub(self, *arg)
    def d(self, *args): return PDiff(self, *args)
    @property
    def inv(self): return Inv(self)

    @property
    def slice(self): return Slicer(self)
    @classproperty
    def doit_rule(cls): return {cls: cls._doit}
    @classmethod
    def _doit(cls, *args): return cls(*args).doit()

    _op_priority = 100.0
    def __rshift__(self, other):
        return Rep(other)(self)
    def __lshift__(self, other):
        if self.verbose:
            tprint(f'Doing replacement: {other}')
        return Expand(self, Rep(other)(self))
    def __mul__(self, other): return Mul(self, other)
    def __rmul__(self, other): return Mul(other, self)
    def __truediv__(self, other): return Frac(self, other)
    def __rtruediv__(self, other): return Frac(other, self)
    def __add__(self, other): return Add(self, other)
    def __radd__(self, other): return Add(other, self)
    def __sub__(self, other): return Add(self, Negative(other))
    def __rsub__(self, other): return Add(other, Negative(self))
    def __matmul__(self, other): return Mul(self, other)
    def __pow__(self, other): return mPow(self, other)
    def __neg__(self): return Negative(self)
    def eqs(self, other): return Eq(self, other)
    @property
    def T(self): return Transpose(self)
    @property
    def H(self): return ComplexTranspose(self)
    def underbrace(self, other): return Underbrace(self, other)
    def __eq__(self, other): return Eq(self, other)
    def __gt__(self, other): return Greater(self, other)
    def __lt__(self, other): return Less(self, other)
    def __ne__(self, other): return Neq(self, other)
    __hash__ = sp.Expr.__hash__
    # def __hash__(self): return hash((self.__class__, ) + self.args)

class BasicSymbol(BasicExpr):
    up_precedence = 10000
    @classmethod
    def symbols(cls, x, **kwargs): return sp.symbols(x, cls=cls, **kwargs)
    @property
    def hat(self): return self.func(f'\hat{{{self.name}}}')

class Wild(sp.Wild, BasicSymbol):
    pass

class Symbol(sp.Symbol, BasicSymbol):
    @property
    def _(self): return Wild(self.name)
    pass

class Set(Symbol):
    pass
Reals = Set(r'\mathbb{R}')

class FunctionWrapper(BasicExpr):
    def __call__(self, *args, **kwargs):
        return self.args[0](*args, **kwargs)
    free_symbols = set()

def _sympify(expr):
    if isinstance(expr, str):
        if expr.find('\n') != -1:
            return expr
        if len(expr) and expr[-1] == '_':
            return Wild(expr[:-1])
        return expr
        return Symbol(expr)
    if isinstance(expr, tuple):
        return sp.Tuple(*map(_sympify, expr))
    if isinstance(expr, dict):
        return sp.Dict({
            _sympify(key): _sympify(value) for key, value in expr.items()
        })
    if not isinstance(expr, sp.Basic) and callable(expr):
        return FunctionWrapper(expr)
    if isinstance(expr, type):
        return Protect(expr)
    return sp.sympify(expr)

def down_precedence(arg): return getattr(arg, 'down_precedence', 0)
def up_precedence(arg):
    if isinstance(arg, sp.Atom): return BasicSymbol.up_precedence
    return getattr(arg, 'up_precedence', 0)

# dp = getattr(parent, 'down_precedence', 0)
# up = getattr(arg, 'up_precedence', 0)

class Expr(BasicExpr):
    @property
    def last(self):
        if not isinstance(self.args[-1], BasicExpr):
            return Protect(self.args[-1])
        return self.args[-1]
    @cached_property
    def free_symbols(self):
        return set().union(*[
            a.free_symbols for a in self.args
            # if (hasattr(a, 'free_symbols') and not isinstance(a, type))
            if (hasattr(a, 'free_symbols') and iterable(a.free_symbols))
        ])

    # @property
    # def final(self):
    #     pass
    @property
    def ready(self):
        return not any([
            isinstance(arg, (sp.Dummy, sp.Wild))
            for arg in self.free_symbols
            # if hasattr(arg, 'free_symbols')
        ])

    @property
    def size(self):
        # import sys
        from pympler import asizeof
        return asizeof.asizeof(self)
        return sys.getsizeof(self)
    hsize = Cache.hsize
    @property
    def clsrepr(self):
        return self.__class__.__name__ + '(' + ','.join([
            arg.__class__.__name__ for arg in self.args
        ]) + ')'

    def __new__(cls, *args, **kwargs):
        rv = BasicExpr.__new__(cls, *map(_sympify, args))
        prv = rv.__post__()
        if rv.verbose:
            tprint(f'Finished post {rv.clsrepr}')
        return prv
    def __post__(rv):
        if rv.verbose:
            tprint(f'new {rv.clsrepr} ({rv.hsize})')
        # return rv
        if not rv.ready:
            return rv
        prv = rv._post_new()
        if prv is not None:
            return prv
        for i, arg in enumerate(rv.args):
            if not hasattr(arg, '_post_new_up'):
                continue
            prv = arg._post_new_up(rv, i)
            if prv is not None:
                return prv
        return rv

    def _post_new(self):
        return None
    def _post_new_up(self, expr, i):
        return None
    def par(self, arg): return Par(arg)
    def _latex(self, printer=None):
        # rp = getattr(self, 'precedence', 0)
        def Latex(parent, arg):
            if isinstance(arg, (tuple, sp.Tuple)):
                par = parent if isinstance(arg, tuple) else arg
                return Map(functools.partial(Latex, par), arg)
            # dp = getattr(parent, 'down_precedence', 0)
            # up = getattr(arg, 'up_precedence', 0)
            if up_precedence(arg) < down_precedence(parent):
                arg = self.par(arg)
            return printer.doprint(arg)
        largs = Latex(self, self.args)
        # try:
        if isinstance(self._mlatex, str):
            return self._mlatex.format(self=self, args=largs)
            # Map(
            #     lambda arg: printer.doprint(arg), self.args
            # ))
        return self._mlatex(largs)
        # except:

    @property
    def _mlatex(self):
    # def _mlatex(self, args):
        rv = rf'\text{{{{{self.__class__.__name__.lower()}}}}}({{args}})'
        return getattr(
            self,
            f'_mlatex{len(self.args)}',
            rv
        )
        # if _mlatex is not
        return rf'\text{{{self.__class__.__name__.lower()}}}({", ".join(args)})'
    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(', '.join(
            map(repr, self.args)
        ))
        try:
            return self.__class__.__name__ + '({})'.format(', '.join(
                map(repr, self.args)
            ))
        except:
            return self.__class__.__name__ + '(...)'
    # __str__ = __repr__
    def __str__(self): return self.l
    # def __str__(self): return self.__repr__()
    # def _sympstr(self, *args, **kwargs):
        # print(args)
        # return 'hi'
        # return str(self)
    # _sympystr = __str__
    # __str__ = __repr__
    def split_affected(self, args):
        a, na = [], []
        for arg in args:
            if self.affects(arg):
                a.append(arg)
            else:
                na.append(arg)
        return a, na

    @property
    def expr(self): return self.args[0]

    def replace_expr(self, other): return self.func(other, *self.args[1:])
    def wrap_expr(self, f): return self.replace_expr(f(self.expr))

    def replace_at(self, i, arg):
        if not isinstance(arg, tuple):
            arg = arg,
        return self.func(*self.args[:i], *arg, *self.args[i+1:])

def last(expr): return expr.last

class Protect(Expr):
    # _mlatex = r'P\left({args[0]}\right)'
    down_precedence = -BasicSymbol.up_precedence
    _mlatex = r'{args[0]}'
    def doit(self):
        return self.args[0]

class Par(Protect):
    down_precedence = Protect.down_precedence
    _mlatex = r'\left({args}\right)'

class Bra(Protect):
    down_precedence = Par.down_precedence
    _mlatex = r'\left[{args[0]}\right]'

class MyTuple(Expr):
    up_precedence = BasicSymbol.up_precedence
    down_precedence = Par.down_precedence
    _mlatex = r'\left({args}\right)'
Tuple = MyTuple


class MyExpand(Expr):
    def _repr_latex_(self): return Eq(*self.args).bl
    def _post_new_up(self, expr, i):
        return expr.replace_at(i, self.args)
        return expr.func(*expr.args[:i], *self.args, *expr.args[i+1:])

    def __rshift__(self, other):
        return (self << other).replace_at(-2, tuple())
    #     return Rep(other)(self)
    # def __lshift__(self, other):
    #     if self.verbose:
    #         tprint(f'Doing replacement: {other}')
    #     return Expand(self, Rep(other)(self))
Expand = MyExpand

class SpExpand(Expr):
    def _post_new(self):
        if isinstance(self.expr, Matrix):
            return self.expr.func(Map(self.func, self.expr))
        return sp.expand(self.args[0])

class Distribute(Expr):
    def _post_new(self):
        if isinstance(self.expr, Matrix):
            return self.expr.func(Map(self.func, self.expr))
        if isinstance(self.expr, Add):
            return self.expr.func(*Map(self.func, self.expr.args))
        if isinstance(self.expr, Mul):
            for i, arg in enumerate(self.expr.args):
                if isinstance(arg, Add):
                    return arg.func(*[
                        self.func(self.expr.replace_at(i, _arg))
                        for _arg in arg.args
                    ])
        return self.expr

class Apply(Expr):
    def _post_new(self):
        return self.args[0](self.args[1])


class Operator(Expr):
    def _mlatex(self, args):
        if len(args) == 1:
            return f'{self.op}{{{args[0]}}}'
        return f' {self.op} '.join(map('{{{}}}'.format, args))

class Min(Operator):
    _mlatex = Expr._mlatex
    # _mlatex = r'\min\left({args}\right)'
class Max(Operator):
    _mlatex = Expr._mlatex
    # _mlatex = r'\max\left({args}\right)'

class Union(Operator):
    down_precedence = Par.down_precedence
    def _mlatex(self, args):
        return rf'\cup_{{{args[1]}={args[2]}}}^{{{args[3]}}} {args[0]}'

class Intersection(Operator):
    down_precedence = Par.down_precedence
    op = r'\cap'

class SetMinus(Operator):
    down_precedence = Par.down_precedence
    op = r'\setminus'

    # _mlatex2 = r'{args[0]} \cap {args[1]}'
    # def _mlatex(self, args):
    #     return rf'\cap_{{{args[1]}={args[2]}}}^{{{args[3]}}} {args[0]}'

class Closure(Operator):
    down_precedence = Par.down_precedence
    _mlatex = r'\overline{{{args}}}'

class COInterval(Expr):
    _mlatex = r'\left[{args}\right)'

class UnaryOperator(Operator):
    pass

class TextOperator(Operator):
    @property
    def _mlatex(self):
        return r'\mathrm{{' + self.tag + r'}}\left({args}\right)'


class Rank(TextOperator):
    pass

class Det(TextOperator):
    pass
    # _mlatex = r'\mathrm{{det}}{args}'

class Diag(TextOperator):
    pass

class Abs(UnaryOperator):
    down_precedence = Par.down_precedence
    # _mlatex = r'\left|{args}\right|'
    _mlatex = r'\left| {args} \right|'

class Norm(UnaryOperator):
    down_precedence = Par.down_precedence
    _mlatex = r'\left\lVert {args[0]} \right\rVert_{{{args[1]}}}'


class Sign(UnaryOperator):
    _mlatex = r'\sigma\left({args}\right)'

class BinaryOperator(Operator):
    @property
    def lhs(self): return self.args[0]
    @property
    def rhs(self): return self.args[1]
    # _mlatex = r'{args[0]} {self.op} {args[1]}'
    @classmethod
    def _func(cls, *args): return functools.reduce(cls._op, map(doit, args))
    def doit(self): return self._func(*self.args)

    @property
    def largs(self):
        from sympy.printing.latex import latex
        return Map(latex, self.args)
    @property
    def bl(self): return self.align(self.abl)
    @property
    def abl(self):
        return """
        {}& {} {}
""".format(
            self.largs[0],
            self.op,
            f'\\\\\n& {self.op} '.join(self.largs[1:])
        )
    @property
    def ibl(self): return self.align(f' {self.op} '.join(self.largs))
#         return r"""
#
# \begin{{align}}\end{{align}}
# """.format(
#             self.largs[0],
#             self.op,
#             f'\\\\\n& {self.op} '.join(self.largs[1:])
#         )
    @property
    def al(self):
        return f' &{self.op} '.join([arg for arg in self.largs])

class MyAdd(BinaryOperator):
    up_precedence = -1
    op = r'+'
    _op = operator.add
    def _mlatex(self, largs):
        rv = largs[0]
        for arg, larg in zip(self.args[1:], largs[1:]):
            if not (isinstance(arg, (Negative, PlusMinus)) and len(arg.args) == 1):
                rv += f' {self.op} '
            rv += larg
        return rv
    def _post_new(self):
        if sp.S.Zero in self.args:
            return self.func(*[arg for arg in self.args if arg is not sp.S.Zero])
        if len(self.args) == 1:
            return self.args[0]
        if len(self.args) == 0:
            return sp.S.Zero
        fns = Map(MyMul.fn, self.args)
        if len(set([fn[1] for fn in fns])) != len(self.args):
            rv = dict()
            for f, n in fns:
                rv[n] = rv.get(n, sp.S.Zero) + f
            return self.func(*[f * n for (n, f) in rv.items()])
        if all([isinstance(arg, Mul) for arg in self.args]):
            if len(set(arg.expr for arg in self.args)) == 1:
                return self.expr.expr * self.func(
                    *[arg.replace_at(0, tuple()) for arg in self.args]
                )

        # for i in range(len(self.args)):
        #     for j in range(i+1, len(self.args)):
        #         if fns[i][1] is fns[j][1]:
        #             return self.func(
        #                 *self.args[:i],
        #                 *self.args[i+1:j],
        #                 *self.args[j+1:]
        #             )

    def _post_new_up(self, expr, i):
        if isinstance(expr, MyAdd):
            return expr.replace_at(i, self.args)
Add = MyAdd

class PlusMinus(BinaryOperator):
    op = r'\pm'

class MyMinus(BinaryOperator):
    op = r'-'
Minus = MyMinus

class MyMul(BinaryOperator):
    precedence = 50
    op = r' '
    _op = operator.mul
    def _post_new(self):
        if sp.S.Zero in self.args:
            return sp.S.Zero
        if sp.S.One in self.args:
            return self.func(*[arg for arg in self.args if arg is not sp.S.One])
        if len(self.args) == 1:
            return self.args[0]
        if len(self.args) == 0:
            return 1
        fs = [arg for arg in self.args if isinstance(arg, sp.Number)]
        if len(fs) > 1:
            nfs = [arg for arg in self.args if not isinstance(arg, sp.Number)]
            return functools.reduce(
                operator.mul,
                fs + nfs,
                sp.S.One
            )

    def _post_new_up(self, expr, i):
        if isinstance(expr, MyMul):
            return expr.replace_at(i, self.args)

    @classmethod
    def fn(cls, expr):
        if isinstance(expr, cls) and isinstance(expr.expr, sp.Number):
            return expr.expr, expr.replace_at(0, tuple())
        if isinstance(expr, sp.Number):
            return expr, sp.S.One
        if isinstance(expr, Negative):
            f, n = cls.fn(expr.expr)
            return -f, n
        return sp.S.One, expr


    # def doit(self):
    #     args = sorted([arg.doit() for arg in self.args])
    #     return self._func(*(args))

    # @classmethod
    # def _func(cls, *args): return functools.reduce(op.mul, args)
Mul = MyMul

class Negative(UnaryOperator):
    _mlatex = r'-{args[0]}'
    def _post_new(self):
        if self.expr is sp.S.Zero:
            return sp.S.Zero
        if isinstance(self.expr, Mul):
            return sp.S.NegativeOne * self.expr
        # return Mul(sp.S.NegativeOne, self.expr)
        if isinstance(self.expr, Negative):
            return self.expr.expr
    def _post_new_up(self, expr, i):
        if isinstance(expr, (Mul, Frac, PDiff)):
            return self.func(expr.replace_at(i, self.expr))



class After(BinaryOperator):
    op = r'\circ'
    def __call__(arg):
        return self.lhs(self.rhs(arg))

class mPow(BinaryOperator):
    op = r'^'
    _op = operator.pow

class Inv(UnaryOperator):
    _mlatex = r'{{{args[0]}}}^{{-1}}'

class Frac(BinaryOperator):
    _mlatex = r'\frac{{{args[0]}}}{{{args[1]}}}'
    _op = operator.truediv
    def _post_new(self):
        if self.args[1] is sp.S.One:
            return self.args[0]
        if self.args[0] is sp.S.Zero:
            return self.args[0]
        if self.args[0] is self.args[1]:
            return sp.S.One
        if isinstance(self.expr, Frac):
            return Frac(self.expr.expr, self.expr.args[1] * self.args[1])
        if isinstance(self.args[0], Mul):
            if self.args[1] in self.args[0].args:
                return self.args[0].replace_at(
                    self.args[0].args.index(self.args[1]),
                    tuple()
                )
            if isinstance(self.expr.expr, (Frac, sp.Number)):
                return self.expr.replace_expr(self.expr.expr / self.args[1])
        if isinstance(self.args[1], Mul):
            if self.args[0] in self.args[1].args:
                return self.func(
                    1,
                    self.args[1].replace_at(
                        self.args[1].args.index(self.args[0]),
                        tuple()
                    )
                )
        if isinstance(self.args[0], Mul) and isinstance(self.args[1], Mul):
            rargs = self.args[1].args
            for i, arg in enumerate(self.args[0].args):
                if arg not in rargs:
                    continue
                j = rargs.index(arg)
                return self.func(
                    self.args[0].replace_at(i, tuple()),
                    self.args[1].replace_at(j, tuple())
                )


    def _post_new_up(self, expr, i):
        if isinstance(expr, Mul) and i != 0:# and expr.is_scalar:
            return self.func(
                expr.replace_at(i, self.args[0]),
                self.args[1]
            )

    @property
    def constant(self):
        return all(map(is_constant, self.args))


class Contains(BinaryOperator):
    op = r'\in'

class Times(BinaryOperator):
    op = r'\times'

class Maps(Operator):
    _mlatex = r'{args[0]}: {args[1]} \rightarrow {args[2]}'

class BinaryRelation(BinaryOperator):
    down_precedence = Par.down_precedence

class Subset(BinaryRelation):
    op = r'\subset'

class SubsetEq(BinaryRelation):
    op = r'\subseteq'

class Expands:
    pass

class Eq(BinaryRelation):
    op = r'='
    @property
    def ready(self): return any(isinstance(arg, (Expand, Expands)) for arg in self.args)
    def __bool__(self):
        if self.verbose:
            print(f'bool({repr(self)})')
        return sp.Expr.__eq__(self.lhs, self.rhs)
    @property
    def trimmed(self):
        for i, arg in enumerate(self.args[:-1]):
            if self.args[i+1] is arg:
                return self.replace_at(i, tuple()).trimmed
        return self

class Neq(BinaryRelation):
    op = r'\neq'
    @property
    def ready(self): return any(isinstance(arg, (Expand, Expands)) for arg in self.args)
    def __bool__(self):
        if self.verbose:
            print(f'bool({repr(self)})')
        return sp.Expr.__ne__(self.lhs, self.rhs)

class Geq(BinaryRelation):
    op = r'\geq'

class Leq(BinaryRelation):
    op = r'\leq'

class Less(BinaryRelation):
    op = r'<'

class Greater(BinaryRelation):
    op = r'>'

class And(BinaryOperator):
    op = r'\text{ and }'

class Or(BinaryOperator):
    op = r'\text{ or }'

class Comma(BinaryOperator):
    down_precedence = Par.down_precedence
    op = r','
class Point(BinaryOperator):
    op = r'.'
class Implies(BinaryOperator):
    op = r'\implies'

class Cases(Expr):
    def _latex(self, printer=None):
        latex = printer.doprint
        coda = []
        if len(self.args) > 1:
            coda = [rf'{latex(self.args[1])} & \text{{ else }}']
        return r'\begin{cases}' + ',\\\\\n'.join([
            rf'{latex(value)} & \text{{ if }} {latex(condition)}'
            for condition, value in self.expr.items()
        ] + coda) + r'\end{cases}'

class Call(Expr):
    down_precedence = Par.down_precedence
    _mlatex = r'{args[0]}\left({args[1:]}\right)'

class Indexer(Expr):
    down_precedence = BasicSymbol.up_precedence
    precedence = 100
    @property
    def is_Symbol(self): return self.args[0].is_Symbol
    @property
    def constant(self): return is_constant(self.expr)
    def par(self, arg): return Bra(arg)
    # def _latex(self, printer=None):
    #     if self.is_Symbol or isinstance(self.expr, Bra):
    #         return Expr._latex(self, printer)
    #     return self.wrap_expr(Bra)._latex(printer)

    # @property
    # def _mlatex(self):
    #     if self.is_Symbol or isinstance(self.expr, Bra):
    #         return  r'{{{args[0]}}}^{{{args[1]}}}_{{{args[2]}}}'
    #     return self.wrap_expr(Bra)._mlatex


class Sub(Indexer):
    _mlatex = r'{{{args[0]}}}_{{{args[1:]}}}'
    def _post_new(self):
        if isinstance(self.expr, Sup):
            return SupSub(self.expr.expr, self.expr.args[1], self.args[1])
    # @property
    # def _mlatex(self):
    #     if self.args[0].is_Symbol or isinstance(self.args[0], Bracket):
    #         return  r'{{{args[0]}}}_{{{args[1:]}}}'
    #     return  r'\left[{args[0]}\right]_{{{args[1:]}}}'
    # def _post_new(self):
    #     if isinstance(self.args[0], Sub):
    #         return Sub(self.args[0].args[0], self.args[0].args[1] + self.args[1])
    #     return self
    # _mlatex = r'{{{args[0]}}}_{{{args[1:]}}}'

class Sup(Indexer):
    _mlatex = r'{{{args[0]}}}^{{{args[1:]}}}'
    # def __getitem__(self, )
    # @property
    # def _mlatex(self):
    #     if self.args[0].is_Symbol:
    #         return  r'{{{args[0]}}}^{{{args[1:]}}}'
    #     return  r'\left[{args[0]}\right]^{{{args[1:]}}}'

class SupSub(Indexer):
    _mlatex = r'{{{args[0]}}}^{{{args[1]}}}_{{{args[2]}}}'
    # @property
    # def _mlatex(self):
    #     if self.is_Symbol or isinstance(self.expr, Bra):
    #         return  r'{{{args[0]}}}^{{{args[1]}}}_{{{args[2]}}}'
    #     return self.wrap_expr(Bra)._mlatex


class Permuter(Expr):
    down_precedence = BasicSymbol.up_precedence
    def par(self, arg): return Bra(arg)
    @property
    def indices(self): return self.args[1:]
    @property
    def flat_indices(self):
        rv = []
        for index in self.indices:
            if isinstance(index, (sp.Tuple, Tuple)):
                rv += list(index.args)
            else:
                rv.append(index)
        return rv

    def affects(self, other):
        return isinstance(other, Protect) or any([
            other.find(index) for index in self.flat_indices
        ])

    def _post_new(self):
        if not self.affects(self.expr):
            # print(list(self.free_symbols), [
            #     (index, self.expr.find(index)) for index in self.indices
            # ])
            return self.expr
        if isinstance(self.expr, sp.Add):
            return sp.Add(*[
                self.replace_expr(arg)
                for arg in self.expr.args
            ])
        if isinstance(self.expr, sp.Mul):
            not_affected = [
                arg for arg in self.expr.args
                if not self.affects(arg)
            ]
            if not_affected:
                affected = [
                    arg for arg in self.expr.args
                    if self.affects(arg)
                ]
                return sp.Mul(*not_affected) * \
                    self.replace_expr(sp.Mul(*affected))


class Swap(Permuter):
    _mlatex = r'{{{args[0]}}}_{{{args[1]} \leftrightarrow {args[2]}}}'
    def doit(self):
        i, j = self.indices
        # print(i.__class__)
        i = list(i.args) if isinstance(i, Tuple) else [i]
        j = list(j.args) if isinstance(j, Tuple) else [j]
        return Rep(dict(zip(i+j,j+i)))(self.expr)

        return Rep({i: j, j: i})(self.expr)#.doit()

class Symm(Permuter):
    _mlatex = r'{{{args[0]}}}_{{\left({args[1:]}\right)}}'
    def doit(self):
        return Frac(1,2)*(self.expr + Swap(self.expr, *self.indices).doit()).doit()

class ASymm(Permuter):
    _mlatex = r'{{{args[0]}}}_{{\left[{args[1:]}\right]}}'
    # @classproperty
    # def rules(cls):
    #     a_, b_, c_ = Wild.symbols('a b c')
    #     rules = {
    #         ASymm(a_, b_, c_): a_ - Swap(a_, b_, c_)
    #     }

    def doit(self):
        return Frac(1,2)*(self.expr - Swap(self.expr, *self.indices).doit()).doit()
    # @property
    # def expr(self): return self.args[0]
    # @property
    # def indices(self): return self.args[1:]
    # def _latex(self, printer=None):
    #     return f'\\left({printer.doprint(self.expr)}\\right)_' + '{{{}}}'.format(', '.join([
    #         printer.doprint(index) for index in self.indices
    #     ]))

    # def doit(self):
    #     if not self.indices: return self.expr
    #     l, r = lr = self.indices[0]
    #     if not isinstance(l, tuple):
    #         l = l,
    #         r = r,
    #     assert(len(l) == len(r))
    #     pairs = list(zip(l+r, r+l))
    #     swapped = self.expr.subs(pairs, simultaneous=True)
    #     mul = 1 if isinstance(lr, tuple) else -1
    #     return Symm((self.expr + mul*swapped)/2, *self.indices[1:]).doit()

class Cycler(Permuter):
    _mlatex = r'{{{args[0]}}}_{{\circlearrowright {args[1:]}}}'
    # @property
    # def _mlatex(self):
    #     if self.args[0].is_Symbol or isinstance(self.args[0], (Protect, Bracket)):
    #         return  r'{{{args[0]}}}_{{\circlearrowright {args[1:]}}}'
    #     return  r'\left[{args[0]}\right]_{{\circlearrowright {args[1:]}}}'
    def doit(self):
        i, j, k = self.indices
        return (
            self.expr +
            Rep({i: j, j: k, k: i})(self.expr) +
            Rep({i: k, j: i, k: j})(self.expr)
        ).doit()
    # _mlatex = r'\left({args[0]}\right)_{{\circlearrowright {args[1:]}}}'

    # def doit(self):
    #     ids = self.args[1]
    #     ids2 = list(ids) + list(ids)
    #     perms = [
    #         ids2[i:i+len(ids)] for i in range(len(ids))
    #     ]
    #     return sum([
    #         self.expr.subs(zip(ids, permi), simultaneous=True)
    #         for permi in perms
    #     ])

# class Functional(Expr):
#     def _latex(self, printer=None):
#         return printer.doprint(self.args[0])


        # return self.args[0]

def Jacobi(P, F, G, H):
    return Cycler(P(F, P(G, H)), (F, G, H))

# class Jacobi(Expr):
#     def _latex(self, printer=None):
#         return printer.doprint(self.doit())
#     def doit(self):
#         P, F, G, H = self.args
#         return Cycle(P(P(F, G), H), (F, G, H))#.doit()
#         return '{' + printer.doprint(self.args[0]) + '}_{' + ', '.join(map(printer.doprint, self.args[1:])) + '}'

class Bracket(Expr):
    down_precedence = Par.down_precedence
    up_precedence = BasicSymbol.up_precedence
    @property
    def lhs(self): return self.args[0]
    @property
    def rhs(self): return self.args[1]

class Poisson(Bracket):
    _mlatex = r'\left\{{{args}\right\}}'
    @classmethod
    def rhs(cls, G, H, mu, i, j, k):
        return Dot(
            FDiff(G, u[i]), (
                b[mu,i,j,k]*u[k,mu] + g[mu,i,j] * PDiff(blank, x[mu])
            ) * FDiff(H, u[j])
        )
    @classmethod
    def rhs_exp(cls, G, H, mu, i, j, k):
        return Dot(
            FDiff(G, u[i]), (
                b[mu, i, j, k] * H.d(u[j]) +
                g[mu, i, j] * H.d(u[j], u[k])
            ) * u[k, mu]
        )
    @classmethod
    def rhs_eexp(cls, G, H, mu, i, j, k):
        return Dot(
            G.d(u[i]) * (
                b[mu, i, j, k] * H.d(u[j]) +
                g[mu, i, j] * H.d(u[j], u[k])
            ), u[k, mu]
        )
    @classmethod
    def d_exp(cls, G, H, mu, i, j, k, l, sim=False):
        if sim:
            return ASymm(PDiff(
                Poisson.rhs_eexp(G, H, mu, i, j, k).lhs,
                u[l]
            ), k, l)*u[k, mu],
        return PDiff(
            Poisson.rhs_eexp(G, H, mu, i, j, k).lhs,
            u[l]
        )*u[k, mu] - PDiff(
            Poisson.rhs_eexp(G, H, mu, i, j, l).lhs,
            x[mu]
        )

class Lie(Bracket):
    _mlatex = r'\left[{args}\right]'


class LinearExpr(Expr):
    @classproperty
    def summation_rule(cls):
        a, b, c = Wild.symbols('a b c')
        return {cls(a+b, c): cls(a, c) + cls(b, c)}

def is_constant(expr):
    if hasattr(expr, 'constant'):
        return expr.constant
    return isinstance(expr, sp.Number)

class Derivation(LinearExpr):
    # @property
    down_precedence = Par.down_precedence
    def _mlatex(self, args):
        num = self.sym if len(self.args) == 2 else self.sym + r'^{{{}}}'.format(len(self.args)-1)
        den = ' '.join([rf'{self.sym} {arg}' for arg in args[1:]])
        if self.args[0].is_Symbol:
            return rf'\frac{{{num} {args[0]}}}{{{den}}}'
        if isinstance(self.args[0], Bracket):
            return rf'\frac{{{num}}}{{{den}}}{args[0]}'
        return rf'\frac{{{num}}}{{{den}}}\left({args[0]}\right)'
    # def _mlatex(self):
    #     if self.args[0].is_Symbol:
    #         return r'\frac{{{self.sym} {args[0]}}}{{{self.sym} {args[1]}}}'
    #     if isinstance(self.args[0], Bracket):
    #         return r'\frac{{{self.sym}}}{{{self.sym} {args[1]}}}{args[0]}'
    #     return r'\frac{{{self.sym}}}{{{self.sym} {args[1]}}}\left({args[0]}\right)'
    # is_commutative =
    @classproperty
    def product_rule(cls):
        a, b, c = Wild.symbols('a b c')
        return {
            cls(a * b, c): cls(a, c) * b + a * cls(b, c)
        }

    # @classproperty
    # def rules(cls):
    #     return dict((cls.product_rule, cls.summation_rule))
    def affects(self, other):
        return not is_constant(other)

    def _post_new(self):
        if not self.affects(self.args[0]):
            return 0
        if isinstance(self.args[0], (Mul, sp.Mul)):
            mul = self.args[0].func
            a, na = self.split_affected(self.args[0].args)
            if na:
                return mul(*na, self.replace_expr(mul(*a)))
        if isinstance(self.expr, sp.Add):
            return sp.Add(*map(self.replace_expr, self.expr.args))
        # if isinstance(self.args[0], sp.Number):
        #     return 0
        if isinstance(self.args[0], self.__class__):
            return self.func(
                self.args[0].args[0],
                *self.args[0].args[1:],
                *self.args[1:]
            )

    def _post_new_up(self, expr, i):
        if isinstance(expr, Sub) and i == 0:
            return self.func(
                self.args[0],
                self.args[1][expr.args[1]]
            )
        return LinearExpr._post_new_up(self, expr, i)


class PDiff(Derivation):
    # @property
    sym = r'\partial'
    # def _mlatex(self, args):
    #     num = r'\partial' if len(self.args) == 2 else r'\partial^{{{}}}'.format(len(self.args)-1)
    #     den = ' '.join([rf'\partial {arg}' for arg in args[1:]])
    #     if self.args[0].is_Symbol:
    #         return rf'\frac{{{num} {args[0]}}}{{{den}}}'
    #     return rf'\frac{{{num}}}{{{den}}}\left({args[0]}\right)'


class FDiff(Derivation):
    sym = r'\delta'
    # @property
    # def _mlatex(self):
    #     if self.args[0].is_Symbol:
    #         return r'\frac{{\delta {args[0]}}}{{\delta {args[1]}}}'
    #     return r'\frac{{\delta}}{{\delta {args[1]}}}\left({args[0]}\right)'

class CoDerivative(Expr):
    def _mlatex(self, args):
        return ' '.join(map(r'\nabla_{{{}}}'.format, args[1:])) + args[0]
    # _mlatex = r'\nabla_{{{}}}{args[0]}'

class Hessian(LinearExpr):
    _mlatex = r'\nabla^2 {args[0]}'


class Dot(Bracket):
    precedence = 1000
    down_precedence = Par.down_precedence
    up_precedence = BasicSymbol.up_precedence
    _mlatex = r'\left<{args}\right>'
    @property
    def density(self): return sp.Mul(*self.args)

def LiePoisson(F, G):
    return Dot('u', Lie(
        FDiff(F, 'u'),
        FDiff(G, 'u')
    ))

# class LiePoisson(Expr):
#     def _latex(self, printer=None):
#         return r'\left<u, {}\right>'.format(printer.doprint(Lie(
#             FDiff(self.args[0], 'u'),
#             FDiff(self.args[1], 'u')
#         )))
#     def doit(self): return Dot('u', Lie(
#         FDiff(self.args[0], 'u'),
#         FDiff(self.args[1], 'u')
#     ))
# class LiePoisson(Expr):
#     pass
    # @property
    # def L(self): return self.args[0]
    # @property
    # def u(self): return self.args[1]
    # def __call__(self, F, G):
    #     return sp.diff(F(self.u), self.u) * sp.diff(G(self.u), self.u)
# def LiePoisson(f, g):
#     print(args)
#     return u

# def doit(expr): return expr.doit()

class Rep(Expr):
    @property
    def ready(self): return True
    def _post_new(self):
        assert(len(self.args) > 0)
        if len(self.args) == 1:
            if isinstance(self.args[0], Rep):
                return self.args[0]
            if isinstance(self.expr, (sp.Dict, dict)) and len(self.expr) == 1:
                [key, value], = self.expr.items()
                # if self.verbose:
                #     tprint(f'Replaced {key}: {value}')
                return self.func(key, value)
            if callable(self.args[0]) or isinstance(self.args[0], sp.Dict):
                return None
            return self.func(*self.args[0])
        # if len(self.args) == 2:
        #     if not isinstance(self.args[0], sp.Basic):
        #         rep, args = self.args
        #         assert(callable(rep))
        #         return Rep(rep(*args))
        if len(self.args) == 3:
            pattern, rep, args = self.args
            assert(callable(rep))
            # print(pattern, rep(*args))
            return self.func(pattern, rep(*args))
        # if callable(self.args[1]) and len(self.args) > 2:
        #     return Rep(self.args[0], self.args[1](*self.args[2]))
        return None

    _mlatex = r'{args[0]} \rightarrow {args[1]}'
    def _latex(self, printer=None):
        if len(self.args) == 1:
            if callable(self.args[0]):
                return printer.doprint(self.args[0])
                # return rf'\text{{{self.args[0]}}}'
            if isinstance(self.args[0], sp.Dict):
                return ', '.join([
                    Rep(pattern, rep)._latex(printer)
                    for pattern, rep in self.args[0].items()
                ])
        return Expr._latex(self, printer)

    def _unwrap(self, expr):
        if isinstance(expr, (FunctionWrapper, Protect)):
            return expr.args[0]
        return expr

    # https://github.com/sympy/sympy/blob/969cf54cf19e830306a84c00864b03f7b66bf8e0/sympy/core/basic.py
    def prepare(self, query, value):
        if isinstance(query, type):
            _query = lambda expr: isinstance(expr, query)
            _value = lambda expr: value(*expr.args)
        elif isinstance(query, sp.Basic):
            _query = lambda expr: expr.match(query)

            if isinstance(value, sp.Basic):
                _value = lambda expr, result: value.subs(result)
            elif callable(value):
                # match dictionary keys get the trailing underscore stripped
                # from them and are then passed as keywords to the callable;
                # if ``exact`` is True, only accept match if there are no null
                # values amongst those matched.
                _value = lambda expr, result: value(**
                    {str(k)[:-1]: v for k, v in result.items()})
            else:
                raise TypeError(
                    "given an expression, replace() expects "
                    "another expression or a callable")
        elif callable(query):
            _query = query

            if callable(value):
                _value = lambda expr, result: value(expr)
            else:
                raise TypeError(
                    "given a callable, replace() expects "
                    "another callable")
        return _query, _value
# [12:36:40]: Replacing 6580123461494129077 << -7116546432610824495
    @cache.memoize()
    def _call(self, expr):#, **kwargs):
        if self.verbose:
            import pickle
            s, e = [pickle.dumps(arg, protocol=pickle.HIGHEST_PROTOCOL) for arg in (self, expr)]
            tprint(f'Replacing {hash(e)} << {hash(s)}')
        if len(self.args) == 1:
            if callable(self.expr):
                return self.expr(expr)
            assert(isinstance(self.expr, sp.Dict))
            # _queries, _values = zip(*map(
            #     self.prepare, *zip(*self.expr.items())
            # ))
            # def _query(_expr):
            #     return any([query(_expr) for query in _queries])
            # def _value(_expr):
            #     for q, v in zip(_queries, _values):
            #         result = q(_expr)
            #         if not result:
            #             continue
            #         rv = v(_expr, result)
            #         if rv is not _expr:
            #             return rv
            #     return _expr
            # return expr.replace(_query, _value, map=False)
            def rep_first(_expr):
                for items in self.args[0].items():
                    _query, _value = map(self._unwrap, items)
                    rv = _expr.replace(_query, _value, map=False)
                    if rv is not _expr:
                        return rv
                return _expr

            def pattern(_expr):
                rv = rep_first(_expr)
                return rv is not _expr
            def rep(_expr):
                return rep_first(_expr)
            return expr.replace(pattern, rep, map=False)
        if len(self.args) == 2:
            pattern, rep = map(self._unwrap, self.args)
            return expr.replace(pattern, rep, map=False)
        raise ValueError(Call(self, expr))

    def __call__(self, expr):#, **kwargs):
        if isinstance(expr, Expand):
            return self(expr.args[-1])#, **kwargs)
        return self._call(expr)#, **kwargs)#[0]

class RRep(Rep):
    def _latex(self, printer):
        return r'\circlearrowright\left(' + Rep._latex(self, printer) + r'\right)'
    def _call(self, expr, **kwargs):
        while True:
            rv = Rep._call(self, expr, **kwargs)
            if rv is expr:
                return rv
            expr = rv

class REPLACE(Expr):
    # def _latex(self, printer=None):
    #     return sp.latex(dict(a='b'))
    #     return r"""
    #     a & b
    #     """
    @property
    def origin(self): return self.args[0]
    @property
    def end(self): return self.steps[-1]

    @property
    def _sm(self):
        steps = [self.origin]
        maps = []
        for rep in self.reps:
            # step, mapping = rep._call(steps[-1])
            # steps.append(step)
            # maps.append(mapping)
            step = rep._call(steps[-1])
            steps.append(step)
            maps.append(dict())
            # for i in range(5):
            #     # print(step, steps[-1], step == steps[-1], mapping)
            #     # break
            #     if step == steps[-1]:
            #         break
            #     steps.append(step)
            #     maps.append(mapping)
        return steps, maps

    @property
    def steps(self): return self._sm[0]
    @property
    def maps(self): return self._sm[1]
    @property
    def reps(self): return [Rep(arg) for arg in self.args[1:]]

    def _repr_latex_(self):
        from sympy.printing.latex import latex
        lsteps = Map(latex, self.steps)
        return self.align("""
        &{}{}
        """.format(
            lsteps[0],
            '\n'.join([
                r'&&\left| {rep} \right.\\=&{now}'.format(
                    rep=latex(rep),
                    now=now
                )
                for rep, now in zip(self.reps, lsteps[1:])
            ])
        ))
        # return r"""$
        # \begin{{align}}
        # &{}{}
        # \end{{align}}
        # $""".format(
        #     lsteps[0],
        #     '\n'.join([
        #         r'&&\left| {rep} \right.\\=&{now}'.format(
        #             rep=latex(rep),
        #             now=now
        #         )
        #         for rep, now in zip(self.reps, lsteps[1:])
        #     ])
        # )
    # @property
    # def collapsed(self): return CollapsedReplace(*self.args)


class Markdown(Expr):
    def _repr_markdown_(self):
        return '\n'.join([
            arg if isinstance(arg, str) else arg.l for arg in self.args
        ])


class NextFree(Expr):
    @property
    def _mlatex(self):
        if len(self.args) < 3:
            return r'{{{args[0]}}}_{{\_}}'
        return r'{{{args[0]}}}_{{' + str(self.offset) + r'\_}}'
    @property
    def offset(self): return 1 if len(self.args) < 3 else self.args[2]
    def _post_new(self, num=None):
        if num is not None:
            return sp.Tuple(*[
                NextFree(self.args[0], self.args[1], i+1)
                for i in range(num)
            ])
        if (self.has(sp.Wild) or self.has(sp.Dummy)):
            return None
        k, expr = self.args[:2]
        i = Wild('i')
        incs = expr.find(k[i])
        return k[len(incs)+self.offset]

class Tensors(Expr):
    _mlatex2 = r'T^{{{args[0]}}}_{{{args[1]}}}'
    # _mlatex = r'T{args[0]}^{{{args[1]}}}_{{{args[2]}}}'
class Vectors(Tensors):
    _mlatex1 = r'T{args}'
class Covectors(Tensors):
    _mlatex1 = r'T^{{*}}{args}'

class FunctionSpace(Expr):
    @property
    def _mlatex(self):
        if len(self.args) == 1:
            return self.sym + r'\left({args[0]}\right)'
        if len(self.args) == 2:
            return self.sym + r'\left({args[0]}; {args[1]}\right)'
        raise ValueError(self)

class C0inf(FunctionSpace):
    sym = r'C_0^\infty'
    # _mlatex = r'C_0^\infty\left({args[0]}; {args[1]}\right)'

class Cinf(FunctionSpace):
    sym = r'C^\infty'
    # _mlatex = r'C^\infty\left({args[0]}; {args[1]}\right)'

class Functional(Symbol):
    domain = C0inf('\mathcal{U}', sp.Reals)
    pass

class HydroFunctional(Functional):
    @property
    def density(self): return Density(self.name.lower())
    @property
    def diff_rule(self):
        a = Wild('a')
        return (FDiff(self, a), PDiff(self.density, a))
    def d(self, *args):
        return self.density[tuple(arg.args[1] for arg in args)]
        return PDiff(self.density, *args)
    @property
    def m(self):
        u, x = Symbol.symbols('u x')
        return Eq(self(u), sp.Integral(self.density(u(x)), x)).l

# class Function(Symbol):
#     pass

UU = Symbol(r'\mathcal{R}')
U = Symbol(r'\mathcal{U}^n')#'^n')
M = Symbol(r'\mathcal{M}^d')

class Density(Symbol):
    domain = Cinf(UU, sp.Reals)
    pass

class SpatialVariable(Symbol):
    domain = M

# class Christoffel(Expr):
#     _mlatex = r'\Gamma'
#     @property
#     def g(self): return self.args[0]

P = Poisson
F, G, H = sp.symbols('F G H', cls=HydroFunctional)
f, g, h = F.density, G.density, H.density
x, y, z = sp.symbols('x y z', cls=SpatialVariable)
t = Symbol('t')
mu, nu = sp.symbols('mu nu', cls=Symbol)
L = sp.symbols('L', cls=Symbol)
u = Symbol('u')

A, B, C = sp.symbols('A B C', cls=Symbol)
a, b, c, d, e, f, g = sp.symbols('a b c d e f g', cls=Wild)
i, j, k, l, m, n, p, q, r, mu, nu = sp.symbols('i j k l m n p q r mu nu', cls=Symbol)
lam, Id = Symbol.symbols(r'\lambda \mathrm{Id}')
# grad = Symbol(r'\nabla')
eps, grad = Symbol.symbols(r'\varepsilon \nabla')

jac = Jacobi(P, F, G, H)


class Dirac(Expr):
    _mlatex = r'\delta_{{{args}}}'



class DiracHelper(Expr):
    # _mlatex = r'\delta_{{{args}}}'
    constant = True
    def _post_new(self):
        if all([isinstance(arg, (int, sp.Integer)) for arg in self.args]):
            return 1 if len(set(self.args)) == 1 else 0
    def _post_new_up(self, expr, i):
        if not isinstance(expr, Mul):
            return None
        if isinstance(self.args[1], sp.Integer):
            return None

        rv = Expr(*expr.args[:i], *expr.args[i+1:])
        if rv.has(self.args[1]):
            return expr.func(*rv.args).subs({self.args[1]: self.args[0]})
        # if rv.has(self.args[0]):
        #     return rv.subs({self.args[0]: self.args[1]})
        return None
dirac = DiracHelper

class Prime(Expr):
    _mlatex = r"{{{args[0]}}}'"

delta = Symbol('delta')

# rep = REPLACE(
#     Dot(
#         G.d(u[i]) *
#         (Symm(L[i, j, mu], i, j)*H.d(u[j], u[k]) + L[i, j, k, mu] * H.d(u[j])) * PDiff(u[k], x[mu])
#     ),
#     (Dot(PDiff(u[a], b) * c), Dot(-u[a] * PDiff(c, b))),
#     RRep(dict((PDiff.product_rule, PDiff.summation_rule)))
# )
rep = REPLACE(
    # LiePoisson(F, G),
    jac,
    (P(a,b), LiePoisson(a,b)),
    # (FDiff(a, Dot(a, b)), b + Dot(a, FDiff(a,)))
    # (Dot(a, b), lambda k: Dot(a[k], b[k]), NextFree(k, (a, b), num=1)),
    (Dot(a, b), Dot(a[NextFree(k, (a, b))], b[NextFree(k, (a, b))])),
    (
        Lie(a, b)[c],
        a[l[1]] * L[l[1], l[2], c]  * b[l[2]]
        # lambda k1, k2: a[k1] * L[k1, k2, c]  * b[k2],
        # NextFree(l, (a, b, c), num=2)
    ),
    # dict((F.diff_rule, G.diff_rule, H.diff_rule)),
    (FDiff(Dot(a[b], c), a[d]), Dirac(b, d) * c + Dot(a[b], PDiff(c, a[d]))),
#     RRep(PDiff.product_rule)
    # PDiff.product_rule, PDiff.product_rule, (PDiff(L[a, b, c], d), 0),
    # dict((F.diff_rule, G.diff_rule, H.diff_rule)),
    # (PDiff(PDiff(a, b), c), PDiff(a, b, c)),
    # (Dot(a, b), a*b),
    # (Dirac(a, b) * c[d, e, a] * f, c[d, e, b] * f),
    # ((a+b)*c, a*c+b*c),
    # ((a+b)*c, a*c+b*c)
)
# rep = REPLACE(
#     LiePoisson(F, G),
#     (Dot(a, b), lambda k: Dot(a[k], b[k]), NextFree(k, (a, b), num=1)),
#     dict((F.diff_rule, G.diff_rule, H.diff_rule)),
#     (
#         Lie(a, b)[c],
#         lambda k1, k2: L[k1, k2, c, mu]  * a[k1] * PDiff(b[k2], x[mu]),
#         NextFree(i, (a, b, c), num=1) + NextFree(j, (a, b, c), num=1)
#     ),
# )
# display(rep)
# Markdown(
#     F.f,
#     F.density.f,
#     x.f
# )
a, b, c, d, e, f, g = sp.symbols('a b c d e f g', cls=Symbol)
n, d = Symbol.symbols('n d', integer=True)





class Sum(Expr):
    _mlatex = r'\sum_{{{args[1]}={args[2]}}}^{{{args[3]}}} {args[0]}'

class Underbrace(Expr):
    down_precedence = Par.down_precedence
    _mlatex = r'\underbrace{{{args[0]}}}_{{{args[1]}}}'

class Overbrace(Expr):
    _mlatex = r'\overbrace{{{args[0]}}}^{{{args[1]}}}'

class Definition(Expr, Expands):
    def _post_new(self):
        if len(self.args) >= 2 and isinstance(self.args[1], str):
            return self.replace_at(1, Symbol(self.args[1]))
    def _post_new_up(self, expr, i):
        return expr.replace_at(i, self.args[1])
    def __new__(cls, *args, **kwargs):
        rv = Expr.__new__(cls, *args, **kwargs)
        if rv.verbose:
            tprint('Definition ' + rv.ref_label)
        return rv
    # _mlatex = r'{self.args[1]}'
    @property
    def sm(self): return f'{self.label} {self.lhs.l}'
    # @property
    # def m(self): return f'_{self.label}_{self.expr.bl}'
    # @pr
    @property
    def abl(self): return self.expr.abl
    @property
    def bl(self): return self.align(self.abl)
    @property
    def m(self): return f'_{self.label}_{self.bl}'
    @property
    def im(self): return f'_{self.label}_ {self.expr.l}'
    @property
    def lexpr(self): return self.args[-1]
    @property
    def lm(self): return f'{self.label} {self.lexpr.l}'
    @property
    def il(self): return sp.latex(self.expr)#f'${sp.latex(self.expr)}$'

    @property
    def label(self): return self.args[0]
    @property
    def expr(self): return self.args[1]
    @property
    def lhs(self): return self.expr.args[0]
    @property
    def rhs(self): return self.expr.args[1]
    def _repr_latex_(self):
        return self.bl


class Equation(Definition):
    @property
    def expr(self): return Eq(*self.args[1:])
    # @property
    # def bl(self): return self.align(self.expr.abl)
    @property
    def m(self): return f'_{self.label}_{self.bl}'
    def rule(self, vars):
        rep = {
            var: Wild(var.name) for var in vars
        }
        return {
            self.lhs.subs(rep): self.last.subs(rep)
        }
    @property
    def ref(self): return rf'(\ref{{{self.ref_label}}})'
    @property
    def label_cf(self): return f'{self.label} (cf. {self.lref})'

class Relations(Definition):
    @property
    def label(self): return ''
    @property
    def abl(self):
        return '\\\\\n'.join([
            arg.al for arg in self.args
        ])

def ZeroEqs(*args):
    return Relations(*[Eq(0, arg) for arg in args])

class MyTranspose(Expr):
    _mlatex = r'\left({args[0]}\right)^T'
Transpose = MyTranspose
class ComplexTranspose(Expr):
    _mlatex = r'{args[0]}^*'

class MyMatrix(Expr):
    is_scalar = False
    def _post_new(self):
        import numpy as np
        if isinstance(self.expr, sp.Matrix):
            return self.func(np.array(self.expr))
        if isinstance(self.expr, np.ndarray):
            return self.func(self.expr.tolist())
        if isinstance(self.expr, list):
            return self.func(sp.Tuple(*map(self.func, self.expr)))
        if isinstance(self.expr, sp.Tuple):
            return None
        return self.expr
    def __array__(self):
        import numpy as np
        if self.rank == 1:
            return np.array(self.expr)
        return np.array([
            np.array(row) for row in self
        ])
    def tolist(self): return self.__array__().tolist()
    @property
    def sp(self): return sp.Matrix(self.tolist())
    def inv(self): return self.func(self.sp.inv().tolist())
    def det(self): return self.sp.det()
    def __matmul__(self, other):
        import numpy as np
        return self.func(np.tensordot(self, other, 1).tolist())
    @property
    def T(self):
        import numpy as np
        return self.func(np.array(self).T.tolist())
    def transpose(self, *axes):
        import numpy as np
        return self.func(np.transpose(np.array(self), axes).tolist())


    @property
    def rank(self): return getattr(self.expr[0], 'rank', 0) + 1
    # @cache.memoize()
    def _latex(self, printer=None):
        if self.rank == 0:
            return printer.doprint(self.expr)
        elif self.rank == 1:
            inner = r'\\'.join(map(printer.doprint, self))
        # elif self.rank == 2:
        else:
            inner = r'\\'.join([
                '&'.join(map(printer.doprint, row))
                for row in self
            ])
        # else:
        #     inner = r'\\'.join([
        #         '&'.join(map(
        #             functools.partial(MyMatrix._latex, printer=printer),
        #             row
        #         ))
        #         for row in self
        #     ])
        #     raise NotImplementedError()
        return r'\begin{bmatrix}' + inner + '\end{bmatrix}'
    def __iter__(self): return iter(self.expr)
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = key,
        if len(key) == 1:
            return self.expr[key[0]]
        return self.expr[key[0]][key[1:]]
Matrix = MyMatrix

def gMatrixHelper(args, expr, cls=Matrix):
    return gMatrix(expr, *args, cls=cls)

def gMatrix(expr, *args, cls=Matrix):
    if isinstance(expr, Wild):
        return functools.partial(gMatrixHelper, args, cls=cls)
    if not args:
        return expr
    return cls([
        gMatrix(expr.subs(args[0][0], 1+ii), *args[1:], cls=list)
        for ii in range(args[0][1])
    ])

def cMatrix(f, shape, cls=Matrix):
    if not shape:
        return f()
    return cls([
        cMatrix(functools.partial(f, i), shape[1:], cls=list)
        for i in range(shape[0])
    ])

def gSum(expr, *args, cls=sum):
    if isinstance(expr, Wild):
        return functools.partial(gMatrixHelper, args, cls=cls)
    if not args:
        return expr
    return cls([
        gSum(expr.subs(args[0][0], 1+ii), *args[1:], cls=cls)
        for ii in range(args[0][1])
    ])

# def wrap(func, name=None):
#     class Wrapped(Expr):
#         def _post_new(self):
#             if isinstance(self.expr, Matrix):
#                 return self.expr.func(Map(self.func, self.expr))
#             return func(self.expr)
#     if name is None:
#         name = func.__name__
#     Wrapped.__name__ = name
#     return Wrapped

class Wrapped(Expr):
    def _post_new(self):
        if isinstance(self.expr, Matrix):
            return self.expr.func(Map(self.func, self.expr))
        return self._func(self.expr)

class Together(Wrapped):
    _func = sp.together

class Apart(Wrapped):
    _func = sp.together

class Simplify(Wrapped):
    _func = cache.memoize()(sp.simplify)
# class Together(Expr):
#     def _post_new(self):
#         if isinstance(self.expr, Matrix):
#             return self.expr.func(Map(self.func, self.expr))
#         return sp.together(self.expr)
# Together = wrap(sp.together, name='Together')
# Apart = wrap(sp.apart, name='Apart')
# Simplify = wrap(cache.memoize()(sp.simplify), name='Simplify')

# def Sum(expr, *args):
#     if not args:
#         return expr
#     return sum([
#         Sum(expr.subs(args[0][0], 1+ii), *args[1:])
#         for ii in range(args[0][1])
#     ])
# class After(BinaryOperator):
#     _op = '\circ'
#     def __call__(arg):
#         return self.lhs(self.rhs(arg))

# class Cases(Expr):
#     pass
class Sorted(Expr):
    pass
    # _mlatex = r'\text{{sorted}}({args})'

class mRational(Expr):
    _mlatex = r'\frac{{{args[0]}}}{{{args[1]}}}'

cdot = Symbol('\cdot')
blank = Symbol('')
dots = Symbol('\dots')

class MyTensor(Expr):
    up_precedence = BasicSymbol.up_precedence
    _mlatex = r'{args[0]}'
    @property
    def is_Symbol(self): return self.args[0].is_Symbol
    def _post_new(self):
        if isinstance(self.args[0], str):
            return self.replace_expr(Symbol(self.expr))
            # return self.func(Symbol(self.args[0]), *self.args[1:])
    @property
    def p(self): return self.args[1]
    @property
    def q(self): return self.args[2]
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = key,
        return SupSub(self, key[:self.p], key[self.p:])

    @property
    def inv(self):
        return self.func(self.args[0], self.args[2], self.args[1])
    @property
    def hat(self): return self.replace_expr(self.expr.hat)
Tensor = MyTensor

class Vector(Tensor):
    p, q = 1, 0

class Covector(Tensor):
    p, q = 0, 1

class cTensor(MyTensor):
    constant = True
# class aTensor(Tensor):
#     @property
#     def rules(self):
#         return {
#             s
#         }

u = Tensor('u', 1, 0)
v = Tensor('v', 1, 0)
w = Tensor('w', 1, 0)
P = Tensor('P', 1, 1)
g = Tensor('g', 3, 0)
g1 = Tensor('g', 2, 0)
gi = g.inv
b = Tensor('b', 3, 1)
b1 = Tensor('b', 2, 1)
L = Tensor('L', 3, 0)
LL = Tensor('\mathcal{L}', 2, 0)
Dirac0 = cTensor('delta', 0, 2)
Dirac1 = Dirac = cTensor('delta', 1, 1)
Dirac2 = cTensor('delta', 2, 1)
uh = Tensor(r'\hat{u}', 1, 0)
ih = Symbol(r'\hat{i}')
jh = Symbol(r'\hat{j}')
kh = Symbol(r'\hat{k}')
gamma = Tensor('Gamma', 1, 2)
Gamma = Tensor('Gamma', 1, 2)

hydro_brackets_u = Equation(
    'multidimensional Poisson brackets of hydrodynamic type',
    Poisson(u[i](x), u[j](y)),
    g[mu,i, j](u(x)) * Prime(delta)[mu](x - y) +
    u[k, mu]*b[mu,i,j, k](u(x))*delta(x-y)
    # r'g^{ij}' + '\n'
)
# hydrodynamic = Ref(hydro_brackets_u, 'hydrodynamic')
dxmu = PDiff(blank, x[mu])
L_op = Equation(
    'differential operator',
    L[i, j],
    g[mu,i,j] * dxmu +
    b[mu,i,j,k] * u[k, mu]
)

hydro_brackets_h = Equation(
    '',#'multidimensional Poisson brackets of hydrodynamic type',
    Poisson(F, H),
    sp.Integral(
        PDiff(F.density, u[i]) *
        L_op.lhs *
        PDiff(H.density, u[j]),
        x
    )
)
u2v = Equation(
    'change of coordinates',
    uh, P(u)
)
u2v_jac = Equation(
    'jacobian',
    P[ih, i], PDiff(u[ih], u[i])
)
gv = Equation(
    '',
    g[mu,ih, jh],
    P[ih, i] * g[mu,i, j] * P[jh, j]
)
bv = Equation(
    '',
    b[mu,ih, jh, kh],
    P[ih, i] * (
        b[mu,i,j,k] * P[jh, j] * P[k, kh] +
        g[mu,i,j] * PDiff(P[jh, j], u[kh])
    )
)
L_opv = Equation(
    '',
    L[ih, jh],
    P[ih, i] * L_op.rhs * P[jh, j],
    P[ih, i] * (
        b[mu,i,j,k] * P[jh, j] * u[k, mu] +
        g[mu,i,j] * dxmu(P[jh, j])
    ) +
    gv.rhs * dxmu,
    bv.rhs * u[kh, mu] + gv.rhs * dxmu,
)

hydro_functionals = Equation(
    'Hamiltonians of hydrodynamic type',
    H*Bra(u),
    sp.Integral(H.density(u(x)), x)
)
xspace = Definition(
    '$d$-dimensional spatial manifold',
    M
)
tspace = Definition(
    '$n$-dimensional target manifold',
    U
)
uspace = Definition(
    'function space',
    Cinf(xspace.expr, tspace.expr)
)


fields = Definition(
    'field variables',
    Contains(u, uspace)
)
uu = Definition('', Contains(u, tspace))
christoffel = Definition(
    'connection',
    gamma[j, l, k]
)
btog = Equation(
    'expression',
    b[i, j, k],
    g[i, l] * christoffel.expr
)

xi = Symbol('xi')
S = Tensor(r'\mathcal{S}', 1, 0)
Sd = S[d-1]
diag = Symbol(r'\mathrm{diag}')
symm = Symbol(r'\mathrm{symm}')
asymm = Symbol(r'\mathrm{asymm}')
const = Symbol(r'\mathrm{const}')
#
# hypm = rf"""#Hyperbolic partial differential equations
# We consider a {qpde.m}.
#
# Such a system is called _hyperbolic_ if {hyp.l} is diagonalizable
# over the real numbers for all {Contains(xi, S[Minus(d, 1)]).l}.
#
# If there exist $d n$ {fluxes.im} such that
# {fluxes.args[2].l} holds, the system
# is called a _system of balance laws_.
#
# If furthermore, for a strictly convex {entropy.im} the system admits an {eeb.m}
# satisfying the {compc.im}, the system
# is called _symmetric hyperbolic_.
#
# It is well-known that for a {qpde.label} to be hyperbolic,
# the existence of the above {eef.label} and the existence of
# {fluxes.label} are sufficient but not necessary conditions.
#
# The existence of {fluxes.label} can be easily verified by requiring
# {fluxes_existence.l} to hold in all of {U.l} (see e.g. [Zori2016]). Given an {entropy.sm},
# the same can be verified by requiring {qflux_existence.l}.
# """
#
# poim = rf"""# Poisson brackets of hydrodynamic type
# In [Novi1983] and [Novi1984] Dubrovin and Novikov introduced
# {hydro_brackets_u.m}
# and
# {hydro_functionals.m},
#
# meaning brackets and functionals that do not depend themselves on spatial
# derivatives of the unknown {fields.m},
#
# mapping some $d$-dimensional {xspace.im}
# smoothly onto another $n$-dimensional
# {uspace.im}.
#
# For any two Functionals {F.m} and {H.m} we get
# {hydro_brackets_h.bl}
# with {L_op.im}.
#
#
# It is well known ([Novi1983], theorem 1) that if $d=1$ and $\det g \neq 0$,
# * {btog.expr.l} transform such that {christoffel.expr.l} are the
# Christoffel symbols of a differential-geometric {christoffel.label},
# * {Poisson(cdot, cdot).l} is skew-symmetric iff $g$ is symmetric and the
# {christoffel.im} is consistent with the metric $g$ and
# * {Poisson(cdot, cdot).l} satisfies the Jacobi identity iff the {christoffel.im}
# has no torsion and curvature."""


# Under a {u2v.im} with {u2v_jac.im},
# the {L_op.sm} transforms as {L_opv.bl}
# where {P[k, kh].l} represents the inverse of the jacobian $P$
# such that {Eq(P[k, kh]*P[kh, l], Dirac[k, l]).l},
# and hence we identify {gv.bl} and {bv.bl}.

# Coordinates $u$ such that {Eq(btog.expr, mRational(1,2)*PDiff(g[i, j], u[k])).l}
# are called _Liouville coordinates_, which we will use throughout this section,
# unless otherwise stated.
# from text import *
# import * from text

class MySet(Expr):
    _mlatex1 = r'\left\{{{args[0]}\right\}}'
    _mlatex2 = r'\left\{{{args[0]}\middle|{args[1]}\right\}}'

class Exists(UnaryOperator):
    op = r'\exists'

class SuchThat(BinaryOperator):
    op = ':'

class Fig(Expr):
    def __call__(self, arg): return self.func(arg.__name__, arg, *self.args)
    @property
    def name(self): return self.args[0]
    @property
    def path(self): return f'../images/{self.name}.pdf'
    @property
    def generator(self): return self.args[1]
    @property
    def short_caption(self): return self.args[2]
    @property
    def long_caption(self):
        return self.args[3] if len(self.args) > 2 else self.short_caption
    @property
    def no_rows(self):
        return int(self.args[4]) if len(self.args) > 3 else 1
    @property
    def no_cols(self):
        return int(self.args[5]) if len(self.args) > 4 else 1
    @property
    def ref_label(self): return f'fig:{self.name}'
    @property
    def ref(self): return rf'figure \ref{{{self.ref_label}}}'

    def generate(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(
            self.no_rows, self.no_cols,
            figsize=(16/2.5, 9/2.5*self.no_rows),
            sharex=True, sharey=True
        )
        for _ax in getattr(ax, 'flat', [ax]):
            _ax.grid(True)
        plt.tight_layout(pad=2, rect=(0, 0, 1, .95))
        self.generator(fig, ax)
        fig.savefig(self.path)

    @property
    def bl(self):
        import os
        if False or not os.path.exists(self.path):
            self.generate()
        return rf"""
\begin{{figure}}
    \centering
    \includegraphics[width=\textwidth]{{{self.name}.pdf}}
    \caption[{self.short_caption}]{{{self.long_caption}}}
    \label{{{self.ref_label}}}
\end{{figure}}"""

class Environment(Expr):
    @property
    def l(self): return self.env(self.tag, self.expr)
#         return rf"""
# \begin{{{self.tag}}}
# {self.expr}
# \end{{{self.tag}}}
# """

class TheoremLike(Environment):
    @property
    def l(self):
        text = Environment.l.__get__(self)
        for arg, cls in zip(self.args[1:], (Proof, Remark)):
            if arg is not None and arg != '':
                text += cls(arg).l
        return text

class Theorem(TheoremLike):
    pass

class Lemma(TheoremLike):
    pass

class Corollary(TheoremLike):
    pass

class Proposition(TheoremLike):
    pass

class Def(Environment):
    tag = 'definition'
    @property
    def long_label(self):
        return self.args[-1] if len(self.args) > 1 else ''

class Proof(Environment):
    pass

class Remark(Environment):
    pass

class Ref(Expr):
    @property
    def ref_label(self): return self.expr.ref_label
    @property
    def long_label(self): return self.args[1]
    def __str__(self): return self.long_ref

# hydrodynamic = Ref(hydro_brackets_u, 'hydrodynamic')
class Notebook(Expr):
    # _mlatex = r'\href{{https://nbviewer.jupyter.org/github/nsiccha/hyperbolic_generic/blob/master/py/{args[0]}.ipynb?flush_cache=true}}{{{args[1]}}}'
    # _mlatex = r'\href{{https://colab.research.google.com/github/nsiccha/hyperbolic_generic/blob/master/nb/{args[0]}.ipynb?flush_cache=true}}{{`{args[0]}.ipynb`}}'
    _mlatex = r'[`{args[0]}.ipynb`](https://colab.research.google.com/github/nsiccha/hyperbolic_generic/blob/master/nb/{args[0]}.ipynb?flush_cache=true)'
    @property
    def l(self): return self.il
