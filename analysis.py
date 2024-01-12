from __future__ import annotations
import numpy as np
import lmfit
import toml
import pathlib
import re
from collections import defaultdict
from itertools import product
from typing import Callable, Optional
try:
    from typing import Self
except ImportError:
    from typing import TypeVar
    Self = TypeVar("Self")

def opt_or(x, default):
    return x if x is not None else default

def qq(X):
    print(X)
    return X

def value_str(
    x: float,
    err: float,
    trunc: bool=True,
    sign: bool=False,
    sci: bool=False,
    latex: bool=False,
    dec: Optional[int]=None,
) -> str:
    ord_x = np.floor(np.log10(abs(x)))
    ord_err = (
        np.floor(np.log10(err)) if np.isfinite(err) and err > 1e-12 else None
    )
    if sci:
        xp = (
            round(x / 10**opt_or(ord_err, 0))
            * 10**(opt_or(ord_err, 0) - ord_x)
        )
        errp = (
            round(err / 10**ord_err) * 10**(ord_err - ord_x)
        ) if ord_err is not None else None
        z = max(ord_x - opt_or(ord_err, 0), 0)
    else:
        xp = (
            round(x / 10**opt_or(ord_err, 0))
            * 10**opt_or(ord_err, 0)
        )
        errp = (
            round(err / 10**ord_err) * 10**ord_err
        ) if ord_err is not None else None
        z = max(-opt_or(ord_err, 0), 0)
    z = abs(min(dec, z) if dec is not None else z)
    if trunc:
        outstr = "{}({}){}".format(
            f"{{:+.{z:.0f}f}}".format(xp)
                if sign else f"{{:.{z:.0f}f}}".format(xp),
            "{:.0f}".format(errp * 10**z) if errp is not None else "nan",
            "e{}{:02.0f}".format("-" if ord_x < 0 else "+", abs(ord_x))
                if sci else "",
        )
    else:
        outstr = "{}{ex} {} {}{ex}".format(
            f"{{:+.{z:.0f}f}}".format(xp)
                if sign else f"{{:.{z:.0f}f}}".format(xp),
            r"\pm" if latex else "+/-",
            f"{{:.{z:.0f}f}}".format(errp) if errp is not None else "nan",
            ex="e{}{:02.0f}".format("-" if ord_x < 0 else "+", abs(ord_x))
                if sci else "",
        )
    if latex:
        return "$" + outstr + "$"
    else:
        return outstr

class ExpVal:
    val: float
    err: float

    @staticmethod
    def from_x(f: Self | float | int) -> Self:
        number = (
            float, int,
            np.float16, np.float32, np.float64, np.float128,
            np.int8, np.int16, np.int32, np.int64,
        )
        if isinstance(f, number):
            return ExpVal(f, 0.0)
        elif isinstance(f, ExpVal):
            return f
        else:
            raise NotImplementedError

    @staticmethod
    def from_num_or(f: Self | float | int) -> Self:
        number = (
            float, int,
            np.float16, np.float32, np.float64, np.float128,
            np.int8, np.int16, np.int32, np.int64,
        )
        if isinstance(f, number):
            return ExpVal(f, 0.0)
        elif isinstance(f, ExpVal):
            return f
        else:
            raise Exception("must be either an ExpVal or a regular number")

    @staticmethod
    def from_str(s: str) -> Self:
        SIGN = r"[+\-]"
        NON_NORMAL = "nan|inf"
        EXP = r"e([+\-]?\d+)"
        DIGITS = "[0123456789]"
        number = r"{nn}|(({d}*\.)?{d}+)".format(nn=NON_NORMAL, d=DIGITS)
        trunc = r"(({s}?)({n}))\(({nn}|{d}+)\)({e})?".format(
            s=SIGN, n=number, nn=NON_NORMAL, d=DIGITS, e=EXP,
        )
        pm = r"(({s}?)({n}({e})?))[ ]*(\+[/]?-|\\pm)[ ]*({n}({e})?)".format(
            s=SIGN, n=number, e=EXP,
        )
        rgx = r"^([$])?({t}|{p})([$])?$".format(
            t=trunc, p=pm,
        )
        pat = re.compile(rgx)
        if (cap := pat.match(s.lower())) is not None:
            if bool(cap.group(1)) ^ bool(cap.group(24)):
                raise Exception("unmatched '$'")
            if "(" in cap.group(2):
                val = float(cap.group(3))
                val_str = cap.group(5).replace(".", "")
                z = (
                    0 if val_str == "nan" or val_str == "inf"
                    else len(val_str) - 1
                )
                err = float(cap.group(8))
                p = float(cap.group(10) if cap.group(10) is not None else "0")
                val = val * 10**p
                err = err * 10**(p - z)
            else:
                val = float(cap.group(11))
                err = float(cap.group(19))
        else:
            raise Exception("malformed input")
        return ExpVal(val, err)

    @staticmethod
    def from_lmfit_param(params: lmfit.Parameters, key: str):
        return ExpVal(params[key].value, params[key].stderr)

    def __init__(self, val: float, err: float):
        self.val = val
        self.err = abs(err) if err is not None else np.nan

    def __eq__(self, rhs: Self | float | int) -> bool:
        other = ExpVal.from_num_or(rhs)
        return self.val == other.val

    def __ne__(self, rhs: Self | float | int) -> bool:
        return not (self == rhs)

    def __lt__(self, rhs: Self | float | int) -> bool:
        other = ExpVal.from_num_or(rhs)
        return self.val < other.val

    def __gt__(self, rhs: Self | float | int) -> bool:
        other = ExpVal.from_num_or(rhs)
        return self.val > other.val

    def __le__(self, rhs: Self | float | int) -> bool:
        return not (self > rhs)

    def __ge__(self, rhs: Self | float | int) -> bool:
        return not (self < rhs)

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return ExpVal(
            -self.val,
            self.err,
        )

    def __add__(self, rhs: Self | float | int) -> Self:
        other = ExpVal.from_num_or(rhs)
        return ExpVal(
            self.val + other.val,
            np.sqrt(self.err**2 + other.err**2),
        )

    def __radd__(self, lhs: Self | float | int) -> Self:
        return self + lhs

    def __iadd__(self, rhs: Self | float | int):
        other = ExpVal.from_num_or(rhs)
        self.val += other.val
        self.err = np.sqrt(self.err**2 + other.err**2)

    def __sub__(self, rhs: Self | float | int) -> Self:
        other = ExpVal.from_num_or(rhs)
        return ExpVal(
            self.val - other.val,
            np.sqrt(self.err**2 + other.err**2),
        )

    def __rsub__(self, lhs: Self | float | int):
        other = ExpVal.from_num_or(lhs)
        return ExpVal(
            other.val - self.val,
            np.sqrt(other.err**2 + self.err**2),
        )

    def __isub__(self, rhs: Self | float | int):
        other = ExpVal.from_num_or(rhs)
        self.val -= other.val
        self.err = np.sqrt(self.err**2 + other.err**2)

    def __mul__(self, rhs: Self | float | int) -> Self:
        other = ExpVal.from_num_or(rhs)
        return ExpVal(
            self.val * other.val,
            np.sqrt((self.err * other.val)**2 + (self.val * other.err)**2),
        )

    def __rmul__(self, lhs: Self | float | int) -> Self:
        return self * lhs

    def __imul__(self, rhs: Self | float | int):
        other = ExpVal.from_num_or(rhs)
        self.val *= other.val
        self.err = np.sqrt((self.err * other.val)**2 + (self.val * other.err)**2)

    def __truediv__(self, rhs: Self | float | int) -> Self:
        other = ExpVal.from_num_or(rhs)
        return ExpVal(
            self.val / other.val,
            np.sqrt(
                (self.err / other.val)**2
                + (other.err * self.val / other.val**2)**2
            ),
        )

    def __rtruediv__(self, lhs: Self | float | int) -> Self:
        other = ExpVal.from_num_or(lhs)
        return ExpVal(
            other.val / self.val,
            np.sqrt(
                (other.err / self.val)**2
                + (self.err * other.val / self.val**2)**2
            ),
        )

    def __itruediv__(self, rhs: Self | float | int):
        other = ExpVal.from_num_or(rhs)
        self.val /= other.val
        self.err = np.sqrt(
            (self.err / other.val)**2
            + (other.err * self.val / other.val**2)**2
        )

    def __mod__(self, rhs: Self | float | int) -> Self:
        other = ExpVal.from_num_or(rhs)
        return ExpVal(
            self.val % other.val,
            np.sqrt(
                self.err**2
                + (np.floor(self.val / other.val) * other.err)**2
            ),
        )

    def __rmod__(self, lhs: Self | float | int) -> Self:
        other = ExpVal.from_num_or(lhs)
        return ExpVal(
            other.val % self.val,
            np.sqrt(
                other.err**2
                + (np.floor(other.val / self.val) * self.err)**2
            ),
        )

    def __imod__(self, rhs: Self | float | int):
        other = ExpVal.from_num_or(rhs)
        self.val %= other.val
        self.err = np.sqrt(
            self.err**2
            + (np.floor(self.val / other.val) * other.err)**2
        )

    def abs(self) -> Self:
        return ExpVal(
            abs(self.val),
            self.err,
        )

    def __abs__(self) -> Self:
        return self.abs()

    def abs_sub(self, other: Self | float | int) -> Self:
        other = ExpVal.from_num_or(other)
        return ExpVal(
            abs(self.val - other.val),
            np.sqrt(self.err**2 + other.err**2),
        )

    def acos(self) -> Self:
        return ExpVal(
            np.arccos(self.val),
            self.err / np.sqrt(1 - self.val**2),
        )

    def acosh(self) -> Self:
        return ExpVal(
            np.arccosh(self.val),
            self.err / np.sqrt(self.val**2 - 1),
        )

    def asin(self) -> Self:
        return ExpVal(
            np.arcsin(self.val),
            self.err / np.sqrt(1 - self.val**2),
        )

    def asinh(self) -> Self:
        return ExpVal(
            np.arcsinh(self.val),
            self.err / np.sqrt(self.val**2 + 1),
        )

    def atan(self) -> Self:
        return ExpVal(
            np.arctan(self.val),
            self.err / (self.val**2 + 1),
        )

    def atan2(self, other: Self | float | int) -> Self:
        other = ExpVal.from_num_or(other)
        return ExpVal(
            np.arctan2(self.val, other.val),
            np.sqrt(
                (self.val * other.err)**2
                + (self.err * other.val)**2
            ) / (self.val**2 + other.val**2),
        )

    def atanh(self) -> Self:
        return ExpVal(
            np.arctanh(self.val),
            self.err / abs(self.val**2 - 1),
        )

    def cbrt(self) -> Self:
        return ExpVal(
            pow(self.val, 1 / 3),
            self.err / pow(self.val, 2 / 3) / 3,
        )

    def ceil(self) -> Self:
        return ExpVal(
            np.ceil(self.val),
            0.0,
        )

    def __ceil__(self) -> Self:
        return self.ceil()

    def is_finite(self) -> bool:
        return np.isfinite(self.val)

    def is_inf(self) -> bool:
        return np.isinf(self.val)

    def is_nan(self) -> bool:
        return np.isnan(self.val)

    def is_neginf(self) -> bool:
        return np.isneginf(self.val)

    def is_posinf(self) -> bool:
        return np.isposinf(self.val)

    def cos(self) -> Self:
        return ExpVal(
            np.cos(self.val),
            self.err * abs(np.sin(self.val)),
        )

    def cosh(self) -> Self:
        return ExpVal(
            np.cosh(self.val),
            self.err * abs(np.sinh(self.val)),
        )

    def exp(self) -> Self:
        ex = np.exp(self.val)
        return ExpVal(
            ex,
            self.err * ex,
        )

    def exp2(self) -> Self:
        ex2 = np.exp2(self.val)
        return ExpVal(
            ex2,
            self.err * np.log(2) * ex2,
        )

    def exp_m1(self) -> Self:
        return ExpVal(
            np.expm1(self.val),
            self.err * np.exp(self.val),
        )

    def floor(self) -> Self:
        return ExpVal(
            np.floor(self.val),
            0.0,
        )

    def __floor__(self) -> Self:
        return self.floor()

    def frac(self) -> Self:
        return self % 1

    def hypot(self, other: Self | float | int) -> Self:
        other = ExpVal.from_num_or(other)
        h = np.sqrt(self.val**2 + other.val**2)
        return ExpVal(
            h,
            np.sqrt(
                (self.err * self.val)**2
                + (other.err * other.val)**2
            ) / h,
        )

    @staticmethod
    def infinity() -> Self:
        return ExpVal(
            np.inf,
            np.nan,
        )

    def ln(self) -> Self:
        return ExpVal(
            np.log(self.val),
            self.err / abs(self.val),
        )

    def ln_1p(self) -> Self:
        return ExpVal(
            np.log1p(self.val),
            self.err / abs(self.val + 1),
        )

    def log(self, base: Self | float | int) -> Self:
        base = ExpVal.from_num_or(base)
        return ExpVal(
            np.log(self.val) / np.log(base.val),
            np.sqrt(
                (
                    (self.err / self.val)**2
                    + (
                        base.err
                        * np.log(self.val)
                        / np.log(base.val)
                        / base.val
                    )**2
                ) / np.log(base.val)
            ),
        )

    def log10(self) -> Self:
        return ExpVal(
            np.log10(self.val),
            self.err / np.log(10) / abs(self.val),
        )

    def log2(self) -> Self:
        return ExpVal(
            np.log2(self.val),
            self.err / np.log(2) / abs(self.val),
        )

    def max(self, other: Self | float | int) -> Self:
        other = ExpVal.from_num_or(other)
        if self > other:
            return self
        elif self == other:
            return self
        elif self < other:
            return other
        else:
            raise Exception

    def min(self, other: Self | float | int) -> Self:
        other = ExpVal.from_num_or(other)
        if self > other:
            return other
        elif self == other:
            return self
        elif self < other:
            return self
        else:
            raise Exception

    @staticmethod
    def nan() -> Self:
        return ExpVal(
            np.nan,
            np.nan,
        )

    @staticmethod
    def neg_infinity() -> Self:
        return ExpVal(
            -np.inf,
            np.nan,
        )

    def pow(self, n: Self | float | int) -> Self:
        n = ExpVal.from_num_or(n)
        return ExpVal(
            pow(self.val, n.val),
            pow(self.val, n.val - 1) * np.sqrt(
                (self.err * n.val)**2
                + (n.err * self.val * np.log(self.val))**2
            ),
        )

    def recip(self) -> Self:
        return ExpVal(
            1 / self.val,
            self.err / self.val**2,
        )

    def round(self) -> Self:
        return ExpVal(
            np.round(self.val),
            0.0,
        )

    def __round__(self) -> Self:
        return self.round()

    def signum(self) -> Self:
        return ExpVal(
            -1.0 if self.val < 0.0 else +1.0,
            0.0,
        )

    def sin(self) -> Self:
        return ExpVal(
            np.sin(self.val),
            self.err * abs(np.cos(self.val)),
        )

    def sin_cos(self) -> (Self, Self):
        return (self.sin(), self.cos())

    def sinh(self) -> Self:
        return ExpVal(
            np.sinh(self.val),
            self.err * np.cosh(self.val),
        )

    def sqrt(self) -> Self:
        sq = np.sqrt(self.val)
        return ExpVal(
            sq,
            self.err / sq / 2,
        )

    def tan(self) -> Self:
        return ExpVal(
            np.tan(self.val),
            self.err / np.cos(self.val)**2,
        )

    def tanh(self) -> Self:
        return ExpVal(
            np.tanh(self.val),
            self.err / np.cosh(self.val)**2,
        )

    def trunc(self) -> Self:
        return ExpVal(
            np.trunc(self.val),
            0.0,
        )

    def __repr__(self) -> str:
        return f"ExpVal(val: {self.val:g}, err: {self.err:g})"

    def __str__(self) -> str:
        return value_str(self.val, self.err)

    def value_str(
        self,
        trunc: bool=True,
        sign: bool=False,
        sci: bool=False,
        latex: bool=False,
        dec: Optional[int]=None,
    ) -> str:
        return value_str(self.val, self.err, trunc, sign, sci, latex, dec)

def residuals(
    params: lmfit.Parameters,
    model: Callable[[lmfit.Parameters, np.ndarray, ...], np.ndarray],
    *indep: np.ndarray,
    data: np.ndarray,
    err: np.ndarray
) -> np.ndarray:
    m = model(params, *indep)
    return ((data - m) / err)**2

def struct(name: str=None, **fields):
    return type("Struct" if name is None else name, (), fields)()

def gen_xplot(xmin: float, xmax: float, N: int=1000, k: float=0.1):
    xmid = (xmin + xmax) / 2
    return np.linspace(xmin - k * (xmid - xmin), xmax + k * (xmax - xmid), N)

def gen_imshow_extent(x: np.ndarray, y: np.ndarray, lower=True) \
    -> [float, float, float, float]:
    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    return [
        np.min(x) - dx / 2,
        np.max(x) + dx / 2,
        *[
            np.min(y) - dy / 2,
            np.max(y) + dy / 2,
        ][::+1 if lower else -1]
    ]

def gen_params(names: list[str], init_vals: dict[str, float]=None,
        default_vals: dict[str, float]=None,
        param_bounds: dict[str, (float | None, float | None)]=None,
        default_bounds: dict[str, (float | None, float | None)]=None) \
    -> lmfit.Parameters:
    init_vals = dict() if init_vals is None else init_vals
    default_vals = dict() if default_vals is None else default_vals
    param_bounds = dict() if param_bounds is None else param_bounds
    default_bounds = dict() if default_bounds is None else default_bounds
    params = lmfit.Parameters()
    for v in names:
        val = init_vals.get(v, default_vals.get(v, 0.0))
        bounds = param_bounds.get(v, default_bounds.get(v, (None, None)))
        params.add(v, value=val, min=bounds[0], max=bounds[1])
    return params

def get_paramvals(
    params: lmfit.Parameters,
    *param_names: str
) -> list[tuple[float, float]]:
    return [(params[name].value, params[name].stderr) for name in param_names]

def gen_paramstr(
    params: lmfit.Parameters,
    param_names: dict[str, str],
    model: str = None,
) -> str:
    return (
        (f"${model}$\n" if model is not None else "")
        + "\n".join(
            f"${symbol}"
            f" = {params[name].value:.5f}"
            f" \\pm {params[name].stderr if params[name].stderr is not None else -1.0:.5f}$"
            for symbol, name in param_names.items()
        )
    )

def dict_get_path(D: dict, path: list[...], default=None):
    """
    Attempts to recursively descend through a dict-like tree structure along a
    path given as a list-like of keys. Returns `default` or raises `KeyError` if
    the final or other keys are not found in their appropriate parent nodes.
    """
    assert len(path) > 0
    d = D.get(path[0], default)
    if len(path) == 1:
        return d
    elif not isinstance(d, dict) or path[0] not in D.keys():
        err = f"""
Descent terminated before the end of the path was reached
Terminating value: {d}
Path remaining: {path}
        """
        raise KeyError(err)
    else:
        return dict_get_path(d, path[1:])

def load_results(
    infile: pathlib.Path,
    label_pat: str,
    data_path: list[str] | tuple[list[str]],
    indep_groups: tuple[int]=None,
    group_filters: dict[int, type(lambda: bool)]=None,
    skip_non_match: bool=True,
    print_skipped: bool=False
) -> (
    list[np.ndarray],
    np.ndarray | list[np.ndarray], np.ndarray,
    np.ndarray | list[np.ndarray], np.ndarray
):
    """
    Fetch data from a toml-formatted file, assuming a structure where top-level
    keys label sets of data with values for independent variable(s) that share
    identical key structures terminating at numerical values therein. Returned
    data arrays have dimensionality and number based on the how many independent
    variables are indicated with `indep_groups`.

    Parameters
    ----------
    infile : pathlib.Path
        Path to the results file. Must be in TOML format.
    label_pat : str
        Regex providing the groupings of numerical values for independent
        variables found in the top-level labels for each group of data contained
        in the results file. Each group (if used) will be interpreted as a
        float if possible, and left as a string otherwise.
    data_path : list[str] | tuple[list[str]]
        Dictionary-key path to the dependent variable value in each set of data
        after label selection, or tuple of multiple such paths.
    indep_groups : tuple[int] (optional)
        Tuple of ints selecting which groups of the groups of `label_pattern` to
        use as independent variables. All other groups are averaged over. The
        number of integers in this tuple determines the dimensionality and
        number of arrays returned; e.g. if two groups are specified, then `X`
        will contain two 1D arrays, and `y` and `err` will both be 2D.
    group_filters : dict[int, function] : (optional)
        Dictionary of regex group numbers mapped to functions returning boolean
        values, acting as filters for which data points are included in the
        returned arrays. Each function acts on independent-variable values
        corresponding to the group number mapped to it.
    skip_non_match : bool (optional)
        If True (default), skip labels that do not match `label_pat`; raise an
        exception otherwise.
    print_skipped : bool (optional)
        Print out any non-matching labels if True (default).

    Returns
    -------
    X : list[ numpy.ndarray[dtype=np.float64] | list[str], len=k ]
        List of `k` independent-variable arrays or lists ordered in
        correspondence with the axes of `y` and `err`, where `k` is the length
        of `indep_groups`.
    Y : numpy.ndarray[dtype=np.float64, ndim=k]
            | list[numpy.ndarray[dtype=np.float64, ndim=k]]
        `k`-dimensional array of values for the dependent variable pointed to by
        `data_path`, where `k` is the length of `indep_groups`, or list of such
        arrays for each dependent variable indicated in `data_path`.
    err : numpy.ndarray[dtype=np.float64, ndim=k]
            | list[numpy.ndarray[dtype=np.float64, ndim=k]]
        `k`-dimensional array of error (uncertainty) values for the single
        dependent variable pointed to by `data_path`, where `k` is the length of
        `indep_groups`, or list of such arrays for each dependent variable
        indicated in `data_path`.
    """
    # setup and load data
    pat = re.compile(label_pat)
    indep_groups = tuple([k for k in range(1, pat.groups + 1)]) \
            if indep_groups is None else indep_groups
    filters = dict() if group_filters is None else group_filters
    with infile.open('r') as f:
        res = toml.load(f)
    indeps = defaultdict(set)
    if ( isinstance(data_path, list)
        and all(isinstance(k, str) for k in data_path) ):
        data = defaultdict(list)
    elif ( isinstance(data_path, tuple)
        and all(all(isinstance(k, str) for k in path) for path in data_path) ):
        data = [defaultdict(list) for p in data_path]
    else:
        raise Exception("Invalid value for argument `data_path`:"
            " must be list[str] or tuple[list[str]]")

    # main processing loop to gather values
    for label, dataset in res.items():
        # parse label
        m = pat.match(label)
        if m is None:
            if skip_non_match:
                if print_skipped:
                    print(f"Skipped non-matching label '{label}'")
                continue
            else:
                raise Exception(f"Encountered non-matching label '{label}'")

        # check filters
        fpass = all(
            filters[g](float(m.group(g)))
            for g in range(1, pat.groups + 1)
            if g in filters.keys()
        )
        if not fpass:
            if print_skipped:
                print(f"Skipped filtered label '{label}'")
            continue

        # assemble simplified key from independent variable values and record
        # in set for uniqueness
        key = list()
        for g in indep_groups:
            try:
                v = float(m.group(g))
            except ValueError:
                v = m.group(g)
            key.append(v)
            indeps[g].update({v})
        key = tuple(key)

        # record dependent variable values under simplified key
        if isinstance(data, defaultdict):
            data[key].append(float(dict_get_path(dataset, data_path)))
        else:
            for d, p in zip(data, data_path):
                d[key].append(float(dict_get_path(dataset, p)))

    # secondary loop to compute averages and uncertainties, and assemble into
    # return structures
    X = [sorted(indeps[g]) for g in indep_groups]
    if isinstance(data, defaultdict):
        Y = np.zeros([len(x) for x in X], dtype=np.float64)
        err = np.zeros([len(x) for x in X], dtype=np.float64)
        for indeps, depset in data.items():
            idx = tuple([x.index(x0) for x, x0 in zip(X, indeps)])
            Y[idx] = np.mean(depset)
            err[idx] = np.std(depset)
    else:
        Y = [np.zeros([len(x) for x in X], dtype=np.float64) for d in data]
        err = [np.zeros([len(x) for x in X], dtype=np.float64) for d in data]
        for k, d in enumerate(data):
            for indeps, depset in d.items():
                idx = tuple([x.index(x0) for x, x0 in zip(X, indeps)])
                Y[k][idx] = np.mean(depset)
                err[k][idx] = np.std(depset)
    X = [np.array(x) if isinstance(x[0], float) else x for x in X]

    return X, Y, err

class Fitter:
    def __init__(self, x: np.ndarray, y: np.ndarray, err: np.ndarray=None):
        self.data = struct("Data",
            x=x, y=y, err=np.ones(x.shape) if err is None else err)
        self.fit = None

    def clone(self) -> Self:
        return Fitter(self.data.x, self.data.y, self.data.err)

    @staticmethod
    def load_results(infile: pathlib.Path, label_pat: str, data_path: list[str],
            indep_group: int,
            group_filters: dict[int, type(lambda: bool)]=None,
            skip_non_match: bool=True, print_skipped: bool=True):
        [x], y, err = load_results(
            infile, label_pat, data_path, (indep_group,), group_filters,
            skip_non_match, print_skipped)
        return Fitter(x, y, err)

    def as_model(self, param_names: list[str],
            model: Callable[[lmfit.Parameters, np.ndarray], np.ndarray],
            derivs: tuple[Callable, Callable]=None,
            init_params: dict[str, float]=None,
            param_bounds: dict[str, (float | None, float | None)]=None,
            overwrite: bool=False):
        assert self.fit is None or overwrite, "Data has aleady been fit"
        assert all(isinstance(x, str) for x in param_names)

        params = gen_params(
            param_names, init_params, dict(), param_bounds, dict())
        fit = lmfit.minimize(
            residuals,
            params,
            args=(model, self.data.x),
            kws={"data": self.data.y, "err": self.data.err},
        )
        if not fit.success:
            raise Exception

        params0: fit.params
        xplot = gen_xplot(self.data.x.min(), self.data.x.max())
        yplot = model(params0, xplot)

        self.fit = struct("Fit",
            params=params_names,
            **{v: (fit.params[v].value, fit.params[v].stderr)
                for v in param_names},
            covar=fit.covar if hasattr(fit, "covar") else None,
            f0=lambda x: model(params0, x),
            f1=(lambda x: derivs[0](params0, x))
                if isinstance(derivs, tuple) and len(derivs) > 0 else None,
            f2=(lambda x: derivs[1](params0, x))
                if isinstance(derivs, tuple) and len(derivs) > 1 else None,
            xplot=xplot,
            yplot=yplot,
        )
        return self

    def as_polynomial(self, deg: int=1, init_params: dict[str, float]=None,
            param_bounds: dict[str, (float | None, float | None)]=None,
            overwrite: bool=False):
        assert self.fit is None or overwrite, "Data has already been fit"
        assert deg >= 0
        varnames = [f"a{k}" for k in range(deg + 1)]

        def model(params: lmfit.Parameters, x: np.ndarray):
            a = [params[v].value for v in varnames]
            return sum(a[k] * x**k for k in range(deg + 1))

        params = gen_params(
            varnames,
            init_params,
            {"a0": self.data.y.mean()},
            param_bounds,
        )
        fit = lmfit.minimize(
            residuals,
            params,
            args=(model, self.data.x),
            kws={"data": self.data.y, "err": self.data.err},
        )
        if not fit.success:
            raise Exception

        params0 = fit.params
        xplot = gen_xplot(self.data.x.min(), self.data.x.max())
        yplot = model(params0, xplot)

        self.fit = struct("Fit",
            params=varnames,
            **{v: (fit.params[v].value, fit.params[v].stderr)
                for v in varnames},
            covar=fit.covar if hasattr(fit, "covar") else None,
            deg=deg,
            f0=lambda x: sum(
                fit.params[f"a{k}"].value * x**k
                for k in range(deg + 1)
            ),
            f1=lambda x: sum(
                k * fit.params[f"a{k}"].value * x**(k - 1)
                for k in range(1, deg + 1)
            ),
            f2=lambda x: sum(
                k * (k - 1) * fit.params[f"a{k}"].value * x**(k - 2)
                for k in range(2, deg + 1)
            ),
            xplot=xplot,
            yplot=yplot,
        )
        return self

    def as_exponential(self, decay: bool=True, rate: bool=False,
            fix_first: bool=False, init_params: dict[str, float]=None,
            param_bounds: dict[str, (float | None, float | None)]=None,
            overwrite: bool=False):
        assert self.fit is None or overwrite, "Data has already been fit"
        varnames = ["A", "y", "B"]
        z = -1 if decay else +1

        def model(params: lmfit.Parameters, x: np.ndarray):
            A = params["A"].value
            y = params["y"].value
            B = params["B"].value
            return A * np.exp(z * x * (y if rate else 1 / y)) + B

        if fix_first:
            params = gen_params(
                varnames[:-1],
                init_params,
                {
                    "A": self.data.y[0],
                    "y": 0.0 if rate else 0.1,
                },
                param_bounds,
                {"y": (0.0, None)},
            )
            params.add("A0", value=self.data.y[0], vary=False)
            params.add("x0", value=self.data.x[0], vary=False)
            params.add("B",
                expr=f"A0 - A * exp({z:.0f} {'*' if rate else '/'} * y * x0)")
        else:
            params = gen_params(
                varnames,
                init_params,
                {
                    "A": self.data.y.max() - self.data.y.min(),
                    "y": 0.0 if rate else 0.1,
                    "B": self.data.y.min(),
                },
                param_bounds,
            )
        fit = lmfit.minimize(
            residuals,
            params,
            args=(model, self.data.x),
            kws={"data": self.data.y, "err": self.data.err},
        )
        if not fit.success:
            raise Exception

        params0 = fit.params
        A = params0["A"].value
        y = params0["y"].value
        B = params0["B"].value
        xplot = gen_xplot(self.data.x.min(), self.data.x.max())
        yplot = model(params0, xplot)

        self.fit = struct("Fit",
            params=varnames,
            **{v: (fit.params[v].value, fit.params[v].stderr)
                for v in varnames},
            covar=fit.covar if hasattr(fit, "covar") else None,
            decay=decay,
            fix_first=fix_first,
            f0=lambda x: A * np.exp(z * y * z) + B,
            f1=lambda x: A * (z * y) * np.exp(z * y * x),
            f2=lambda x: A * y**2 * np.exp(z * y * x),
            xplot=xplot,
            yplot=yplot,
        )
        return self

    def as_gaussian(self, stdev_s: bool=False, fix_max: bool=False,
            init_params: dict[str, float]=None,
            param_bounds: dict[str, (float | None, float | None)]=None,
            overwrite: bool=False):
        assert self.fit is None or overwrite, "Data has already been fit"
        varnames = ["A", "u", "s", "B"]
        z = 2 if stdev_s else 1

        def model(params: lmfit.Parameters, x: np.ndarray):
            A = params["A"].value
            u = params["u"].value
            s = params["s"].value
            B = params["B"].value
            return A * np.exp(-(x - u)**2 / (z * s**2)) + B

        if fix_max:
            params = gen_params(
                ["A", "s"],
                init_params,
                {
                    "A": self.data.y.max(),
                    "s": self.data.x.std(),
                },
                param_bounds,
                {"s": (0.0, None)},
            )
            params.add("A0", value=self.data.y.max(), vary=False)
            params.add("u", value=self.data.x[self.data.y.argmax()], vary=False)
            params.add("B", expr="A0 - A")
        else:
            params = gen_params(
                varnames,
                init_params,
                {
                    "A": self.data.y.max(),
                    "u": self.data.x[self.data.y.argmax()],
                    "s": self.data.x.std(),
                    "B": self.data.y.min(),
                },
                param_bounds,
            )
        fit = lmfit.minimize(
            residuals,
            params,
            args=(model, self.data.x),
            kws={"data": self.data.y, "err": self.data.err},
        )
        if not fit.success:
            raise Exception

        params0 = fit.params
        A = params0["A"].value
        u = params0["u"].value
        s = params0["s"].value
        B = params0["B"].value
        xplot = gen_xplot(self.data.x.min(), self.data.x.max())
        yplot = model(params0, xplot)

        self.fit = struct("Fit",
            params=varnames,
            **{v: (fit.params[v].value, fit.params[v].stderr)
                for v in varnames},
            covar=fit.covar if hasattr(fit, "covar") else None,
            stdev_s=stdev_s,
            fix_max=fix_max,
            f0=lambda x: A * np.exp(-(x - u)**2 / (z * s**2)) + B,
            f1=lambda x: (
                2 * A * (u - x) / (z * s**2) * np.exp(-(x - u)**2 / (z * s**2))
            ),
            f2=lambda x: (
                2 * A * (2 * (x - u)**2 / (z * s**2) - 1) / (z * s**2)
                * np.exp(-(x - u)**2 / (z * s**2))
            ),
            xplot=xplot,
            yplot=yplot,
        )
        return self

    def as_lorentzian(self, fix_max: bool=False,
            init_params: dict[str, float]=None,
            param_bounds: dict[str, (float | None, float | None)]=None,
            overwrite: bool=False):
        assert self.fit is None or overwrite, "Data has already been fit"
        varnames = ["A", "u", "s", "B"]

        def model(params: lmfit.Parameters, x: np.ndarray):
            A = params["A"].value
            u = params["u"].value
            s = params["s"].value
            B = params["B"].value
            return A / (1 + ((x - u) / s)**2) + B

        if fix_max:
            params = gen_params(
                ["A", "s"],
                init_params,
                {
                    "A": self.data.y.max(),
                    "s": self.data.x.std(),
                },
                param_bounds,
                {"s": (0.0, None)},
            )
            params.add("A0", value=self.data.y.max(), vary=False)
            params.add("u", value=self.data.x[self.data.y.argmax()], vary=False)
            params.add("B", expr="A0 - A")
        else:
            params = gen_params(
                varnames,
                init_params,
                {
                    "A": self.data.y.max(),
                    "u": self.data.x[self.data.y.argmax()],
                    "s": self.data.x.std(),
                    "B": self.data.y.min(),
                },
                param_bounds,
            )
        fit = lmfit.minimize(
            residuals,
            params,
            args=(model, self.data.x),
            kws={"data": self.data.y, "err": self.data.err},
        )
        if not fit.success:
            raise Exception

        params0 = fit.params
        A = params0["A"].value
        u = params0["u"].value
        s = params0["s"].value
        B = params0["B"].value
        xplot = gen_xplot(self.data.x.min(), self.data.x.max())
        yplot = model(params0, xplot)

        self.fit = struct("Fit",
            params=varnames,
            **{v: (fit.params[v].value, fit.params[v].stderr)
                for v in varnames},
            covar=fit.covar if hasattr(fit, "covar") else None,
            fix_max=fix_max,
            f0=lambda x: A / (1 + ((x - u) / s)**2),
            f1=lambda x: (
                2 * A * (u - x) / (1 + ((x - u) / s)**2) / s**2
            ),
            f2=lambda x: (
                2 * A * s**2 * (3 * (x - u)**2 - s**2)
                / (s**2 + (x - u)**2)**3
            ),
            xplot=xplot,
            yplot=yplot,
        )
        return self

    def as_oscillatory(self, decay: bool=True,
            init_params: dict[str, float]=None,
            param_bounds: dict[str, (float | None, float | None)]=None,
            overwrite: bool=False):
        assert self.fit is None or overwrite, "Data has already been fit"
        varnames = ["A", "y", "w", "d", "B"]
        z = -1 if decay else +1

        def model(params: lmfit.Parameters, x: np.ndarray):
            A = params["A"].value
            y = params["y"].value
            w = params["w"].value
            d = params["d"].value
            B = params["B"].value
            return A * np.exp(z * y * x) * np.cos(w * x + d) + B

        params = gen_params(
            varnames,
            init_params,
            {
                "A": self.data.y.max() - self.data.t.mean(),
                "B": self.data.y.mean(),
            },
            param_bounds,
            {
                "y": (0.0, None),
                "w": (0.0, None),
                "d": (0.0, 2.0 * np.pi),
            },
        )
        fit = lmfit.minimize(
            residuals,
            params,
            args=(model, self.data.x),
            kws={"data": self.data.y, "err": self.data.err},
        )
        if not fit.success:
            raise Exception

        params0 = fit.params
        A = params0["A"].value
        y = params0["y"].value
        w = params0["w"].value
        d = params0["d"].value
        B = params0["B"].value
        xplot = gen_xplot(self.data.x.min(), self.data.x.max())
        yplot = model(params0, xplot)

        self.fit = struct("Fit",
            params=varnames,
            **{v: (fit.params[v].value, fit.params[v].stderr)
                for v in varnames},
            covar=fit.covar if hasattr(fit, "covar") else None,
            decay=decay,
            f0=lambda x: A * np.exp(z * y * x) * np.cos(w * x + d) + B,
            f1=lambda x: (
                A * (z * y) * np.exp(z * y * x) * np.cos(w * x + d)
                - A * w * np.exp(z * y * x) * np.sin(w * x + d)
            ),
            f2=lambda x: (
                A * y**2 * np.exp(z * y * x) * np.cos(w * x + d)
                - A * w**2 * np.exp(z * y * x) * np.cos(w * x + d)
                - 2 * A * w * (z * y) * np.exp(z * y * x) * np.sin(w * x + d)
            ),
            xplot=xplot,
            yplot=yplot,
        )
        return self

Fitter1D = Fitter

class Fitter2D:
    def __init__(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
            err: np.ndarray=None):
        self.data = Struct("Data",
            X=X, Y=Y, Z=Z, err=np.ones(X.shape) if err is None else err)
        self.fit = None

        def clone(self) -> Self:
            return Fitter2D(
                self.data.X, self.data.Y, self.data.Z, self.data.err)

    @staticmethod
    def load_results(infile: pathlib.Path, label_pat: str, data_path: list[str],
            indep_groups: (int, int),
            group_filters: dict[int, type(lambda: bool)]=None,
            skip_non_match: bool=True, print_skipped: bool=True):
        [x, y], Z, err = load_results(
            infile, label_pat, data_path, indep_groups, group_filters,
            skip_non_match, print_skipped)
        X, Y = np.meshgrid(x, y)
        return Fitter2D(X, Y, Z, err)

    def as_polynomial(self, deg_x: int=1, deg_y: int=1,
            init_params: dict[str, float]=None,
            param_bounds: dict[str, (float | None, float | None)]=None,
            overwrite: bool=False):
        assert self.fit is None or overwrite, "Data has already been fit"
        assert deg_x >= 0
        assert deg_y >= 0
        deg_generator = product(range(deg_x + 1), range(deg_y + 1))
        varnames = [
            f"a{i}{j}" for i, j in product(range(deg_x + 1), range(deg_y + 1))]

        def model(params: lmfit.Parameters, X: np.ndarray, Y: np.ndarray):
            a = [
                [params[i][j].value for j in range(deg_y)]
                for i in range(deg_x)
            ]
            return sum(a[i][j] * X**i * Y**j for i, j in deg_generator)

        params = gen_params(
            varnames,
            init_params,
            {"a00": self.data.Z.mean()},
            param_bounds,
        )
        fit = lmfit.minimize(
            residuals,
            params,
            args=(model, self.data.X, self.data.Y),
            kws={"data": self.data.Z, "err": self.data.err},
        )
        if not fit.success:
            raise Exception

        params0 = fit.params
        xplot = gen_xplot(self.data.X.min(), self.data.X.max())
        yplot = gen_xplot(self.data.Y.min(), self.data.Y.max())
        Xplot, Yplot = np.meshgrid(xplot, yplot)
        Zplot = model(params0, Xplot, Yplot)

        self.fit = struct("Fit",
            params=varnames,
            **{v: (params0[v].value, params0[v].stderr)
                for v in varnames},
            covar=fit.covar if hasattr(fit, "covar") else None,
            deg_x=deg_x,
            deg_y=deg_y,
            f0=lambda x, y: sum(
                params0[f"a{i}{j}"].value * x**i * y**j
                for i, j in product(range(deg_x + 1), range(deg_y + 1))
            ),
            f1=lambda x, y: np.array([
                sum(
                    i * params0[f"a{i}{j}"].value * x**(i - 1) * y**j
                    for i, j in product(range(1, deg_x + 1), range(deg_y + 1))
                ),
                sum(
                    j * params0[f"a{i}{j}"].value * x**i * y**(j - 1)
                    for i, j in product(range(deg_x + 1), range(1, deg_y + 1))
                ),
            ]),
            f2=lambda x, y: (
                sum(
                    i * (i - 1) * params0[f"a{i}{j}"].value * x**(i - 2) * y**j
                    for i, j in product(range(2, deg_x + 1), range(deg_y + 1))
                )
                + sum(
                    j * (j - 1) * params0[f"a{i}{j}"].value * x**i * y**(i - 2)
                    for i, j in product(range(deg_x + 1), range(2, deg_y + 1))
                )
            ),
            Xplot=Xplot,
            Yplot=Yplot,
            Zplot=Zplot,
        )
        return self

    def as_gaussian(self, stdev_s: bool=False, fix_max: bool=False,
            init_params: dict[str, float]=None,
            param_bounds: dict[str, (float | None, float | None)]=None,
            overwrite: bool=False):
        assert self.fit is None or overwrite, "Data has already been fit"
        varnames = ["A", "ux", "uy", "sx", "sy", "th", "B"]
        z = 2 if stdev_s else 1

        def model(params: lmfit.Parameters, X: np.ndarray, Y: np.ndarray):
            A = params["Ax"].value
            ux = params["ux"].value
            uy = params["uy"].value
            sx = params["sx"].value
            sy = params["sy"].value
            th = params["th"].value
            Xrel = X - ux
            Yrel = Y - uy
            Xrot = np.cos(th) * Xrel + np.sin(th) * Yrel
            Yrot = -np.sin(th) * Xrel + np.cos(th) * Yrel
            return A * np.exp(-(Xrot / sx)**2 / z - (Yrot / sy)**2 / z) + B

        if fix_max:
            params = gen_params(
                ["A", "sx", "sy", "th"],
                init_params,
                {
                    "A": self.data.Z.max(),
                    "sx": self.data.X.std(),
                    "sy": self.data.Y.std(),
                },
                param_bounds,
                {
                    "sx": (0.0, None),
                    "sy": (0.0, None),
                    "th": (0.0, np.pi / 2.0),
                },
            )
            params.add("A0", value=self.data.Z.max(), vary=False)
            params.add("ux",
                value=self.data.X[self.data.Z.argmax()], vary=False)
            params.add("uy",
                value=self.data.Y[self.data.Z.argmax()], vary=False)
            params.add("B", expr="A0 - A")
        else:
            params = gen_params(
                varnames,
                init_params,
                {
                    "A": self.data.Z.max(),
                    "ux": self.data.X[self.data.Z.argmax()],
                    "sx": self.data.X.std(),
                    "uy": self.data.Y[self.data.Z.argmax()],
                    "sy": self.data.Y.std(),
                    "B": self.data.Z.min(),
                },
                param_bounds,
                {
                    "sx": (0.0, None),
                    "sy": (0.0, None),
                    "th": (0.0, np.pi / 2.0),
                },
            )
        fit = lmfit.minimize(
            residuals,
            params,
            args=(model, self.data.X, self.data.Y),
            kws={"data": self.data.Z, "err": self.data.err},
        )
        if not fit.success:
            raise Exception

        params0 = fit.params
        A = params0["A"].value
        ux = params0["ux"].value
        uy = params0["uy"].value
        sx = params0["sx"].value
        sy = params0["sy"].value
        th = params0["th"].value
        B = params0["B"].value
        xplot = gen_xplot(self.data.X.min(), self.data.X.max())
        yplot = gen_xplot(self.data.Y.min(), self.data.Y.max())
        Xplot, Yplot = np.meshgrid(xplot, yplot)
        Zplot = model(params0, Xplot, Yplot)

        xp = lambda x, y: np.cos(th) * (x - ux) + np.sin(th) * (y - uy)
        yp = lambda x, y: -np.sin(th) * (x - ux) + np.cos(th) * (y - uy)

        self.fit = struct("Fit",
            params=varnames,
            stdev_s=stdev_s,
            fix_max=fix_max,
            **{v: (params0[v].value, params0[v].stderr)
                for v in varnames},
            covar=fit.covar if hasattr(fit, "covar") else None,
            f0=lambda x, y: (
                A * np.exp(-(xp(x, y) / sx)**2 / z - (yp(x, y) / sy)**2 / z) + B
            ),
            f1=lambda x, y: np.array([
                -2 * A * (
                    + xp(x, y) / (z * sx**2) * np.cos(th)
                    - yp(x, y) / (z * sy**2) * np.sin(th)
                ) * np.exp(-(xp / sx)**2 / z - (yp / sy)**2 / z),
                -2 * A * (
                    + xp(x, y) / (z * sx**2) * np.sin(th)
                    + yp(x, y) / (z * sy**2) * np.cos(th)
                ) * np.exp(-(xp / sx)**2 / z - (yp / sy)**2 / z),
            ]),
            f2=lambda x, y: (
                2 * A / z * (
                    2 / z * ((xp(x, y) / sx**2)**2 + (yp(x, y) / sy**2)**2)
                    - (1 / sx**2 + 1 / sy**2)
                ) * np.exp(-(xp / sx)**2 / z - (yp / sy)**2 / z)
            ),
            Xplot=Xplot,
            Yplot=Yplot,
            Zplot=Zplot,
        )
        return self

    def as_lorentzian(self, fix_max: bool=False,
            init_params: dict[str, float]=None,
            param_bounds: dict[str, (float | None, float | None)]=None,
            overwrite: bool=False):
        assert self.fit is None or overwrite, "Data has already been fit"
        varnames = ["A", "ux", "uy", "sx", "sy", "th", "B"]

        def model(params: lmfit.Parameters, X: np.ndarray, Y: np.ndarray):
            A = params["Ax"].value
            ux = params["ux"].value
            uy = params["uy"].value
            sx = params["sx"].value
            sy = params["sy"].value
            th = params["th"].value
            Xrel = X - ux
            Yrel = Y - uy
            Xrot = np.cos(th) * Xrel + np.sin(th) * Yrel
            Yrot = -np.sin(th) * Xrel + np.cos(th) * Yrel
            return A / (1 + (xrot / sx)**2 + (yrot / sy)**2) + B

        if fix_max:
            params = gen_params(
                ["A", "sx", "sy", "th"],
                init_params,
                {
                    "A": self.data.Z.max(),
                    "sx": self.data.X.std(),
                    "sy": self.data.Y.std(),
                },
                param_bounds,
                {
                    "sx": (0.0, None),
                    "sy": (0.0, None),
                    "th": (0.0, np.pi / 2.0),
                },
            )
            params.add("A0", value=self.data.Z.max(), vary=False)
            params.add("ux",
                value=self.data.X[self.data.Z.argmax()], vary=False)
            params.add("uy",
                value=self.data.Y[self.data.Z.argmax()], vary=False)
            params.add("B", expr="A0 - A")
        else:
            params = gen_params(
                varnames,
                init_params,
                {
                    "A": self.data.Z.max(),
                    "ux": self.data.X[self.data.Z.argmax()],
                    "sx": self.data.X.std(),
                    "uy": self.data.Y[self.data.Z.argmax()],
                    "sy": self.data.Y.std(),
                    "B": self.data.Z.min(),
                },
                param_bounds,
                {
                    "sx": (0.0, None),
                    "sy": (0.0, None),
                    "th": (0.0, np.pi / 2.0),
                },
            )
        fit = lmfit.minimize(
            residuals,
            params,
            args=(model, self.data.X, self.data.Y),
            kws={"data": self.data.Z, "err": self.data.err},
        )
        if not fit.success:
            raise Exception

        params0 = fit.params
        A = params0["A"].value
        ux = params0["ux"].value
        uy = params0["uy"].value
        sx = params0["sx"].value
        sy = params0["sy"].value
        th = params0["th"].value
        B = params0["B"].value
        xplot = gen_xplot(self.data.X.min(), self.data.X.max())
        yplot = gen_xplot(self.data.Y.min(), self.data.Y.max())
        Xplot, Yplot = np.meshgrid(xplot, yplot)
        Zplot = model(params0, Xplot, Yplot)

        xp = lambda x, y: np.cos(th) * (x - ux) + np.sin(th) * (y - uy)
        yp = lambda x, y: -np.sin(th) * (x - ux) + np.cos(th) * (y - uy)

        self.fit = struct("Fit",
            params=varnames,
            fix_max=fix_max,
            **{v: (params0[v].value, params0[v].stderr)
                for v in varnames},
            covar=fit.covar if hasattr(fit, "covar") else None,
            f0=lambda x, y: (
                A / (1 + (xp(x, y) / sx)**2 + (yp(x, y) / sy)**2) + B
            ),
            f1=lambda x, y: np.array([
                -2 * A * (
                    + xp(x, y) / sx**2 * np.cos(th)
                    - yp(x, y) / sy**2 * np.sin(th)
                ) / (1 + (xp(x, y) / sx)**2 + (yp(x, y) / sy)**2)**2,
                -2 * A * (
                    + xp(x, y) / sx**2 * np.sin(th)
                    + yp(x, y) / sy**2 * np.cos(th)
                ) / (1 + (xp(x, y) / sx)**2 + (yp(x, y) / sy)**2)**2,
            ]),
            f2=lambda x, y: (
                2 * A * (
                    8 * ((xp(x, y) / sx**2)**2 + (yp(x, y) / sy**2)**2)
                        / (1 + (xp(x, y) / sx)**2 + (yp(x, y) / sy)**2)
                    - (1 / sx**2 + 1 / sy**2)
                ) / (1 + (xp(x, y) / sx)**2 + (yp(x, y) / sy)**2)**2
            ),
            Xplot=Xplot,
            Yplot=Yplot,
            Zplot=Zplot,
        )
        return self

