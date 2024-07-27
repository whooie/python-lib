from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import pathlib
import re
from typing import Callable, Optional
try:
    from typing import Self
except ImportError:
    from typing import TypeVar
    Self = TypeVar("Self")
import lmfit
import numpy as np
import toml

def opt_or(x, default):
    return x if x is not None else default

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

    def as_list(self) -> list[float]:
        return [self.val, self.err]

    def as_bounds(self) -> list[float]:
        return [self.val - self.err, self.val + self.err]

    def __init__(self, val: float, err: float):
        self.val = float(val)
        self.err = abs(float(err)) if err is not None else np.nan

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
    err: Optional[np.ndarray],
) -> np.ndarray:
    m = model(params, *indep)
    if err is None:
        return (data - m)**2
    else:
        return ((data - m) / err)**2

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

def get_paramvals(
    params: lmfit.Parameters,
    *param_names: str
) -> list[tuple[float, float]]:
    return [(params[name].value, params[name].stderr) for name in param_names]

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

@dataclass
class Param:
    """
    Settings for a parameter passed to an `lmfit` model.

    Fields
    ------
    value : float
        Initial value for the Levenberg-Marquardt algorithm.
    min : float = -numpy.inf
        Lower bound constraint for the parameter.
    max : float = +numpy.inf
        Upper bound constraint for the parameter.
    vary : bool = True
        Allow this parameter to vary while fitting.
    expr : Optional[str] = None
        Analytical expression to constrain the parameter to the values of other
        parameters.
    brute_step : Optional[float] = None
    """
    value: float
    min: float = -np.inf
    max: float = +np.inf
    vary: bool = True
    expr: str = None
    brute_step: float = None

class ModelBase:
    """
    Abstract type for a model function to fit to.

    Fields
    ------
    MODELSTR : str
        Description of the functional form of the model.
    PARAMS : set[str]
        Set of parameters required to evaluate the model at a point in
        parameter space.
    """
    MODELSTR: str
    PARAMS: set[str]

    def __init__(self):
        pass

    def f(self, params: lmfit.Parameters, x: np.ndarray) -> np.ndarray:
        """
        Model evaluator.
        """
        raise NotImplementedError()

ModelFn = Callable[[lmfit.Parameters, np.ndarray], np.ndarray]
CostFn = Callable[[lmfit.Parameters, ModelFn, ...], np.ndarray]

class Fit1D:
    """
    Simple driver class for the `lmfit.minimizer` model-fitting process.

    Fields
    ------
    model : ModelBase
        Fit to this model. See `help(ModelBase)` for more info.
    init_params : lmfit.Parameters
        lmfit record of initial parameter values.
    fit_result : lmfit.minimizer.MinimizerResult
        Output of the `lmfit.minimizer` call.
    """
    model: ModelBase
    init_params: lmfit.Parameters
    fit_result: lmfit.minimizer.MinimizerResult

    def __init__(self, model: ModelBase, init_params: dict[str, Param]):
        """
        Construct a new model-fitting driver.

        Parameters
        ----------
        model : ModelBase
            Fit to this model. See `help(ModelBase)` for more info.
        init_params : dict[str, Param]
            Dictionary mapping parameter names to initial settings. The
            parameter names contained here are checked against those declared
            by the model.
        """
        self.model = model
        model_keys = set(self.model.PARAMS)
        missing_keys = {k for k in params.keys() if k not in model_keys}
        if len(missing_keys) > 0:
            raise ValueError(f"missing keys: {missing_keys}")
        self.params = lmfit.Parameters()
        for (name, settings) in params.items():
            self.params.add(
                name,
                value=settings.value,
                min=settings.min,
                max=settings.max,
                vary=settings.vary,
                expr=settings.expr,
                brute_step=settings.brute_step,
            )
        self.fit_result = None

    def set_model(self, model: ModelBase):
        """
        Set a new model to fit to.

        Returns `self` afterward.
        """
        self.model = model
        return self

    def set_params(self, params: dict[str, Param]):
        """
        Set new initial parameters.

        Returns `self` afterward.
        """
        for (name, settings) in params.items():
            self.params.add(
                name,
                value=settings.value,
                min=settings.min,
                max=settings.max,
                vary=settings.vary,
                expr=settings.expr,
                brute_step=settings.brute_step,
            )
        return self

    def get_init_params(self) -> lmfit.Parameters:
        """
        Get an `lmfit` record of the initial parameters.

        This is equivalent to direct access of the `init_params` field.
        """
        return self.init_params

    def do_fit(
        self,
        costf: CostFn,
        costf_args: tuple[...],
        *minimizer_args,
        **minimizer_kwargs,
    ):
        """
        Perform the Levenberg-Marquardt fit. Raises `lmfit.MinimizerException`
        if the fit is unseccessful.

        Returns `self` afterward.

        Parameters
        ----------
        costf : CostFn
            Cost function to pass to the lmfit routine. This function must have
            the signature
                costf(lmfit.Parameters, ModelFn, ...) -> numpy.ndarray
            where ModelFn is a function with the signature
                model_fn(lmfit.Parameters, numpy.ndarray) -> numpy.ndarray
        costf_args : tuple[...]
            Tuple of extra arguments to pass to the cost function.
        *minimizer_args : ...
            Extra positional arguments to pass to `lmfit.minimize`.
        **minimizer_kwargs : ...
            Extra keyword arguments to pass to `lmfit.minimze`.
        """
        fit_result = lmfit.minimize(
            costf,
            self.init_params,
            args=(self.model.f, *costf_args),
            *args,
            **kwargs,
        )
        if not fit_result.success:
            raise lmfit.MinimizerException("fit failed")
        self.fit_result = fit_result
        return self

    def is_fit(self) -> bool:
        """
        Return `True` if a successful fit has been performed.
        """
        return self.fit_result is not None

    def get_fit_result(self) -> Optional[lmfit.minimizer.MinimizerResult]:
        """
        Get the bare `lmfit.minimizer.MinimizerResult` returned by
        `lmfit.minimize` if a successful fit has been performed.

        This is equivalent to direct access of the `fit_result` field.
        """
        return self.fit_result

    def get_result_param(self, key: str) -> ExpVal:
        """
        Get the value and standard error of a single fit parameter as an
        `ExpVal`.

        See `help(libscratch.analysis.ExpVal)` for more info.

        Raises `RuntimeError` if a successful fit has not been performed, or
        `ValueError` if `key` is not a valid parameter name for the model.
        """
        if self.fit_result is None:
            raise RuntimeError("model has not been fit")
        elif key not in self.model.PARAMS:
            raise ValueError(f"invalid key {key}")
        else:
            return ExpVal(
                self.fit_result.params[key].value,
                self.fit_result.params[key].stderr,
            )

    def get_result_params(self) -> dict[str, ExpVal]:
        """
        Get the values and standard errors of all fit parameters as `ExpVal`s.

        See `help(libscratch.analysis.ExpVal)` for more info.

        Raises `RuntimeError` if a successful fit has not been performed.
        """
        return {key: self.get_result_param(key) for key in self.model.PARAMS}

    def gen_paramstr(
        self,
        comment: Optional[str]=None,
        units: Optional[dict[str, str]]=None,
        with_modelstr: bool=True,
        latex_math: bool=True,
    ) -> str:
        """
        Generate a single multiline string giving information on the model
        function and its fit parameter values.

        Raises `RuntimeError` if a successful fit has not been performed.

        Parameters
        ----------
        comment : Optional[str] = None
            Optional notes to attach to the end of the string.
        units : Optional[dict[str, str]] = None
            Optional units to attach to all parameters.
        with_modelstr : bool = True
            Include a string representation of the model function.
        latex_math : bool = True
            Wrap parameters in '$' for latex math rendering.
        """
        if self.fit_result is None:
            raise RuntimeError("model has not been fit")
        units = dict() if units is None else units
        rp = self.get_result_params()
        m = lambda s: ("$" + s + "$") if latex_math else s
        return (
            (m(self.model.MODELSTR + "\n") if comment is not None else "")
            + "\n".join(
                (
                    f"{m(name)} = {param.value_str(latex=latex_math)}"
                    if param.err is not None
                    else f"{m(name)} = {f'{param.err:.5f}'}(nan)"
                ) + " " + units.get(name, "")
                for (name, param) in rp
            )
            + ((comment + "\n") if comment is not None else "")
        )

    def gen_fit_curve(self, x: np.ndarray) -> np.ndarray:
        """
        Sample the resulting fit curve.

        Raises `RuntimeError` if a successful fit has not been performed.
        """
        if self.fit_result is None:
            raise RuntimeError("model has not been fit")
        else:
            return self.model.f(self.fit_result.params, x)

