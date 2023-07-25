"""
General-purpose miscellaneous declarations useful to all submodules.
"""

import numpy as np
from functools import wraps

Real = (int, np.int32, np.int64, float, np.float32, np.float64)
NPReal = (*Real, np.ndarray)
Int = (int, np.int32, np.int64)
NPInt = (*Int, np.ndarray)
List_like = (list, tuple)
NPList_like = (*List_like, np.ndarray)
Iterable = (*List_like, set)
NPIterable = (*Iterable, np.ndarray)
Parameter = (*Real, type(None))
Function = type(lambda:0)

class BadType(Exception):
    def __init__(self, argname, given=None, expected=None):
        if given != None and expected != None:
            self.message = f"arg `{argname}` is not of expected type (expected {expected}, but got {given})"
        elif expected == None:
            self.message = f"arg `{argname}` is not of expected type (expected {expected})"
        elif given == None:
            self.message = f"arg `{argname}` is not of expected type (got {given})"
        else:
            self.message = f"arg `{argname}` is not of expected type"
        Exception.__init__(self, self.message)

do_typecheck = True
def typecheck(types, conds=dict()):
    """
    Decorator which performs a high-level check that all arguments passed to a
    given function it satisfy specified type requirements and additional
    conditions expressable as single-argument functions which return `True` or
    `False`.

    Parameters
    ----------
    types : dict-like[str : type or Tuple of types or None]
        Dict-like mapping argument names (as strs) to types or Tuples of types
        or `None`. If an argument name is mapped to `None`, then all types are
        accepted, and additional conditions on the argument are still performed.
    conds : dict-like[int : True/False Callable or Iterable of True/False Callables]
        Dict-like mapping argument names (as strs) to single-argument Callables
        which return `True` or `False` (with `True` being returned when a
        favorable condition is satisfied) or an Iterable of such Callables.
    """
    def decorator(f):
        @wraps(f)
        def checker(*args, **kwargs):
            if do_typecheck:
                argnames = f.__code__.co_varnames
                argvs = dict()
                argvs.update(kwargs)
                argvs.update(dict(zip(argnames, args)))

                for arg in argvs.keys():
                    typespec = types.get(arg, None)
                    if typespec == None:
                        pass
                    elif not isinstance(argvs[arg], typespec):
                        if isinstance(typespec, tuple):
                            expected = [t.__name__ for t in typespec]
                        else:
                            expected = typespec.__name__
                        raise BadType(arg, type(argvs[arg]).__name__, expected)

                    conditions = conds.get(arg, list())
                    if isinstance(conditions, type(lambda:None)):
                        conditions = [conditions]
                    elif not isinstance(conditions, (tuple, set, list)):
                        raise Exception(f"{f.__name__}: typecheck: invalid conditions provided for arg `{arg}`")
                    for cond in conditions:
                        try:
                            cond_check = cond(argvs[arg])
                        except Exception as e:
                            raise Exception(f"{f.__name__}: typecheck: error occurred while testing conditions for arg {arg}:\n{e}")
                        if not cond_check:
                            raise Exception(f"{f.__name__}: typecheck: arg `{arg}` did not meet specified conditions")
            return f(*args, **kwargs)
        return checker
    return decorator

def handler(funcs, value):
    def add_to_handler(func):
        funcs[value] = func
        return func
    return add_to_handler

def gen_table_fmt(label_fmts, s="  ", L=12, P=5, K=2) -> (str, str):
    """
    Generate the column labels and format string of a table from a list of
    tuples following
        (
            'column label',
            x in {'s','s>','i','f','g','e'},
            {l: length override, p: precision override} (optional)
        )
    """
    head = ""
    lines = ""
    fmt = ""
    names = list()
    for label_fmt in label_fmts:
        names.append(label_fmt[0])
        overrides = dict() if len(label_fmt) < 3 else label_fmt[2]
        l = overrides.get("l",
            max(int((len(label_fmt[0])+K-1)/K)*K, L*(label_fmt[1] in ['e','f','g']))
        )
        p = overrides.get("p",
            l-7 if (l-7 >= 1 and l-7 <= P) else P
        )
        head += "{:"+str(l)+"s}"+s
        lines += l*"-" + s
        if label_fmt[1] == 's':
            fmt += "{:"+str(l)+"s}"+s
        elif label_fmt[1] == 's>':
            fmt += "{:>"+str(l)+"s}"+s
        elif label_fmt[1] == 'i':
            fmt += "{:"+str(l)+".0f}"+s
        elif label_fmt[1] in ['e', 'f', 'g']:
            fmt += "{:"+str(l)+"."+str(p)+label_fmt[1]+"}"+s
        else:
            raise Exception("Format is not one of {'s', 's>', 'i', 'f', 'g', 'e'}")
    head = head[:-len(s)]
    lines = lines[:-len(s)]
    fmt = fmt[:-len(s)]
    return head.format(*names)+"\n"+lines, fmt

def print_write(outfile, s, end="\n", flush=True) -> None:
    print(s, end=end, flush=flush)
    outfile.write(s+end)
    if flush:
        outfile.flush()
    return None

def config_fn(filename: str, subtab: str,
        props: list[(str, ..., type, str, type(lambda: ...))]) \
    -> (type(lambda: ConfigStruct), type(ConfigStruct)):
    """
    Returns a function and a class for reading and storing data from a TOML
    config file. Returned functions have the signature

    func(infile: pathlib.Path=pathlib.Path(`filename`))

    and the returned classes have attributes populated according to the elements
    of `props`, where each element is expected in the form

    (config_key, config_key_default, intype, attribute_name, processor)

    where `intype` is a Constructor taking a single argument for a type into
    which the value taken from the file is coerced, and `processor` is a
    function that transforms from the value taken from the file to the value
    ultimately stored in the Config object returned by `func`.

    Parameters
    ----------
    filename : str
        Path to default config file.
    subtab : str
        Key leading to the subtable in the config file storing relevant values.
    props : list[(str, ..., Constructor, str, Function)]
        Specification for processing relevant values.

    Returns
    -------
    func : Function
        Function for pulling and interpreting values from the config file.
    ConfigType : type
        Type for storing values pulled from the config file.
    """
    ConfigType = type(
        "Config",
        (ConfigStruct,),
        {field: proc(default) for _, default, _, field, proc in props}
    )

    def load_config(infile: pathlib.Path=pathlib.Path(filename)) -> ConfigType:
        table = toml.load(infile.open('r'))
        config = ConfigType()
        subtable = table.get(subtab, dict())
        for key, default, typecast, field, proc in props:
            X = typecast(subtable.get(key, default))
            setattr(config, field, proc(X))
        return config

    return (load_config, ConfigType)

def loop_call(func: type(lambda: ...), varies: dict[str, np.ndarray],
        fixes: dict[str, ...], printflag: bool=True, lspace: int=2) \
    -> list[np.ndarray]:
    N = [len(A) for A in varies.values()]
    Z = [int(np.log10(n)) + 1 for n in N]
    tot = prod(N)
    fmt = (
        lspace * " "
        + ";  ".join(f"{{:{z:.0f}.0f}} / {n:.0f}" for z, n in zip(Z, N))
        + ";  [{:6.2f}%] \r"
    )
    outputs = defaultdict(list)
    inputs = product(*[enumerate(A) for A in varies.values()])
    NN = [prod(N[-k:]) for k in range(1, len(N))][::-1] + [1]
    if printflag:
        print(
            fmt.format(*[1 for k in range(len(varies) - 1)], 0, 0.0),
            end="", flush=True
        )
    t0 = timeit.default_timer()
    for kX in inputs:
        K, X = zip(*kX)
        if printflag:
            print(
                fmt.format(
                    *[k + 1 for k in K],
                    100.0 * sum(k * nn for k, nn in zip(K, NN)) / tot
                ),
                end="", flush=True
            )
        Y = func(**dict(zip(varies.keys(), X)), **fixes)
        for j, y in enumerate(Y):
            outputs[j].append(y)
    T = timeit.default_timer() - t0
    if printflag:
        print(
            fmt.format(
                *[k + 1 for k in K],
                100.0 * (sum(k * nn for k, nn in zip(K, NN)) + 1) / tot
            ),
            flush=True
        )
        print(lspace * " " + f"total elapsed time: {T:.2f} s")
        print(lspace * " " + f"average time per call: {T / tot:.2f} s")
    return [np.array(y).reshape(tuple(N)) for y in outputs.values()]

class Q:
    def __call__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return
        elif len(args) == 1 and len(kwargs) == 0:
            print(args[0])
            return args[0]
        else:
            if len(args) > 0:
                print(args)
            if len(kwargs) > 0:
                print(kwargs)
            return args if len(kwargs) == 0 else (args, kwargs)

    def __truediv__(self, X):
        print(X)
        return X

    def __or__(self, X):
        print(X)
        return X
qq = Q()

