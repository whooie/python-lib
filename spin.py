from enum import IntEnum
from itertools import product
from math import factorial, sqrt
from typing import Generator, Optional, Self

class SpinProj:
    """
    A single spin-projection quantum number.

    Data is stored internally as the number of half-units of hbar.
    """
    m: int

    def __init__(self, halves: int):
        """
        Create a new SpinProj.
        """
        self.m = halves

    def __eq__(self, other: Self) -> bool:
        return self.m == other.m

    def __ne__(self, other: Self) -> bool:
        return self.m != other.m

    def __lt__(self, other: Self) -> bool:
        return self.m < other.m

    def __gt__(self, other: Self) -> bool:
        return self.m > other.m

    def __le__(self, other: Self) -> bool:
        return self.m <= other.m

    def __ge__(self, other: Self) -> bool:
        return self.m >= other.m

    def __hash__(self) -> int:
        return hash(self.m)

    def __str__(self) -> str:
        if self.m % 2 == 0:
            return f"proj:{self.m // 2}"
        else:
            return f"proj:{self.m}/2"

    @staticmethod
    def new(halves: int) -> Self:
        """
        Create a new SpinProj.
        """
        return SpinProj(halves)

    def refl(self) -> Self:
        """
        Reflect the projection number across the orthogonal plane; i.e. flip its
        sign.
        """
        return SpinProj(-self.m)

    def raised(self) -> Self:
        """
        Increase the projection number by 1 quantum.
        """
        return SpinProj(self.m + 2)

    def lowered(self) -> Self:
        """
        Decrease the projection number by 1 quantum.
        """
        return SpinProj(self.m - 2)

    def halves(self) -> int:
        """
        Return the projection number as a bare number of halves.
        """
        return self.m

    def f(self) -> float:
        """
        Return the projection number as an ordinary floating-point value.
        """
        return self.m / 2.0

class SpinTotal:
    """
    A single total-spin quantum number.

    Data is stored internally as the number of half-units of hbar.
    """
    j: int

    def __init__(self, halves: int):
        """
        Create a new SpinTotal.
        """
        if halves < 0:
            raise ValueError("total spin quantum number must be positive")
        self.j = halves

    def __eq__(self, other: Self) -> bool:
        return self.j == other.j

    def __ne__(self, other: Self) -> bool:
        return self.j != other.j

    def __lt__(self, other: Self) -> bool:
        return self.j < other.j

    def __gt__(self, other: Self) -> bool:
        return self.j > other.j

    def __le__(self, other: Self) -> bool:
        return self.j <= other.j

    def __ge__(self, other: Self) -> bool:
        return self.j >= other.j

    def __hash__(self) -> int:
        return hash(self.j)

    def __str__(self) -> str:
        if self.j % 2 == 0:
            return f"tot:{self.j // 2}"
        else:
            return f"tot:{self.j}/2"

    @staticmethod
    def new(halves: int) -> Self:
        """
        Create a new SpinTotal.
        """
        return SpinTotal(halves)

    def halves(self) -> int:
        """
        Return the total-spin number as a bare number of halves.
        """
        return self.j

    def f(self) -> float:
        """
        Return the projection number as an ordinary floating-point value.
        """
        return self.j / 2.0

    def iter(self) -> Generator["Spin", None, None]:
        """
        Return a generator yielding all possible `Spin` states with total-spin
        number equal to `self`, starting from most-negative projection number.
        """
        cur = Spin.new_stretched_neg(self)
        yield cur
        while not cur.is_stretched_pos():
            cur = cur.raised()
            yield cur

    def iter_rev(self) -> Generator["Spin", None, None]:
        """
        Return a generator yielding all possible `Spin` states with total-spin
        number equal to `self`, starting from most-positive projection number.
        """
        cur = Spin.new_stretched_pos(self)
        yield cur
        while not cur.is_stretched_neg():
            cur = cur.lowered()
            yield cur

class StretchedState(IntEnum):
    Pos = 0
    Neg = 1

class Order(IntEnum):
    Lt = -1
    Eq = 0
    Gt = +1

class Spin:
    """
    A `(total, projection)` spin quantum number pair.
    """
    tot: SpinTotal
    proj: SpinProj

    def __init__(self, tot: SpinTotal, proj: SpinProj) -> Self:
        """
        Create a new `Spin`.

        Raises `ValueError` if the projection number exceeds the total in
        magnitude or if the two have non-equal parity.
        """
        if proj.m not in range(-tot.j, tot.j + 1) or (proj.m - tot.j) % 2 != 0:
            raise ValueError(
                "invalid spin number pair: projection must not exceed total in"
                " magnitude and must have equal parity"
            )
        self.tot = tot
        self.proj = proj

    def __eq__(self, other: Self) -> bool:
        return self.tot == other.tot and self.proj == other.proj

    def __ne__(self, other: Self) -> bool:
        return self.tot != other.tot or self.proj != other.proj

    def __hash__(self) -> int:
        return hash(self.tot) * hash(self.proj)

    def __str__(self) -> str:
        return f"({str(self.tot)}, {str(self.proj)})"

    @staticmethod
    def new(tot: SpinTotal, proj: SpinProj) -> Self:
        """
        Create a new `Spin`.

        Raises `ValueError` if the projection number exceeds the total in
        magnitude or if the two have non-equal parity.
        """
        return Spin(tot, proj)

    @staticmethod
    def from_halves(tot: int, proj: int) -> Self:
        """
        Create a new `Spin` from bare numbers of halves.

        Raises `ValueError` if the projection number exceeds the total in
        magnitude or if the two have non-equal parity.
        """
        return Spin(SpinTotal.new(tot), SpinProj.new(proj))

    @staticmethod
    def from_halves_pair(tp: (int, int)) -> Self:
        """
        Create a new `Spin` from bare numbers of halves.

        Raises `ValueError` if the projection number exceeds the total in
        magnitude or if the two have non-equal parity.
        """
        return Spin.from_halves(*tp)

    @staticmethod
    def new_stretched_pos(tot: SpinTotal) -> Self:
        """
        Create a new stretched state where the projection number is equal to the
        total in magnitude and oriented in the +z direction.
        """
        return Spin(tot, SpinProj.new(tot.j))

    @staticmethod
    def new_stretched_neg(tot: SpinTotal) -> Self:
        """
        Create a new stretched state where the projection number is equal to the
        total in magnitude and oriented in the -z direction.
        """
        return Spin(tot, SpinProj.new(-tot.j))

    @staticmethod
    def new_stretched(tot: SpinTotal, direction: StretchedState) -> Self:
        """
        Create a new stretched state.
        """
        match direction:
            case StretchedState.Pos:
                return Spin.new_stretched_pos(tot)
            case StretchedState.Neg:
                return Spin.new_stretched_neg(tot)

    def to_halves(self) -> (int, int):
        """
        Return the `(total, projection)` numbers as a pair of bare numbers of
        halves.
        """
        return (self.tot.halves(), self.proj.halves())

    def to_floats(self) -> (float, float):
        """
        Return the `(total, projection)` numbers as a pair of ordinary
        floating-point values.
        """
        return (self.tot.f(), self.proj.f())

    def is_stretched_neg(self) -> bool:
        """
        Return `True` if `self` is a stretched state pointing in the -z
        direction.
        """
        return self.proj.halves() == -self.tot.halves()

    def is_stretched_pos(self) -> bool:
        """
        Return `True` if `self` is a stretched state pointing in the +z
        direction.
        """
        return self.proj.halves() == self.tot.halves()

    def is_stretched(self) -> bool:
        """
        Return `True` if `self` is a stretched state.
        """
        return abs(self.proj.halves()) == self.tot.halves()

    def refl(self) -> Self:
        """
        Reflect the projection number across the orthogonal plane; i.e. flip its
        sign.
        """
        return Spin.new(self.tot, self.proj.refl())

    def raised(self) -> Self:
        """
        Increase the projection number by 1 quantum.

        Raises `ValueError` if `self` is already a positively stretched state.
        """
        if self.is_stretched_pos():
            raise ValueError(f"cannot raise stretched state {str(self)}")
        else:
            return Spin.new(self.tot, self.proj.raised())

    def lowered(self) -> Self:
        """
        Decrease the projection number by 1 quantum.

        Raises `ValueError` if `self` is already a negatively stretched state.
        """
        if self.is_stretched_neg():
            raise ValueError(f"cannot lower stretched state {str(self)}")
        else:
            return Spin.new(self.tot, self.proj.lowered())

    def cmp(self, other: Self) -> Optional[Order]:
        """
        Compare two `Spin`s, if possible.

        Two `Spin`s are comparable if they have equal total-spin numbers.
        """
        if self.tot.halves() == other.tot.halves():
            if self.proj.halves() < other.proj.halves():
                return Order.Lt
            elif self.proj.halves() > other.proj.halves():
                return Order.Gt
            else:
                return Order.Eq
        else:
            return None

def cg(jm1: Spin, jm2: Spin, jm12: Spin) -> float:
    """
    Compute the Clebsch-Gordan coefficient `⟨jm1, jm2∣jm12⟩`.
    """
    (j1, m1) = jm1.halves()
    (j2, m2) = jm2.halves()
    (j12, m12) = jm12.halves()
    if m1 + m2 != m12 or (j1 + j2) % 2 != j12 % 2:
        return 0.0
    else:
        kmin = max(0, max(-(j12 - j2 + m1) // 2, -(j12 - j1 - m2) // 2))
        kmax = min((j1 + j2 - j12) // 2, min((j1 - m1) // 2, (j2 + m2) // 2))
        if kmax < kmin:
            return 0.0
        else:
            def summand(k: int) -> float:
                sign = 1.0 if k % 2 == 0 else -1.0
            f_k = factorial(k)
            f_j1_pj2_mj12_mk = factorial((j1 + j2 - j12) // 2 - k)
            f_j1_mm1_mk = factorial((j1 - m1) // 2 - k)
            f_j2_pm2_mk = factorial((j2 + m2) // 2 - k)
            f_j12_mj2_pm1_pk = factorial((j12 - j2 + m1) // 2 + k)
            f_j12_mj1_mm2_pk = factorial((j12 - j1 - m2) // 2 + k)
            return (
                sign
                / f_k
                / f_j1_pj2_mj12_mk
                / f_j1_mm1_mk
                / f_j2_pm2_mk
                / f_j12_mj2_pm1_pk
                / f_j12_mj1_mm2_pk
            )

            s = sum(summand(k) for k in range(kmin, kmax + 1))
            j12t2_p1 = j12 + 1
            f_j12_pj1_mj2 = factorial((j12 + j1 - j2) // 2)
            f_j12_mj1_pj2 = factorial((j12 - j1 + j2) // 2)
            f_j1_pj2_mj12 = factorial((j1 + j2 - j12) // 2)
            f_j1_pj2_pj12_p1 = factorial((j1 + j2 + j12) // 2 + 1)
            f_j12_pm12 = factorial((j12 + m12) // 2)
            f_j12_mm12 = factorial((j12 - m12) // 2)
            f_j1_mm1 = factorial((j1 - m1) // 2)
            f_j1_pm1 = factorial((j1 + m1) // 2)
            f_j2_mm2 = factorial((j2 - m2) // 2)
            f_j2_pm2 = factorial((j2 + m2) // 2)
            return s * sqrt(
                j12t2_p1
                * f_j12_pj1_mj2
                * f_j12_mj1_pj2
                * f_j1_pj2_mj12
                * f_j12_pm12
                * f_j12_mm12
                * f_j1_mm1
                * f_j1_pm1
                * f_j2_mm2
                * f_j2_pm2
                / f_j1_pj2_pj12_p1
            )

def w3j_sel(jm1: Spin, jm2: Spin, jm3: Spin) -> bool:
    """
    Return `True` if `jm1`, `jm2`, and `jm3` satisfy the selection rules of the
    Wigner 3*j* symbol `(jm1 jm2 jm3)`.
    """
    (j1, m1) = jm1.halves()
    (j2, m2) = jm2.halves()
    (j3, m3) = jm3.halves()
    return (
        m1 + m2 + m3 == 0
        and abs(j1 - j2) <= j3
        and j3 <= j1 + j2
        and (
            not (m1 == 0 and m2 == 0 and m3 == 0)
            and ((j1 + j2 + j3) // 2) % 2 == 0
        )
    )

def w3j(jm1: Spin, jm2: Spin, jm3: Spin) -> float:
    """
    Compute the Wigner 3*j* symbol `(jm1 jm2 jm3)`.
    """
    if not w3j_sel(jm1, jm2, jm3):
        return 0.0
    else:
        j1 = jm1.tot.halves()
        j2 = jm2.tot.halves()
        j3 = jm3.tot.halves()
        m3 = jm3.proj.halves()
        sign = 1.0 if ((j1 - j2 - m3) // 2) % 2 == 0 else -1.0
        denom = sqrt(j3 + 1)
        cg = cg(jm1, jm2, jm3.refl())
        return sign * cg / denom

def w6j(
    j1: SpinTotal,
    j2: SpinTotal,
    j3: SpinTotal,
    j4: SpinTotal,
    j5: SpinTotal,
    j6: SpinTotal,
) -> float:
    """
    Compute the Wigner 6*j* symbol `{j1 j2 j3; j4 j5 j6}` (where `j1`, ...,
    `j3` are in the top row).
    """
    def term_filter(jm: (Spin, Spin, Spin, Spin, Spin, Spin)) -> bool:
        (jm1, jm2, jm3, jm4, jm5, jm6) = jm
        return (
            w3j_sel(jm1.refl(), jm2.refl(), jm3.refl())
            and w3j_sel(jm1, jm5.refl(), jm6)
            and w3j_sel(jm4, jm2, jm6.refl())
            and w3j_sel(jm4.refl(), jm5, jm3)
        )

    def sign(jm: (Spin, Spin, Spin, Spin, Spin, Spin)) -> float:
        (j1, m1) = jm1.halves()
        (j2, m2) = jm2.halves()
        (j3, m3) = jm3.halves()
        (j4, m4) = jm4.halves()
        (j5, m5) = jm5.halves()
        (j6, m6) = jm6.halves()
        k = (j1 - m1 + j2 - m2 + j3 - m3 + j4 - m4 + j5 - m5 + j6 - m6) // 2
        return 1.0 if k % 2 == 0 else -1.0

    def term_map(jm: (Spin, Spin, Spin, Spin, Spin, Spin)) -> float:
        (jm1, jm2, jm3, jm4, jm5, jm6) = jm
        return (
            sign(jm)
            * w3j(jm1.refl(), jm2.refl(), jm3.refl())
            * w3j(jm1, jm5.refl(), jm6)
            * w3j(jm4, jm2, jm6.refl())
            * w3j(jm4.refl(), jm5, jm3)
        )

    return sum(
        term_map(jm)
        for jm in product(
            j1.projections(),
            j2.projections(),
            j3.projections(),
            j4.projections(),
            j5.projections(),
            j6.projections(),
        )
        if term_filter(jm)
    )

