from collections import Counter

from hsnf import row_style_hermite_normal_form, smith_normal_form
import numpy as np
from numpy.linalg import inv


def inverse_matrix(m):
    return inv(m).astype(np.int64)


def trailing_zero_row_count(m):
    count = 0
    while m.shape[0] - count - 1 >= 0 and not m[-count - 1].any():
        count += 1
    return count


def integer_division(val, mod):
    if mod == 0:
        return 0, val
    return val // mod, val % mod


class HomologyGroup:
    def __init__(self, a, b):
        self.a, self.b = a, b
        h, q = row_style_hermite_normal_form(b)
        self.u = inverse_matrix(q)
        self.nullity = trailing_zero_row_count(h)
        d, o, self.p = smith_normal_form((a @ self.u)[:, -self.nullity:])
        g = inverse_matrix(self.p)
        nonzero_count = d.shape[0] - trailing_zero_row_count(d)
        self.fillers = o[:nonzero_count]
        self.finite_periods = d.diagonal()[:nonzero_count]
        self.cycles = g[:, -self.nullity:] @ q[-self.nullity:]
        self.torsion_cycles = self.cycles[:nonzero_count]
        self.free_cycles = self.cycles[nonzero_count:]
        self.ones_count = Counter(self.finite_periods)[1]
        self.generating_cycles = np.vstack((self.torsion_cycles[self.ones_count:], self.free_cycles))
        self.zeros_count = self.free_cycles.shape[0]
        self.elementary_divisors = list(self.finite_periods) + [0] * self.zeros_count
        self.pruned_divisors = self.elementary_divisors[self.ones_count:]

    def standardize_cycle(self, z):
        if (z @ self.b).any():
            raise ValueError("Not a cycle", z, z @ self.b)
        coefs = (z @ self.u)[-self.nullity:] @ self.p
        if coefs.shape == (0,):
            return tuple(np.zeros((2, 0), dtype=np.int64))
        error_coefs, cycle_coefs = zip(*[integer_division(v, d) for v, d in zip(coefs, self.elementary_divisors)])
        return np.array(cycle_coefs, dtype=np.int64), np.array(error_coefs, dtype=np.int64)

    def homology_vector(self, z):
        return self.standardize_cycle(z)[0][self.ones_count:]
