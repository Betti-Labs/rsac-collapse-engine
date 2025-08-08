import itertools
import numpy as np
from collections import Counter
from .collapse import (
    extended_signature_from_seq,
    vectorized_extended_signature,
    all_bit_arrays,
)
from .utils import bits_to_seq


# ---------- Basic LUT and no-oracle search ----------
def build_lut_basic(n: int):
    """Build lookup table of signatures to bit assignments for given problem size."""
    groups = {}
    for bits in itertools.product([0, 1], repeat=n):
        key = extended_signature_from_seq(bits_to_seq(bits))
        groups.setdefault(key, []).append(bits)
    ordered = sorted(groups.items(), key=lambda kv: len(kv[1]))
    return groups, ordered


def bucket_search_no_oracle(clauses, n: int, eval_clause_fn):
    """Search through buckets in ascending size order to find satisfying assignment."""
    # Build LUT for this problem size
    groups, lut = build_lut_basic(n)

    checks = 0
    for key, bits_list in lut:
        for bits in bits_list:
            checks += 1
            if all(eval_clause_fn(c, bits) for c in clauses):
                return bits, checks, key
    return None, checks, None


# ---------- RSAC + Unit Propagation + Vectorized partial signatures ----------
_LUT_CACHE = {}
_BITS_CACHE = {}


def get_lut_for_m(m: int):
    if m in _LUT_CACHE:
        return _LUT_CACHE[m], _BITS_CACHE[m]
    bits_mat = all_bit_arrays(m)
    keys = vectorized_extended_signature(bits_mat)
    groups = {}
    for idx, key in enumerate(keys):
        groups.setdefault(key, []).append(idx)
    for k in list(groups.keys()):
        groups[k] = np.array(groups[k], dtype=np.int32)
    _LUT_CACHE[m] = groups
    _BITS_CACHE[m] = bits_mat
    return groups, bits_mat


def rsac_up_vectorized_search(
    clauses_reduced, fixed_assignment: dict, remaining_vars: list
):
    m = len(remaining_vars)
    groups, bits_mat = get_lut_for_m(m)
    ordered = sorted(groups.items(), key=lambda kv: kv[1].shape[0])
    var_to_col = {v: i for i, v in enumerate(remaining_vars)}
    checks = 0

    for key, idxs in ordered:
        rows = bits_mat[idxs]
        B = rows.shape[0]
        row_ok = np.ones(B, dtype=bool)
        for cl in clauses_reduced:
            satisfied = np.zeros(B, dtype=bool)
            for lit in cl:
                v = abs(lit)
                col = var_to_col.get(v, None)
                if col is None:
                    val = fixed_assignment[v]
                    val = val if lit > 0 else 1 - val
                    if val == 1:
                        satisfied[:] = True
                        break
                    else:
                        continue
                vals = rows[:, col]
                if lit < 0:
                    vals = 1 - vals
                satisfied |= vals == 1
            row_ok &= satisfied
            if not row_ok.any():
                break
        if row_ok.any():
            first_idx = int(np.argmax(row_ok))
            checks += first_idx + 1
            row = rows[first_idx]
            asn = dict(fixed_assignment)
            for j, v in enumerate(remaining_vars):
                asn[v] = int(row[j])
            return asn, checks, key, B
        else:
            checks += B
    return None, checks, None, None
