# Copyright 2025 Gregory Betti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools, random
from collections import Counter


def gen_random_kcnf(num_vars: int, num_clauses: int, k: int, rng: random.Random):
    clauses = []
    for _ in range(num_clauses):
        vars_chosen = rng.sample(range(1, num_vars + 1), k)
        clause = [rng.choice([1, -1]) * v for v in vars_chosen]
        clauses.append(clause)
    return clauses


def eval_clause(clause, bits_tuple):
    for lit in clause:
        v = abs(lit) - 1
        val = bits_tuple[v]
        if lit < 0:
            val = 1 - val
        if val == 1:
            return 1
    return 0


def sat_bruteforce(clauses, n):
    checks = 0
    for bits in itertools.product([0, 1], repeat=n):
        checks += 1
        if all(eval_clause(c, bits) for c in clauses):
            return bits, checks
    return None, checks


# ---------- Simplification ----------
def simplify_with_assignment(clauses, assignment_partial: dict):
    new_clauses = []
    for cl in clauses:
        satisfied = False
        new_clause = []
        for lit in cl:
            v = abs(lit)
            sign = 1 if lit > 0 else -1
            if v in assignment_partial:
                val = assignment_partial[v]
                lit_val = val if sign > 0 else 1 - val
                if lit_val == 1:
                    satisfied = True
                    break
                else:
                    continue
            else:
                new_clause.append(lit)
        if not satisfied:
            if len(new_clause) == 0:
                return None
            new_clauses.append(new_clause)
    return new_clauses


def unit_propagate(clauses):
    assignment = {}
    changed = True
    while changed:
        changed = False
        unit_literals = [cl[0] for cl in clauses if len(cl) == 1]
        if not unit_literals:
            break
        for lit in unit_literals:
            v = abs(lit)
            desired = 1 if lit > 0 else 0
            if v in assignment and assignment[v] != desired:
                return None, None
            assignment[v] = desired
        clauses = simplify_with_assignment(clauses, assignment)
        if clauses is None:
            return None, None
        changed = True
    return clauses, assignment


def pure_literal_elim(clauses, current_assignment):
    changed = True
    while changed:
        changed = False
        counts_pos = Counter()
        counts_neg = Counter()
        for cl in clauses:
            for lit in cl:
                if lit > 0:
                    counts_pos[lit] += 1
                else:
                    counts_neg[-lit] += 1
        assigns = {}
        for v in set(list(counts_pos.keys()) + list(counts_neg.keys())):
            pos = counts_pos.get(v, 0)
            neg = counts_neg.get(v, 0)
            if pos > 0 and neg == 0:
                assigns[v] = 1
            elif neg > 0 and pos == 0:
                assigns[v] = 0
        if assigns:
            current_assignment.update(assigns)
            clauses = simplify_with_assignment(clauses, assigns)
            if clauses is None:
                return None, None
            changed = True
    return clauses, current_assignment
