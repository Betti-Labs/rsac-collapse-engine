import argparse, random, os, json
import numpy as np
import matplotlib.pyplot as plt

from rsac.collapse import (
    extended_signature_from_seq,
    vectorized_extended_signature,
    all_bit_arrays,
)
from rsac.sat import (
    gen_random_kcnf,
    sat_bruteforce,
    eval_clause,
    unit_propagate,
    pure_literal_elim,
    simplify_with_assignment,
)
from rsac.bucket_search import (
    build_lut_basic,
    bucket_search_no_oracle,
    rsac_up_vectorized_search,
)
from rsac.utils import bits_to_seq


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("paper/figs", exist_ok=True)


def series_sat(ns, instances):
    ensure_dirs()
    results = []
    for n in ns:
        # Build once per n
        groups = {}
        ordered_keys = []
        # Use basic LUT build (Python) for clarity here
        groups, ordered = build_lut_basic(n)

        for i in range(instances):
            rng = random.Random(4242 + 97 * i + n)
            clauses = gen_random_kcnf(
                n, rng.randint(int(1.2 * n), int(1.8 * n)), 3, rng
            )
            _, bf_checks = sat_bruteforce(clauses, n)
            _, rsac_checks, key = bucket_search_no_oracle(clauses, n, eval_clause)
            results.append({"n": n, "bf_checks": bf_checks, "rsac_checks": rsac_checks})
    # Save CSV
    import csv

    with open("data/sat_series_checks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "bf_checks", "rsac_checks"])
        w.writeheader()
        w.writerows(results)

    # Plot averages
    import pandas as pd

    df = pd.DataFrame(results)
    g = df.groupby("n").mean()
    xs = list(g.index)
    bf = list(g["bf_checks"])
    rs = list(g["rsac_checks"])

    plt.figure()
    plt.plot(xs, bf, marker="o", label="BF (avg checks)")
    plt.plot(xs, rs, marker="s", label="RSAC (avg checks)")
    plt.title("SAT: Average checks vs n")
    plt.xlabel("Variables (n)")
    plt.ylabel("Average checks")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("paper/figs/fig_checks_vs_n.png", dpi=200)
    plt.close()

    plt.figure()
    sp = [bf[i] / rs[i] for i in range(len(xs))]
    plt.plot(xs, sp, marker="o")
    plt.title("SAT: Speedup (BF/RSAC)")
    plt.xlabel("n")
    plt.ylabel("Speedup (x)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("paper/figs/fig_speedup.png", dpi=200)
    plt.close()


def series_sat_up(n, instances):
    ensure_dirs()
    rows = []
    for i in range(instances):
        rng = random.Random(9999 + 131 * i + n)
        clauses = gen_random_kcnf(n, rng.randint(int(1.6 * n), int(2.0 * n)), 3, rng)

        # BF on full n
        _, bf_checks = sat_bruteforce(clauses, n)

        # UP + pure, reduce CNF
        c2, fixed = unit_propagate(clauses)
        if c2 is None:
            continue
        c3, fixed = pure_literal_elim(c2, fixed if fixed else {})
        if c3 is None:
            continue
        cnf_reduced = simplify_with_assignment(clauses, fixed)
        if cnf_reduced is None:
            continue

        rem_vars = [v for v in range(1, n + 1) if v not in fixed]
        _, rsac_checks, key, bucket = rsac_up_vectorized_search(
            cnf_reduced, fixed, rem_vars
        )
        rows.append(
            {
                "n": n,
                "fixed": len(fixed),
                "remaining": len(rem_vars),
                "bf_checks": bf_checks,
                "rsac_checks": rsac_checks,
                "bucket": bucket,
            }
        )

    import csv

    with open("data/sat_up_n{}.csv".format(n), "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "fixed",
                "remaining",
                "bf_checks",
                "rsac_checks",
                "bucket",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    # Simple printout
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        print(df.describe())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", choices=["sat", "sat-up"], required=True)
    ap.add_argument("--ns", nargs="*", type=int, help="n values for SAT series")
    ap.add_argument("--n", type=int, help="single n for SAT-UP")
    ap.add_argument("--instances", type=int, default=6)
    args = ap.parse_args()

    if args.series == "sat":
        if not args.ns:
            raise SystemExit("--ns required for series=sat")
        series_sat(args.ns, args.instances)
    else:
        if not args.n:
            raise SystemExit("--n required for series=sat-up")
        series_sat_up(args.n, args.instances)


if __name__ == "__main__":
    main()
