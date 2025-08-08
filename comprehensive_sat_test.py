#!/usr/bin/env python3
"""
Comprehensive SAT testing for RSAC vs traditional approaches.
Tests on various problem types and sizes.
"""

import time
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.rsac.sat import gen_random_kcnf, sat_bruteforce, eval_clause
from src.rsac.bucket_search import bucket_search_no_oracle


def generate_pigeonhole_cnf(n_pigeons, n_holes):
    """Generate pigeonhole principle CNF (unsatisfiable for n_pigeons > n_holes)."""
    clauses = []
    var_count = 0

    # Variable mapping: x[i][j] = variable for pigeon i in hole j
    var_map = {}
    for i in range(n_pigeons):
        for j in range(n_holes):
            var_count += 1
            var_map[(i, j)] = var_count

    # Each pigeon must be in at least one hole
    for i in range(n_pigeons):
        clause = [var_map[(i, j)] for j in range(n_holes)]
        clauses.append(clause)

    # No two pigeons in the same hole
    for j in range(n_holes):
        for i1 in range(n_pigeons):
            for i2 in range(i1 + 1, n_pigeons):
                clauses.append([-var_map[(i1, j)], -var_map[(i2, j)]])

    return clauses, var_count


def generate_graph_coloring_cnf(edges, n_colors):
    """Generate graph coloring CNF."""
    n_vertices = max(max(edge) for edge in edges) + 1
    clauses = []
    var_count = 0

    # Variable mapping: x[v][c] = variable for vertex v with color c
    var_map = {}
    for v in range(n_vertices):
        for c in range(n_colors):
            var_count += 1
            var_map[(v, c)] = var_count

    # Each vertex must have at least one color
    for v in range(n_vertices):
        clause = [var_map[(v, c)] for c in range(n_colors)]
        clauses.append(clause)

    # Each vertex has at most one color
    for v in range(n_vertices):
        for c1 in range(n_colors):
            for c2 in range(c1 + 1, n_colors):
                clauses.append([-var_map[(v, c1)], -var_map[(v, c2)]])

    # Adjacent vertices have different colors
    for v1, v2 in edges:
        for c in range(n_colors):
            clauses.append([-var_map[(v1, c)], -var_map[(v2, c)]])

    return clauses, var_count


def create_test_suite():
    """Create a comprehensive test suite."""
    test_cases = []

    # 1. Random 3-SAT instances (various sizes)
    for n in [8, 10, 12, 14, 16]:
        for density in [3.0, 4.0, 5.0]:  # clauses per variable
            rng = random.Random(42 + n * 10 + int(density * 10))
            clauses = gen_random_kcnf(n, int(n * density), 3, rng)
            test_cases.append(
                {
                    "name": f"random_3sat_{n}v_{density}d",
                    "type": "random_3sat",
                    "clauses": clauses,
                    "n_vars": n,
                    "expected_difficulty": "medium" if density < 4.2 else "hard",
                }
            )

    # 2. Pigeonhole instances (unsatisfiable)
    for n_pigeons, n_holes in [(4, 3), (5, 4), (6, 5)]:
        clauses, n_vars = generate_pigeonhole_cnf(n_pigeons, n_holes)
        if n_vars <= 20:  # Keep it manageable for RSAC
            test_cases.append(
                {
                    "name": f"pigeonhole_{n_pigeons}p_{n_holes}h",
                    "type": "pigeonhole",
                    "clauses": clauses,
                    "n_vars": n_vars,
                    "expected_difficulty": "hard",
                    "expected_result": "UNSAT",
                }
            )

    # 3. Graph coloring instances
    # Triangle graph (3-colorable)
    triangle_edges = [(0, 1), (1, 2), (2, 0)]
    for n_colors in [2, 3]:  # 2-coloring should be UNSAT, 3-coloring SAT
        clauses, n_vars = generate_graph_coloring_cnf(triangle_edges, n_colors)
        test_cases.append(
            {
                "name": f"triangle_coloring_{n_colors}c",
                "type": "graph_coloring",
                "clauses": clauses,
                "n_vars": n_vars,
                "expected_difficulty": "easy",
                "expected_result": "UNSAT" if n_colors < 3 else "SAT",
            }
        )

    # 4. Satisfiable instances with known solutions
    # Simple satisfiable patterns
    for n in [6, 8, 10]:
        rng = random.Random(1000 + n)
        # Generate with lower clause density to ensure satisfiability
        clauses = gen_random_kcnf(n, max(n, int(n * 2.5)), 3, rng)
        test_cases.append(
            {
                "name": f"easy_sat_{n}v",
                "type": "easy_sat",
                "clauses": clauses,
                "n_vars": n,
                "expected_difficulty": "easy",
            }
        )

    return test_cases


def run_comprehensive_test(test_cases, max_vars=18):
    """Run comprehensive testing on all test cases."""
    results = []

    print("=" * 80)
    print("COMPREHENSIVE SAT TESTING SUITE")
    print("=" * 80)

    for i, test_case in enumerate(test_cases):
        if test_case["n_vars"] > max_vars:
            print(f"SKIP {test_case['name']}: {test_case['n_vars']} vars > {max_vars}")
            continue

        print(f"\n[{i+1}/{len(test_cases)}] Testing: {test_case['name']}")
        print(f"  Type: {test_case['type']}")
        print(
            f"  Variables: {test_case['n_vars']}, Clauses: {len(test_case['clauses'])}"
        )
        print(f"  Expected difficulty: {test_case['expected_difficulty']}")

        # Run brute force (with timeout for larger instances)
        bf_result = None
        bf_time = None
        bf_checks = None

        if test_case["n_vars"] <= 16:  # Only run BF on smaller instances
            try:
                start_time = time.time()
                bf_result, bf_checks = sat_bruteforce(
                    test_case["clauses"], test_case["n_vars"]
                )
                bf_time = time.time() - start_time
                bf_status = "SAT" if bf_result else "UNSAT"
                print(
                    f"  Brute Force: {bf_status} in {bf_checks} checks ({bf_time:.3f}s)"
                )
            except Exception as e:
                print(f"  Brute Force: ERROR - {e}")
        else:
            print(f"  Brute Force: SKIPPED (too large)")

        # Run RSAC
        try:
            start_time = time.time()
            rsac_result = bucket_search_no_oracle(
                test_case["clauses"], test_case["n_vars"], eval_clause
            )
            rsac_time = time.time() - start_time

            rsac_assignment, rsac_checks, rsac_key = rsac_result
            rsac_status = "SAT" if rsac_assignment else "UNSAT"
            print(f"  RSAC: {rsac_status} in {rsac_checks} checks ({rsac_time:.3f}s)")

            # Calculate speedup if both methods ran
            speedup = None
            if bf_checks and rsac_checks:
                speedup = bf_checks / rsac_checks
                print(f"  Speedup: {speedup:.2f}x")

            # Check correctness
            correctness = "UNKNOWN"
            if bf_result is not None:
                correctness = (
                    "CORRECT"
                    if (bf_result is not None) == (rsac_assignment is not None)
                    else "INCORRECT"
                )

            results.append(
                {
                    "name": test_case["name"],
                    "type": test_case["type"],
                    "n_vars": test_case["n_vars"],
                    "n_clauses": len(test_case["clauses"]),
                    "expected_difficulty": test_case["expected_difficulty"],
                    "bf_result": (
                        bf_status
                        if bf_result is not None or bf_result == False
                        else None
                    ),
                    "bf_checks": bf_checks,
                    "bf_time": bf_time,
                    "rsac_result": rsac_status,
                    "rsac_checks": rsac_checks,
                    "rsac_time": rsac_time,
                    "speedup": speedup,
                    "correctness": correctness,
                }
            )

        except Exception as e:
            print(f"  RSAC: ERROR - {e}")
            results.append(
                {
                    "name": test_case["name"],
                    "type": test_case["type"],
                    "n_vars": test_case["n_vars"],
                    "n_clauses": len(test_case["clauses"]),
                    "error": str(e),
                }
            )

    return results


def analyze_results(results):
    """Analyze and visualize comprehensive test results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # Filter successful results
    successful = [r for r in results if "error" not in r and r.get("speedup")]

    if not successful:
        print("No successful results to analyze.")
        return

    # Create DataFrame for analysis
    df = pd.DataFrame(successful)

    # Summary by problem type
    print("\nResults by Problem Type:")
    print("-" * 40)
    type_summary = (
        df.groupby("type")
        .agg(
            {
                "speedup": ["count", "mean", "std", "max"],
                "rsac_checks": "mean",
                "correctness": lambda x: (x == "CORRECT").sum(),
            }
        )
        .round(2)
    )
    print(type_summary)

    # Summary by problem size
    print("\nResults by Problem Size:")
    print("-" * 40)
    size_summary = (
        df.groupby("n_vars")
        .agg({"speedup": ["count", "mean", "max"], "rsac_time": "mean"})
        .round(3)
    )
    print(size_summary)

    # Best speedups
    print("\nTop 10 Speedups:")
    print("-" * 40)
    top_speedups = df.nlargest(10, "speedup")[
        ["name", "n_vars", "speedup", "rsac_checks"]
    ]
    for _, row in top_speedups.iterrows():
        print(
            f"{row['name']:<25} {row['n_vars']:>3}v {row['speedup']:>8.1f}x {row['rsac_checks']:>6} checks"
        )

    # Correctness check
    correct_count = (df["correctness"] == "CORRECT").sum()
    total_count = len(df)
    print(
        f"\nCorrectness: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)"
    )

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Speedup by problem size
    size_speedups = df.groupby("n_vars")["speedup"].mean()
    ax1.plot(size_speedups.index, size_speedups.values, "bo-")
    ax1.set_xlabel("Number of Variables")
    ax1.set_ylabel("Average Speedup")
    ax1.set_title("RSAC Speedup vs Problem Size")
    ax1.grid(True)

    # Speedup by problem type
    type_speedups = df.groupby("type")["speedup"].mean()
    ax2.bar(range(len(type_speedups)), type_speedups.values)
    ax2.set_xticks(range(len(type_speedups)))
    ax2.set_xticklabels(type_speedups.index, rotation=45)
    ax2.set_ylabel("Average Speedup")
    ax2.set_title("RSAC Speedup by Problem Type")

    # Checks distribution
    ax3.hist(df["rsac_checks"], bins=20, alpha=0.7, edgecolor="black")
    ax3.set_xlabel("RSAC Checks")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Distribution of RSAC Checks")
    ax3.set_yscale("log")

    # Time vs problem size
    ax4.scatter(df["n_vars"], df["rsac_time"], alpha=0.6)
    ax4.set_xlabel("Number of Variables")
    ax4.set_ylabel("RSAC Time (seconds)")
    ax4.set_title("RSAC Runtime vs Problem Size")
    ax4.set_yscale("log")

    plt.tight_layout()
    plt.savefig("comprehensive_sat_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    return df


def main():
    # Create test suite
    print("Creating comprehensive test suite...")
    test_cases = create_test_suite()
    print(f"Generated {len(test_cases)} test cases")

    # Run tests
    results = run_comprehensive_test(test_cases, max_vars=18)

    # Analyze results
    df = analyze_results(results)

    # Save results
    with open("comprehensive_sat_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if df is not None:
        df.to_csv("comprehensive_sat_results.csv", index=False)

    print(f"\nResults saved to comprehensive_sat_results.json and .csv")
    print("Analysis plot saved as comprehensive_sat_analysis.png")


if __name__ == "__main__":
    main()
