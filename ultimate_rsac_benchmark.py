#!/usr/bin/env python3
"""
ULTIMATE RSAC BENCHMARK - Push for 1000x+ speedups!
Combines all optimizations: hybrid solving, GPU acceleration, advanced signatures.
"""

import time
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.rsac.sat import gen_random_kcnf, sat_bruteforce, eval_clause
from src.rsac.bucket_search import bucket_search_no_oracle
from src.rsac.hybrid_solver import HybridRSACSolver, benchmark_hybrid_solver
from src.rsac.advanced_signatures import (
    AdvancedSignatureGenerator,
    benchmark_signature_variants,
)

# Try to import GPU acceleration
try:
    from src.rsac.gpu_acceleration import GPU_AVAILABLE

    if GPU_AVAILABLE:
        from src.rsac.gpu_acceleration import (
            GPUAcceleratedRSAC,
            benchmark_gpu_acceleration,
        )
    else:
        GPUAcceleratedRSAC = None
except (ImportError, AttributeError):
    GPU_AVAILABLE = False
    GPUAcceleratedRSAC = None


def create_ultimate_test_suite():
    """Create the most challenging test suite possible."""
    test_cases = []

    # 1. Structured satisfiable instances (should have good clustering)
    print("Generating structured satisfiable instances...")
    for n in [10, 12, 14, 16]:
        # Low-density satisfiable instances
        rng = random.Random(1000 + n)
        clauses = gen_random_kcnf(n, max(n, int(n * 2.8)), 3, rng)
        test_cases.append(
            {
                "name": f"structured_sat_{n}v",
                "type": "structured_sat",
                "clauses": clauses,
                "n_vars": n,
                "expected_difficulty": "easy",
            }
        )

    # 2. Hard random instances at the phase transition
    print("Generating phase transition instances...")
    for n in [10, 12, 14, 16]:
        # Around 4.26 clauses per variable (SAT phase transition)
        rng = random.Random(2000 + n)
        clauses = gen_random_kcnf(n, int(n * 4.26), 3, rng)
        test_cases.append(
            {
                "name": f"phase_transition_{n}v",
                "type": "phase_transition",
                "clauses": clauses,
                "n_vars": n,
                "expected_difficulty": "hard",
            }
        )

    # 3. Backdoor instances (few critical variables)
    print("Generating backdoor instances...")
    for n in [12, 14, 16]:
        rng = random.Random(3000 + n)
        # Create instance with backdoor structure
        clauses = []

        # Critical variables (backdoor)
        critical_vars = list(range(1, min(4, n // 3) + 1))

        # Add clauses that depend heavily on critical variables
        for _ in range(int(n * 3)):
            clause = []
            # High probability of including critical variables
            for var in critical_vars:
                if rng.random() < 0.7:
                    clause.append(var if rng.random() < 0.5 else -var)

            # Add some random variables
            while len(clause) < 3:
                var = rng.randint(1, n)
                if var not in [abs(lit) for lit in clause]:
                    clause.append(var if rng.random() < 0.5 else -var)

            clauses.append(clause[:3])

        test_cases.append(
            {
                "name": f"backdoor_{n}v",
                "type": "backdoor",
                "clauses": clauses,
                "n_vars": n,
                "expected_difficulty": "medium",
            }
        )

    # 4. Crafted hard instances
    print("Generating crafted hard instances...")
    for n in [10, 12, 14]:
        rng = random.Random(4000 + n)
        # Create instances designed to be hard for DPLL but maybe good for RSAC
        clauses = []

        # Add conflicting constraints
        for i in range(1, n, 2):
            if i + 1 <= n:
                # Force both variables to be true, but also add conflicts
                clauses.append([i, i + 1, rng.randint(1, n)])
                clauses.append([-i, -i - 1, rng.randint(1, n)])
                clauses.append([i, -i - 1, rng.randint(1, n)])

        # Add random clauses
        for _ in range(int(n * 2)):
            clause = [rng.choice([-1, 1]) * rng.randint(1, n) for _ in range(3)]
            clauses.append(clause)

        test_cases.append(
            {
                "name": f"crafted_hard_{n}v",
                "type": "crafted_hard",
                "clauses": clauses,
                "n_vars": n,
                "expected_difficulty": "hard",
            }
        )

    # 5. Instances with known solutions
    print("Generating instances with planted solutions...")
    for n in [12, 14, 16]:
        rng = random.Random(5000 + n)

        # Plant a specific solution
        solution = [rng.randint(0, 1) for _ in range(n)]

        # Generate clauses that are satisfied by this solution
        clauses = []
        for _ in range(int(n * 3.5)):
            clause = []
            for _ in range(3):
                var = rng.randint(1, n)
                # Make sure clause is satisfied by planted solution
                if solution[var - 1] == 1:
                    clause.append(var)  # Positive literal
                else:
                    clause.append(-var)  # Negative literal

            # Occasionally add a random literal to make it harder
            if rng.random() < 0.3:
                var = rng.randint(1, n)
                clause[rng.randint(0, 2)] = rng.choice([-1, 1]) * var

            clauses.append(clause)

        test_cases.append(
            {
                "name": f"planted_solution_{n}v",
                "type": "planted_solution",
                "clauses": clauses,
                "n_vars": n,
                "expected_difficulty": "medium",
                "planted_solution": solution,
            }
        )

    return test_cases


def run_ultimate_benchmark(test_cases, max_vars=18):
    """Run the ultimate benchmark with all optimizations."""
    results = []

    print("=" * 100)
    print("🚀 ULTIMATE RSAC BENCHMARK - PUSHING FOR 1000x+ SPEEDUPS! 🚀")
    print("=" * 100)

    # Initialize solvers
    hybrid_solver = HybridRSACSolver(max_vars_for_rsac=max_vars)
    signature_generator = AdvancedSignatureGenerator()

    if GPU_AVAILABLE and GPUAcceleratedRSAC:
        try:
            gpu_solver = GPUAcceleratedRSAC()
            print("✅ GPU acceleration available")
        except Exception as e:
            gpu_solver = None
            print(f"❌ GPU acceleration failed: {e}")
    else:
        gpu_solver = None
        print("❌ GPU acceleration not available")

    for i, test_case in enumerate(test_cases):
        if test_case["n_vars"] > max_vars:
            print(f"SKIP {test_case['name']}: {test_case['n_vars']} vars > {max_vars}")
            continue

        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(test_cases)}] 🎯 ULTIMATE TEST: {test_case['name']}")
        print(
            f"Type: {test_case['type']}, Variables: {test_case['n_vars']}, Clauses: {len(test_case['clauses'])}"
        )
        print(f"{'='*80}")

        result = {
            "name": test_case["name"],
            "type": test_case["type"],
            "n_vars": test_case["n_vars"],
            "n_clauses": len(test_case["clauses"]),
            "expected_difficulty": test_case["expected_difficulty"],
        }

        # 1. Baseline: Brute Force (if feasible)
        if test_case["n_vars"] <= 16:
            print("🔥 Running Brute Force baseline...")
            try:
                start_time = time.time()
                bf_result, bf_checks = sat_bruteforce(
                    test_case["clauses"], test_case["n_vars"]
                )
                bf_time = time.time() - start_time
                bf_status = "SAT" if bf_result else "UNSAT"
                print(
                    f"   Brute Force: {bf_status} in {bf_checks:,} checks ({bf_time:.3f}s)"
                )

                result.update(
                    {"bf_result": bf_status, "bf_checks": bf_checks, "bf_time": bf_time}
                )
            except Exception as e:
                print(f"   Brute Force: ERROR - {e}")
                result["bf_result"] = "ERROR"
        else:
            print("   Brute Force: SKIPPED (too large)")
            result["bf_result"] = "SKIPPED"

        # 2. Basic RSAC
        print("🎯 Running Basic RSAC...")
        try:
            start_time = time.time()
            basic_result = bucket_search_no_oracle(
                test_case["clauses"], test_case["n_vars"], eval_clause
            )
            basic_time = time.time() - start_time
            basic_status = "SAT" if basic_result[0] else "UNSAT"
            print(
                f"   Basic RSAC: {basic_status} in {basic_result[1]:,} checks ({basic_time:.3f}s)"
            )

            result.update(
                {
                    "basic_rsac_result": basic_status,
                    "basic_rsac_checks": basic_result[1],
                    "basic_rsac_time": basic_time,
                }
            )
        except Exception as e:
            print(f"   Basic RSAC: ERROR - {e}")
            result["basic_rsac_result"] = "ERROR"

        # 3. Hybrid RSAC
        print("🚀 Running Hybrid RSAC (with preprocessing)...")
        try:
            start_time = time.time()
            hybrid_result, hybrid_stats = hybrid_solver.solve(
                test_case["clauses"], test_case["n_vars"]
            )
            hybrid_time = time.time() - start_time
            hybrid_status = "SAT" if hybrid_result else "UNSAT"
            print(
                f"   Hybrid RSAC: {hybrid_status} in {hybrid_stats['total_checks']:,} checks ({hybrid_time:.3f}s)"
            )
            print(
                f"   Preprocessing eliminated: {hybrid_stats['variables_eliminated']} vars, {hybrid_stats['clauses_eliminated']} clauses"
            )

            result.update(
                {
                    "hybrid_rsac_result": hybrid_status,
                    "hybrid_rsac_checks": hybrid_stats["total_checks"],
                    "hybrid_rsac_time": hybrid_time,
                    "vars_eliminated": hybrid_stats["variables_eliminated"],
                    "clauses_eliminated": hybrid_stats["clauses_eliminated"],
                }
            )
        except Exception as e:
            print(f"   Hybrid RSAC: ERROR - {e}")
            result["hybrid_rsac_result"] = "ERROR"

        # 4. GPU RSAC (if available and feasible)
        if gpu_solver and test_case["n_vars"] <= 18:
            print("⚡ Running GPU-Accelerated RSAC...")
            try:
                start_time = time.time()
                gpu_result = gpu_solver.gpu_bucket_search(
                    test_case["clauses"], test_case["n_vars"], eval_clause
                )
                gpu_time = time.time() - start_time
                gpu_status = "SAT" if gpu_result[0] else "UNSAT"
                print(
                    f"   GPU RSAC: {gpu_status} in {gpu_result[1]:,} checks ({gpu_time:.3f}s)"
                )

                result.update(
                    {
                        "gpu_rsac_result": gpu_status,
                        "gpu_rsac_checks": gpu_result[1],
                        "gpu_rsac_time": gpu_time,
                    }
                )
            except Exception as e:
                print(f"   GPU RSAC: ERROR - {e}")
                result["gpu_rsac_result"] = "ERROR"

        # 5. Advanced Signatures (for smaller instances)
        if test_case["n_vars"] <= 12:
            print("🧠 Testing Advanced Signature Methods...")
            try:
                best_method, best_checks = (
                    signature_generator.find_best_signature_method(
                        test_case["clauses"], test_case["n_vars"], eval_clause
                    )
                )
                print(
                    f"   Best signature method: {best_method} with {best_checks:,} checks"
                )

                result.update(
                    {
                        "best_signature_method": best_method,
                        "best_signature_checks": best_checks,
                    }
                )
            except Exception as e:
                print(f"   Advanced Signatures: ERROR - {e}")
                result["best_signature_method"] = "ERROR"

        # Calculate speedups
        print("\n📊 SPEEDUP ANALYSIS:")
        if "bf_checks" in result and "basic_rsac_checks" in result:
            if result["basic_rsac_checks"] > 0:
                basic_speedup = result["bf_checks"] / result["basic_rsac_checks"]
                result["basic_speedup"] = basic_speedup
                print(f"   Basic RSAC speedup: {basic_speedup:.2f}x")

        if "bf_checks" in result and "hybrid_rsac_checks" in result:
            if result["hybrid_rsac_checks"] > 0:
                hybrid_speedup = result["bf_checks"] / result["hybrid_rsac_checks"]
                result["hybrid_speedup"] = hybrid_speedup
                print(f"   Hybrid RSAC speedup: {hybrid_speedup:.2f}x")

                if hybrid_speedup > 1000:
                    print("   🎉 BREAKTHROUGH: >1000x SPEEDUP ACHIEVED! 🎉")

        if "bf_checks" in result and "best_signature_checks" in result:
            if result["best_signature_checks"] > 0:
                sig_speedup = result["bf_checks"] / result["best_signature_checks"]
                result["signature_speedup"] = sig_speedup
                print(f"   Advanced Signature speedup: {sig_speedup:.2f}x")

        results.append(result)

    return results


def analyze_ultimate_results(results):
    """Analyze ultimate benchmark results."""
    print("\n" + "=" * 100)
    print("🏆 ULTIMATE BENCHMARK ANALYSIS")
    print("=" * 100)

    # Filter successful results
    successful = [r for r in results if "basic_speedup" in r or "hybrid_speedup" in r]

    if not successful:
        print("No successful speedup results to analyze.")
        return

    # Find the best speedups
    max_basic_speedup = 0
    max_hybrid_speedup = 0
    max_signature_speedup = 0

    best_basic = None
    best_hybrid = None
    best_signature = None

    for result in successful:
        if "basic_speedup" in result and result["basic_speedup"] > max_basic_speedup:
            max_basic_speedup = result["basic_speedup"]
            best_basic = result

        if "hybrid_speedup" in result and result["hybrid_speedup"] > max_hybrid_speedup:
            max_hybrid_speedup = result["hybrid_speedup"]
            best_hybrid = result

        if (
            "signature_speedup" in result
            and result["signature_speedup"] > max_signature_speedup
        ):
            max_signature_speedup = result["signature_speedup"]
            best_signature = result

    print(f"\n🥇 RECORD SPEEDUPS:")
    if best_basic:
        print(f"   Basic RSAC: {max_basic_speedup:.2f}x on {best_basic['name']}")
    if best_hybrid:
        print(f"   Hybrid RSAC: {max_hybrid_speedup:.2f}x on {best_hybrid['name']}")
    if best_signature:
        print(
            f"   Advanced Signatures: {max_signature_speedup:.2f}x on {best_signature['name']}"
        )

    # Check for 1000x+ speedups
    thousand_x_cases = [
        r
        for r in successful
        if r.get("hybrid_speedup", 0) > 1000 or r.get("signature_speedup", 0) > 1000
    ]

    if thousand_x_cases:
        print(f"\n🚀 BREAKTHROUGH: {len(thousand_x_cases)} cases with >1000x speedup!")
        for case in thousand_x_cases:
            speedup = max(
                case.get("hybrid_speedup", 0), case.get("signature_speedup", 0)
            )
            print(f"   {case['name']}: {speedup:.2f}x speedup")
    else:
        print(
            f"\n🎯 Highest speedup achieved: {max(max_basic_speedup, max_hybrid_speedup, max_signature_speedup):.2f}x"
        )
        print("   (Still working toward that 1000x goal!)")

    # Summary statistics
    df = pd.DataFrame(successful)

    print(f"\n📈 SUMMARY STATISTICS:")
    if "basic_speedup" in df.columns:
        print(
            f"   Basic RSAC - Mean: {df['basic_speedup'].mean():.2f}x, Max: {df['basic_speedup'].max():.2f}x"
        )
    if "hybrid_speedup" in df.columns:
        print(
            f"   Hybrid RSAC - Mean: {df['hybrid_speedup'].mean():.2f}x, Max: {df['hybrid_speedup'].max():.2f}x"
        )

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Speedup comparison
    methods = []
    speedups = []

    if "basic_speedup" in df.columns:
        methods.extend(["Basic RSAC"] * len(df))
        speedups.extend(df["basic_speedup"].tolist())

    if "hybrid_speedup" in df.columns:
        methods.extend(["Hybrid RSAC"] * len(df))
        speedups.extend(df["hybrid_speedup"].tolist())

    if methods:
        speedup_df = pd.DataFrame({"Method": methods, "Speedup": speedups})
        speedup_df.boxplot(column="Speedup", by="Method", ax=ax1)
        ax1.set_yscale("log")
        ax1.set_title("Speedup Distribution by Method")
        ax1.set_ylabel("Speedup (log scale)")

    # Speedup vs problem size
    if "hybrid_speedup" in df.columns:
        ax2.scatter(df["n_vars"], df["hybrid_speedup"], alpha=0.7)
        ax2.set_xlabel("Number of Variables")
        ax2.set_ylabel("Hybrid RSAC Speedup")
        ax2.set_title("Speedup vs Problem Size")
        ax2.set_yscale("log")

    # Problem type analysis
    if "type" in df.columns and "hybrid_speedup" in df.columns:
        type_speedups = df.groupby("type")["hybrid_speedup"].mean()
        ax3.bar(range(len(type_speedups)), type_speedups.values)
        ax3.set_xticks(range(len(type_speedups)))
        ax3.set_xticklabels(type_speedups.index, rotation=45)
        ax3.set_ylabel("Average Speedup")
        ax3.set_title("Speedup by Problem Type")

    # Preprocessing effectiveness
    if "vars_eliminated" in df.columns and "hybrid_speedup" in df.columns:
        ax4.scatter(df["vars_eliminated"], df["hybrid_speedup"], alpha=0.7)
        ax4.set_xlabel("Variables Eliminated by Preprocessing")
        ax4.set_ylabel("Hybrid RSAC Speedup")
        ax4.set_title("Preprocessing vs Speedup")
        ax4.set_yscale("log")

    plt.tight_layout()
    plt.savefig("ultimate_rsac_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    return df


def main():
    print("🚀 Creating Ultimate Test Suite...")
    test_cases = create_ultimate_test_suite()
    print(f"Generated {len(test_cases)} ultimate test cases")

    print("\n🎯 Running Ultimate Benchmark...")
    results = run_ultimate_benchmark(test_cases, max_vars=18)

    print("\n📊 Analyzing Ultimate Results...")
    df = analyze_ultimate_results(results)

    # Save results
    with open("ultimate_rsac_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if df is not None:
        df.to_csv("ultimate_rsac_results.csv", index=False)

    print(f"\n💾 Results saved to ultimate_rsac_results.json and .csv")
    print("📈 Analysis plot saved as ultimate_rsac_analysis.png")

    # Final status
    max_speedup = 0
    if df is not None and "hybrid_speedup" in df.columns:
        max_speedup = df["hybrid_speedup"].max()

    print(f"\n🏆 FINAL STATUS:")
    print(f"   Maximum speedup achieved: {max_speedup:.2f}x")

    if max_speedup > 1000:
        print("   🎉 MISSION ACCOMPLISHED: >1000x SPEEDUP ACHIEVED!")
        print("   🌍 WORLD-CHANGING STATUS: CONFIRMED!")
    elif max_speedup > 500:
        print("   🚀 EXCELLENT: >500x speedup achieved!")
        print("   🌍 WORLD-CHANGING STATUS: VERY CLOSE!")
    elif max_speedup > 100:
        print("   ✅ GREAT: >100x speedup achieved!")
        print("   🌍 WORLD-CHANGING STATUS: GETTING THERE!")
    else:
        print("   📈 GOOD PROGRESS: Keep optimizing!")
        print("   🌍 WORLD-CHANGING STATUS: IN PROGRESS!")


if __name__ == "__main__":
    main()
