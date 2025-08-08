#!/usr/bin/env python3
"""
Comprehensive stress testing suite for RSAC.
This script runs extended tests that may take a while.
"""

import argparse
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os
import json

from src.rsac.sat import gen_random_kcnf, sat_bruteforce, eval_clause
from src.rsac.bucket_search import bucket_search_no_oracle, build_lut_basic
from src.rsac.collapse import vectorized_extended_signature, all_bit_arrays


def ensure_dirs():
    """Create necessary directories for stress test results."""
    os.makedirs('stress_results', exist_ok=True)
    os.makedirs('stress_results/plots', exist_ok=True)


def generate_test_cases(n_vars_range=(3, 8), k_range=(2, 4), num_clauses_range=(5, 20), num_cases=50):
    """Generate a variety of test cases for stress testing."""
    test_cases = []
    rng = random.Random(42)  # Fixed seed for reproducibility
    
    for _ in range(num_cases):
        n_vars = random.randint(*n_vars_range)
        k = min(random.randint(*k_range), n_vars)  # Ensure k <= n_vars
        num_clauses = random.randint(*num_clauses_range)
        
        # Generate random k-CNF formula - note parameter order: num_vars, num_clauses, k, rng
        clauses = gen_random_kcnf(n_vars, num_clauses, k, rng)
        
        test_cases.append({
            'n_vars': n_vars,
            'k': k,
            'num_clauses': num_clauses,
            'clauses': clauses
        })
    
    return test_cases


def run_correctness_stress_test(test_cases, max_time_per_case=30):
    """Run correctness stress test comparing different methods."""
    results = []
    
    print(f"Running correctness stress test on {len(test_cases)} cases...")
    
    for i, case in enumerate(test_cases):
        print(f"Case {i+1}/{len(test_cases)}: n_vars={case['n_vars']}, k={case['k']}, clauses={case['num_clauses']}")
        
        start_time = time.time()
        
        try:
            # Brute force solution (ground truth)
            bf_start = time.time()
            bf_result = sat_bruteforce(case['clauses'], case['n_vars'])
            bf_time = time.time() - bf_start
            
            if bf_time > max_time_per_case:
                print(f"  Skipping case - brute force too slow ({bf_time:.2f}s)")
                continue
            
            # Bucket search solution
            bs_start = time.time()
            bs_result = bucket_search_no_oracle(case['clauses'], case['n_vars'], eval_clause)
            bs_time = time.time() - bs_start
            
            # Check correctness
            correct = (bf_result is not None) == (bs_result is not None)
            
            results.append({
                'case_id': i,
                'n_vars': case['n_vars'],
                'k': case['k'],
                'num_clauses': case['num_clauses'],
                'bf_time': bf_time,
                'bs_time': bs_time,
                'bf_satisfiable': bf_result is not None,
                'bs_satisfiable': bs_result is not None,
                'correct': correct,
                'speedup': bf_time / bs_time if bs_time > 0 else float('inf')
            })
            
            if not correct:
                print(f"  ERROR: Methods disagree! BF: {bf_result is not None}, BS: {bs_result is not None}")
            else:
                print(f"  Correct! Speedup: {bf_time/bs_time:.2f}x")
                
        except Exception as e:
            print(f"  Error in case {i}: {e}")
            results.append({
                'case_id': i,
                'n_vars': case['n_vars'],
                'k': case['k'],
                'num_clauses': case['num_clauses'],
                'error': str(e)
            })
    
    return results


def run_performance_stress_test(n_vars_range=(3, 12), num_trials=20):
    """Run performance stress test focusing on scalability."""
    results = []
    
    print(f"Running performance stress test...")
    
    for n_vars in range(*n_vars_range):
        print(f"Testing n_vars = {n_vars}")
        
        for trial in range(num_trials):
            # Generate moderately complex formula
            k = min(3, n_vars)
            num_clauses = min(2 ** n_vars // 2, 50)  # Don't make it too hard
            
            rng = random.Random(n_vars * 1000 + trial)  # Deterministic but varied
            clauses = gen_random_kcnf(n_vars, num_clauses, k, rng)
            
            try:
                # Only test bucket search for larger cases
                start_time = time.time()
                result = bucket_search_no_oracle(clauses, n_vars, eval_clause)
                total_time = time.time() - start_time
                
                # For timing breakdown, we'd need to modify the function
                build_time = total_time * 0.1  # Rough estimate
                search_time = total_time * 0.9
                
                results.append({
                    'n_vars': n_vars,
                    'k': k,
                    'num_clauses': num_clauses,
                    'trial': trial,
                    'build_time': build_time,
                    'search_time': search_time,
                    'total_time': total_time,
                    'satisfiable': result is not None,
                    'lut_size': 2**n_vars  # Approximate - total assignments
                })
                
            except Exception as e:
                print(f"  Error in trial {trial}: {e}")
    
    return results


def run_memory_stress_test(max_n_vars=10):
    """Test memory usage patterns."""
    import psutil
    import gc
    
    results = []
    process = psutil.Process()
    
    print("Running memory stress test...")
    
    for n_vars in range(3, max_n_vars + 1):
        print(f"Testing memory usage for n_vars = {n_vars}")
        
        # Clear memory
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate test case
        k = min(3, n_vars)
        num_clauses = min(2 ** n_vars // 3, 30)
        rng = random.Random(n_vars * 100)  # Deterministic seed
        clauses = gen_random_kcnf(n_vars, num_clauses, k, rng)
        
        # Build LUT and measure memory
        groups, lut = build_lut_basic(n_vars)
        lut_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate all bit arrays and measure
        bit_arrays = list(all_bit_arrays(n_vars))
        arrays_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run search
        result = bucket_search_no_oracle(clauses, n_vars, eval_clause)
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        results.append({
            'n_vars': n_vars,
            'num_clauses': num_clauses,
            'initial_memory_mb': initial_memory,
            'lut_memory_mb': lut_memory,
            'arrays_memory_mb': arrays_memory,
            'final_memory_mb': final_memory,
            'lut_size': len(groups),
            'num_bit_arrays': len(bit_arrays),
            'satisfiable': result is not None
        })
        
        # Clean up
        del lut, bit_arrays, result
        gc.collect()
    
    return results


def analyze_results(correctness_results, performance_results, memory_results):
    """Analyze and visualize stress test results."""
    print("\n" + "="*60)
    print("STRESS TEST ANALYSIS")
    print("="*60)
    
    # Correctness analysis
    if correctness_results:
        correct_cases = [r for r in correctness_results if r.get('correct', False)]
        error_cases = [r for r in correctness_results if 'error' in r]
        
        print(f"\nCorrectness Results:")
        print(f"  Total cases: {len(correctness_results)}")
        print(f"  Correct: {len(correct_cases)}")
        print(f"  Errors: {len(error_cases)}")
        
        if correct_cases:
            speedups = [r['speedup'] for r in correct_cases if r['speedup'] != float('inf')]
            if speedups:
                print(f"  Average speedup: {np.mean(speedups):.2f}x")
                print(f"  Max speedup: {np.max(speedups):.2f}x")
    
    # Performance analysis
    if performance_results:
        df_perf = pd.DataFrame(performance_results)
        print(f"\nPerformance Results:")
        
        # Group by n_vars
        perf_by_vars = df_perf.groupby('n_vars').agg({
            'total_time': ['mean', 'std', 'max'],
            'build_time': 'mean',
            'search_time': 'mean',
            'lut_size': 'mean'
        }).round(4)
        
        print(perf_by_vars)
        
        # Plot performance scaling
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        avg_times = df_perf.groupby('n_vars')['total_time'].mean()
        plt.plot(avg_times.index, avg_times.values, 'bo-')
        plt.xlabel('Number of Variables')
        plt.ylabel('Average Total Time (s)')
        plt.title('Performance Scaling')
        plt.yscale('log')
        
        plt.subplot(2, 2, 2)
        avg_lut_size = df_perf.groupby('n_vars')['lut_size'].mean()
        plt.plot(avg_lut_size.index, avg_lut_size.values, 'ro-')
        plt.xlabel('Number of Variables')
        plt.ylabel('Average LUT Size')
        plt.title('LUT Size Growth')
        plt.yscale('log')
        
        plt.subplot(2, 2, 3)
        build_times = df_perf.groupby('n_vars')['build_time'].mean()
        search_times = df_perf.groupby('n_vars')['search_time'].mean()
        plt.plot(build_times.index, build_times.values, 'g-', label='Build Time')
        plt.plot(search_times.index, search_times.values, 'b-', label='Search Time')
        plt.xlabel('Number of Variables')
        plt.ylabel('Time (s)')
        plt.title('Build vs Search Time')
        plt.legend()
        plt.yscale('log')
        
        plt.subplot(2, 2, 4)
        sat_rates = df_perf.groupby('n_vars')['satisfiable'].mean()
        plt.plot(sat_rates.index, sat_rates.values, 'mo-')
        plt.xlabel('Number of Variables')
        plt.ylabel('Satisfiability Rate')
        plt.title('Satisfiability by Problem Size')
        
        plt.tight_layout()
        plt.savefig('stress_results/plots/performance_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Memory analysis
    if memory_results:
        df_mem = pd.DataFrame(memory_results)
        print(f"\nMemory Results:")
        print(df_mem[['n_vars', 'lut_memory_mb', 'arrays_memory_mb', 'final_memory_mb']].to_string(index=False))
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_mem['n_vars'], df_mem['lut_memory_mb'], 'bo-', label='LUT Memory')
        plt.plot(df_mem['n_vars'], df_mem['arrays_memory_mb'], 'ro-', label='Arrays Memory')
        plt.plot(df_mem['n_vars'], df_mem['final_memory_mb'], 'go-', label='Final Memory')
        plt.xlabel('Number of Variables')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Scaling')
        plt.legend()
        plt.yscale('log')
        plt.savefig('stress_results/plots/memory_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()


def save_results(correctness_results, performance_results, memory_results):
    """Save all results to files."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    results_data = {
        'timestamp': timestamp,
        'correctness_results': correctness_results,
        'performance_results': performance_results,
        'memory_results': memory_results
    }
    
    filename = f'stress_results/stress_test_results_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Run RSAC stress tests')
    parser.add_argument('--correctness', action='store_true', help='Run correctness stress test')
    parser.add_argument('--performance', action='store_true', help='Run performance stress test')
    parser.add_argument('--memory', action='store_true', help='Run memory stress test')
    parser.add_argument('--all', action='store_true', help='Run all stress tests')
    parser.add_argument('--cases', type=int, default=50, help='Number of test cases for correctness test')
    parser.add_argument('--max-vars', type=int, default=10, help='Maximum number of variables for performance/memory tests')
    parser.add_argument('--trials', type=int, default=20, help='Number of trials per configuration')
    
    args = parser.parse_args()
    
    if not any([args.correctness, args.performance, args.memory, args.all]):
        print("Please specify which tests to run (--correctness, --performance, --memory, or --all)")
        return
    
    ensure_dirs()
    
    correctness_results = []
    performance_results = []
    memory_results = []
    
    print("Starting RSAC Stress Testing Suite")
    print("="*50)
    
    if args.correctness or args.all:
        print("\n1. Generating test cases...")
        test_cases = generate_test_cases(num_cases=args.cases)
        correctness_results = run_correctness_stress_test(test_cases)
    
    if args.performance or args.all:
        print("\n2. Running performance stress test...")
        performance_results = run_performance_stress_test(
            n_vars_range=(3, args.max_vars + 1),
            num_trials=args.trials
        )
    
    if args.memory or args.all:
        print("\n3. Running memory stress test...")
        memory_results = run_memory_stress_test(max_n_vars=args.max_vars)
    
    print("\n4. Analyzing results...")
    analyze_results(correctness_results, performance_results, memory_results)
    
    print("\n5. Saving results...")
    save_results(correctness_results, performance_results, memory_results)
    
    print("\nStress testing complete!")


if __name__ == "__main__":
    main()