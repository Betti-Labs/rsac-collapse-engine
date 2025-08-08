"""Performance and stress tests for RSAC."""

import unittest
import time
import random
import numpy as np
from collections import Counter
from src.rsac.sat import gen_random_kcnf, sat_bruteforce, eval_clause
from src.rsac.bucket_search import bucket_search_no_oracle, build_lut_basic, get_lut_for_m
from src.rsac.collapse import vectorized_extended_signature, all_bit_arrays


class TestPerformance(unittest.TestCase):
    def test_scaling_behavior(self):
        """Test how performance scales with problem size."""
        results = []
        
        for n in [6, 8, 10, 12]:
            rng = random.Random(42)
            clauses = gen_random_kcnf(n, int(1.5 * n), 3, rng)
            
            # Time brute force
            start = time.time()
            bf_solution, bf_checks = sat_bruteforce(clauses, n)
            bf_time = time.time() - start
            
            # Time RSAC
            start = time.time()
            rsac_solution, rsac_checks, key = bucket_search_no_oracle(clauses, n, eval_clause)
            rsac_time = time.time() - start
            
            results.append({
                'n': n,
                'bf_checks': bf_checks,
                'rsac_checks': rsac_checks,
                'bf_time': bf_time,
                'rsac_time': rsac_time,
                'speedup_checks': bf_checks / rsac_checks if rsac_checks > 0 else float('inf'),
                'speedup_time': bf_time / rsac_time if rsac_time > 0 else float('inf')
            })
        
        # Print results for analysis
        print("\nScaling Results:")
        print("n\tBF_checks\tRSAC_checks\tSpeedup_checks\tBF_time\tRSAC_time\tSpeedup_time")
        for r in results:
            print(f"{r['n']}\t{r['bf_checks']}\t{r['rsac_checks']}\t{r['speedup_checks']:.2f}\t"
                  f"{r['bf_time']:.4f}\t{r['rsac_time']:.4f}\t{r['speedup_time']:.2f}")
        
        # Basic sanity checks
        for r in results:
            self.assertGreater(r['speedup_checks'], 1.0, f"No speedup for n={r['n']}")
    
    def test_vectorized_performance(self):
        """Test performance of vectorized vs Python signature generation."""
        n = 12
        bits_mat = all_bit_arrays(n)
        
        # Time vectorized version
        start = time.time()
        vec_sigs = vectorized_extended_signature(bits_mat)
        vec_time = time.time() - start
        
        # Time Python version (sample only to avoid timeout)
        from src.rsac.collapse import extended_signature_from_seq
        from src.rsac.utils import bits_to_seq
        
        sample_size = min(1000, len(bits_mat))
        start = time.time()
        for i in range(sample_size):
            seq = bits_to_seq(tuple(bits_mat[i]))
            extended_signature_from_seq(seq)
        py_time = time.time() - start
        
        # Extrapolate Python time
        py_time_full = py_time * (len(bits_mat) / sample_size)
        
        print(f"\nVectorized signature generation:")
        print(f"Vectorized time: {vec_time:.4f}s")
        print(f"Python time (estimated): {py_time_full:.4f}s")
        print(f"Speedup: {py_time_full / vec_time:.2f}x")
        
        self.assertLess(vec_time, py_time_full, "Vectorized should be faster")


class TestStress(unittest.TestCase):
    def test_random_seed_variation(self):
        """Test RSAC with many different random seeds."""
        n = 8
        num_seeds = 50
        speedups = []
        
        for seed in range(num_seeds):
            rng = random.Random(seed)
            clauses = gen_random_kcnf(n, n + 3, 3, rng)
            
            bf_solution, bf_checks = sat_bruteforce(clauses, n)
            rsac_solution, rsac_checks, key = bucket_search_no_oracle(clauses, n, eval_clause)
            
            # Verify correctness
            if bf_solution is None:
                self.assertIsNone(rsac_solution)
            else:
                self.assertIsNotNone(rsac_solution)
                for clause in clauses:
                    self.assertEqual(eval_clause(clause, rsac_solution), 1)
            
            if rsac_checks > 0:
                speedups.append(bf_checks / rsac_checks)
        
        # Analyze speedup distribution
        speedups = np.array(speedups)
        print(f"\nSpeedup statistics over {num_seeds} seeds:")
        print(f"Mean: {np.mean(speedups):.2f}")
        print(f"Median: {np.median(speedups):.2f}")
        print(f"Min: {np.min(speedups):.2f}")
        print(f"Max: {np.max(speedups):.2f}")
        print(f"Std: {np.std(speedups):.2f}")
        
        # Most instances should show speedup
        good_speedups = np.sum(speedups > 1.0)
        self.assertGreater(good_speedups / len(speedups), 0.7, 
                          "At least 70% of instances should show speedup")
    
    def test_bucket_distribution_analysis(self):
        """Analyze bucket size distributions."""
        for n in [8, 10, 12]:
            groups, ordered = build_lut_basic(n)
            bucket_sizes = [len(bucket) for bucket in groups.values()]
            
            print(f"\nBucket analysis for n={n}:")
            print(f"Total buckets: {len(bucket_sizes)}")
            print(f"Total assignments: {sum(bucket_sizes)}")
            print(f"Expected total: {2**n}")
            
            # Size distribution
            size_counts = Counter(bucket_sizes)
            print("Bucket size distribution:")
            for size in sorted(size_counts.keys())[:10]:  # Show first 10
                count = size_counts[size]
                print(f"  Size {size}: {count} buckets")
            
            # Statistics
            bucket_sizes = np.array(bucket_sizes)
            print(f"Mean bucket size: {np.mean(bucket_sizes):.2f}")
            print(f"Median bucket size: {np.median(bucket_sizes):.2f}")
            print(f"Max bucket size: {np.max(bucket_sizes)}")
            print(f"Min bucket size: {np.min(bucket_sizes)}")
            
            # Check for extreme skew (many tiny buckets)
            tiny_buckets = np.sum(bucket_sizes <= 2)
            tiny_fraction = tiny_buckets / len(bucket_sizes)
            print(f"Fraction of buckets with size ≤ 2: {tiny_fraction:.3f}")
            
            self.assertGreater(tiny_fraction, 0.5, 
                             f"Should have many tiny buckets for n={n}")
    
    def test_memory_usage(self):
        """Test memory usage for larger instances."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        for n in [10, 12, 14]:
            # Measure memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Build LUT
            groups, ordered = build_lut_basic(n)
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            
            print(f"\nMemory usage for n={n}:")
            print(f"Memory used: {mem_used:.1f} MB")
            print(f"Buckets created: {len(groups)}")
            print(f"Memory per bucket: {mem_used * 1024 / len(groups):.2f} KB")
            
            # Clean up
            del groups, ordered
    
    def test_hard_instances(self):
        """Test on potentially hard SAT instances."""
        # Generate instances at the SAT/UNSAT phase transition
        n = 10
        critical_ratio = 4.26  # Approximate critical ratio for 3-SAT
        num_clauses = int(critical_ratio * n)
        
        hard_speedups = []
        
        for seed in range(20):
            rng = random.Random(1000 + seed)
            clauses = gen_random_kcnf(n, num_clauses, 3, rng)
            
            bf_solution, bf_checks = sat_bruteforce(clauses, n)
            rsac_solution, rsac_checks, key = bucket_search_no_oracle(clauses, n, eval_clause)
            
            # Verify correctness
            if bf_solution is None:
                self.assertIsNone(rsac_solution)
            else:
                self.assertIsNotNone(rsac_solution)
            
            if rsac_checks > 0:
                speedup = bf_checks / rsac_checks
                hard_speedups.append(speedup)
        
        if hard_speedups:
            print(f"\nHard instance speedups (n={n}, {num_clauses} clauses):")
            print(f"Mean speedup: {np.mean(hard_speedups):.2f}")
            print(f"Min speedup: {np.min(hard_speedups):.2f}")
            
            # Even hard instances should show some speedup
            self.assertGreater(np.mean(hard_speedups), 1.0, 
                             "Hard instances should still show average speedup")


if __name__ == '__main__':
    unittest.main(verbosity=2)