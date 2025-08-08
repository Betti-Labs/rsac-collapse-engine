#!/usr/bin/env python3
"""
Advanced signature variants for RSAC - multiple signature types for maximum effectiveness.
This is where we get creative and push beyond the basic digital-root approach!
"""

import numpy as np
from typing import List, Tuple, Dict, Callable
import hashlib
from collections import defaultdict

from .collapse import digital_root, symbolic_reduction_loop
from .utils import bits_to_seq


class AdvancedSignatureGenerator:
    """Generate multiple types of signatures and pick the best one."""

    def __init__(self):
        self.signature_methods = {
            "digital_root": self._digital_root_signature,
            "fibonacci": self._fibonacci_signature,
            "prime_hash": self._prime_hash_signature,
            "polynomial": self._polynomial_signature,
            "chaos_map": self._chaos_map_signature,
            "cellular_automata": self._cellular_automata_signature,
            "fractal": self._fractal_signature,
        }

    def generate_all_signature_variants(self, n_vars: int) -> Dict[str, List[Tuple]]:
        """Generate all signature variants and return as buckets."""
        all_assignments = []
        for bits in range(1 << n_vars):
            assignment = tuple((bits >> i) & 1 for i in range(n_vars))
            all_assignments.append(assignment)

        signature_buckets = {}

        for method_name, method_func in self.signature_methods.items():
            print(f"Generating {method_name} signatures...")
            buckets = defaultdict(list)

            for assignment in all_assignments:
                signature = method_func(assignment)
                buckets[signature].append(assignment)

            # Sort by bucket size
            sorted_buckets = sorted(buckets.items(), key=lambda x: len(x[1]))
            signature_buckets[method_name] = sorted_buckets

            print(
                f"  {method_name}: {len(sorted_buckets)} buckets, "
                f"smallest={len(sorted_buckets[0][1])}, largest={len(sorted_buckets[-1][1])}"
            )

        return signature_buckets

    def _digital_root_signature(self, assignment: Tuple[int]) -> Tuple:
        """Original RSAC digital-root signature."""
        seq = bits_to_seq(assignment)
        hist = symbolic_reduction_loop(seq, depth=16)
        L = len(hist)

        final_layer = tuple(hist[-1]) if L >= 1 else ()
        penultimate = tuple(hist[-2]) if L >= 2 else ()
        antepenultimate = tuple(hist[-3]) if L >= 3 else ()
        ent_tail = (
            tuple(len(set(stage)) for stage in hist[-5:])
            if L >= 5
            else tuple(len(set(stage)) for stage in hist)
        )

        return (final_layer, penultimate, antepenultimate, ent_tail)

    def _fibonacci_signature(self, assignment: Tuple[int]) -> Tuple:
        """Fibonacci-based signature generation."""
        n = len(assignment)

        # Generate Fibonacci sequence up to n
        fib = [1, 1]
        while len(fib) < n:
            fib.append(fib[-1] + fib[-2])

        # Weight assignment by Fibonacci numbers
        weighted_sum = sum(assignment[i] * fib[i] for i in range(n))

        # Create signature layers
        layers = []
        current = list(assignment)

        while len(current) > 1:
            next_layer = []
            for i in range(len(current) - 1):
                # Fibonacci reduction rule
                val = (
                    current[i] * fib[i % len(fib)]
                    + current[i + 1] * fib[(i + 1) % len(fib)]
                ) % 13
                next_layer.append(val)
            layers.append(tuple(current))
            current = next_layer

        layers.append(tuple(current))

        # Extract signature components
        final = layers[-1] if layers else ()
        penult = layers[-2] if len(layers) >= 2 else ()

        return (final, penult, weighted_sum % 1000)

    def _prime_hash_signature(self, assignment: Tuple[int]) -> Tuple:
        """Prime-number-based hash signature."""
        primes = [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
        ]

        # Create prime-weighted hash
        hash_val = 1
        for i, bit in enumerate(assignment):
            if bit == 1:
                hash_val = (hash_val * primes[i % len(primes)]) % 10007

        # Create secondary hash with different primes
        hash2 = (
            sum(
                assignment[i] * primes[(i * 3) % len(primes)]
                for i in range(len(assignment))
            )
            % 997
        )

        # Bit pattern analysis
        run_lengths = []
        current_run = 1
        for i in range(1, len(assignment)):
            if assignment[i] == assignment[i - 1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)

        return (hash_val, hash2, tuple(run_lengths[:5]))  # Limit run_lengths for memory

    def _polynomial_signature(self, assignment: Tuple[int]) -> Tuple:
        """Polynomial evaluation signature."""
        n = len(assignment)

        # Evaluate polynomial p(x) = sum(a_i * x^i) at multiple points
        evaluations = []
        for x in [2, 3, 5, 7]:  # Evaluate at small primes
            val = sum(assignment[i] * (x**i) for i in range(n)) % 1009
            evaluations.append(val)

        # Derivative information
        derivative_at_1 = sum(i * assignment[i] for i in range(1, n)) % 101

        # Coefficient patterns
        even_sum = sum(assignment[i] for i in range(0, n, 2))
        odd_sum = sum(assignment[i] for i in range(1, n, 2))

        return (tuple(evaluations), derivative_at_1, even_sum, odd_sum)

    def _chaos_map_signature(self, assignment: Tuple[int]) -> Tuple:
        """Chaotic map-based signature."""
        # Logistic map: x_{n+1} = r * x_n * (1 - x_n)
        r = 3.9  # Chaotic parameter

        # Initialize with assignment-based seed
        x = sum(assignment[i] * (0.1 ** (i + 1)) for i in range(len(assignment)))
        x = x - int(x)  # Keep fractional part
        if x == 0:
            x = 0.5

        # Iterate chaotic map
        trajectory = []
        for _ in range(10):
            x = r * x * (1 - x)
            trajectory.append(int(x * 1000) % 100)

        # Analyze trajectory
        mean_val = int(np.mean(trajectory))
        std_val = int(np.std(trajectory) * 10)

        # Lyapunov exponent approximation
        lyap = sum(np.log(abs(r * (1 - 2 * (t / 1000)))) for t in trajectory[:5]) / 5
        lyap_int = int(abs(lyap) * 100) % 1000

        return (tuple(trajectory[:5]), mean_val, std_val, lyap_int)

    def _cellular_automata_signature(self, assignment: Tuple[int]) -> Tuple:
        """Cellular automata evolution signature."""
        # Rule 30 (chaotic rule)
        rule = 30

        # Pad assignment to avoid boundary effects
        state = list(assignment) + [0] * 4
        n = len(state)

        generations = []
        for gen in range(5):  # Evolve for 5 generations
            generations.append(tuple(state))
            new_state = [0] * n

            for i in range(1, n - 1):
                # Rule 30: look at left, center, right
                pattern = (state[i - 1] << 2) | (state[i] << 1) | state[i + 1]
                new_state[i] = (rule >> pattern) & 1

            state = new_state

        # Extract features
        final_gen = generations[-1]
        density = sum(final_gen) / len(final_gen)

        # Pattern analysis
        patterns = []
        for gen in generations:
            pattern_count = 0
            for i in range(len(gen) - 2):
                if gen[i : i + 3] == (1, 0, 1):  # Count specific pattern
                    pattern_count += 1
            patterns.append(pattern_count)

        return (final_gen[:8], int(density * 100), tuple(patterns))

    def _fractal_signature(self, assignment: Tuple[int]) -> Tuple:
        """Fractal dimension-based signature."""
        n = len(assignment)

        # Convert to complex number
        real_part = sum(assignment[i] * (0.5**i) for i in range(0, n, 2))
        imag_part = sum(assignment[i] * (0.5**i) for i in range(1, n, 2))
        c = complex(real_part - 0.5, imag_part - 0.5)

        # Mandelbrot-like iteration
        z = 0
        iterations = []
        for i in range(10):
            z = z * z + c
            iterations.append(int(abs(z) * 100) % 1000)
            if abs(z) > 2:
                break

        # Julia set-like iteration with different c
        z2 = complex(real_part, imag_part)
        c2 = complex(-0.7, 0.27015)  # Interesting Julia set parameter
        julia_iterations = []
        for i in range(5):
            z2 = z2 * z2 + c2
            julia_iterations.append(int(abs(z2) * 100) % 1000)
            if abs(z2) > 2:
                break

        # Fractal dimension approximation (box counting)
        # Simplified version for signature
        box_count = len(set(iterations + julia_iterations))

        return (tuple(iterations[:5]), tuple(julia_iterations), box_count)

    def find_best_signature_method(
        self, clauses: List[List[int]], n_vars: int, eval_fn: Callable
    ) -> Tuple[str, int]:
        """Find which signature method gives the best performance on this instance."""
        if n_vars > 16:  # Too expensive for large problems
            return "digital_root", float("inf")

        all_signatures = self.generate_all_signature_variants(n_vars)

        best_method = None
        best_checks = float("inf")

        for method_name, buckets in all_signatures.items():
            print(f"\nTesting {method_name} signature...")
            checks = 0
            found = False

            for signature, assignments in buckets:
                for assignment in assignments:
                    checks += 1
                    if all(eval_fn(clause, assignment) for clause in clauses):
                        print(f"  Found solution in {checks} checks")
                        found = True
                        break
                if found:
                    break

            if not found:
                print(f"  No solution found after {checks} checks")

            if checks < best_checks:
                best_checks = checks
                best_method = method_name

        print(f"\nBest method: {best_method} with {best_checks} checks")
        return best_method, best_checks


def benchmark_signature_variants(test_cases, max_vars=14):
    """Benchmark different signature variants."""
    generator = AdvancedSignatureGenerator()
    results = []

    print("=" * 80)
    print("ADVANCED SIGNATURE VARIANTS BENCHMARK")
    print("=" * 80)

    for test_case in test_cases[:5]:  # Limit to first 5 for time
        if test_case["n_vars"] > max_vars:
            continue

        print(f"\n{'='*60}")
        print(f"Testing: {test_case['name']} (n={test_case['n_vars']})")
        print(f"{'='*60}")

        from .sat import eval_clause

        best_method, best_checks = generator.find_best_signature_method(
            test_case["clauses"], test_case["n_vars"], eval_clause
        )

        results.append(
            {
                "name": test_case["name"],
                "n_vars": test_case["n_vars"],
                "best_method": best_method,
                "best_checks": best_checks,
            }
        )

    return results


if __name__ == "__main__":
    # Quick test
    generator = AdvancedSignatureGenerator()

    # Test on small assignment
    assignment = (1, 0, 1, 0)

    print("Testing signature variants on assignment (1,0,1,0):")
    for method_name, method_func in generator.signature_methods.items():
        signature = method_func(assignment)
        print(f"{method_name}: {signature}")
