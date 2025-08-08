#!/usr/bin/env python3
"""
Hybrid RSAC solver combining preprocessing, unit propagation, and signature-based search.
This is where we push for 1000x+ speedups!
"""

import time
from typing import Optional, Tuple, Dict, List
import numpy as np

from .sat import unit_propagate, pure_literal_elim, simplify_with_assignment, eval_clause
from .bucket_search import bucket_search_no_oracle, rsac_up_vectorized_search


class HybridRSACSolver:
    """Advanced RSAC solver with multiple optimization stages."""
    
    def __init__(self, max_vars_for_rsac=18, enable_preprocessing=True, enable_learning=True):
        self.max_vars_for_rsac = max_vars_for_rsac
        self.enable_preprocessing = enable_preprocessing
        self.enable_learning = enable_learning
        self.stats = {
            'preprocessing_time': 0,
            'rsac_time': 0,
            'total_checks': 0,
            'variables_eliminated': 0,
            'clauses_eliminated': 0
        }
    
    def solve(self, clauses: List[List[int]], n_vars: int) -> Tuple[Optional[Dict], Dict]:
        """
        Solve SAT instance using hybrid approach.
        Returns (assignment, stats) where assignment is None if UNSAT.
        """
        start_time = time.time()
        self.stats = {k: 0 for k in self.stats}
        
        original_clauses = len(clauses)
        
        # Stage 1: Preprocessing
        if self.enable_preprocessing:
            clauses, assignment, n_vars = self._preprocess(clauses, n_vars)
            if clauses is None:
                return None, self.stats  # UNSAT during preprocessing
            if not clauses:  # All clauses satisfied
                return assignment, self.stats
        else:
            assignment = {}
        
        self.stats['preprocessing_time'] = time.time() - start_time
        self.stats['variables_eliminated'] = n_vars - len([v for v in range(1, n_vars+1) if v not in assignment])
        self.stats['clauses_eliminated'] = original_clauses - len(clauses)
        
        # Stage 2: RSAC Search
        rsac_start = time.time()
        
        remaining_vars = [v for v in range(1, n_vars+1) if v not in assignment]
        
        if len(remaining_vars) <= self.max_vars_for_rsac:
            # Use RSAC on remaining variables
            if len(remaining_vars) == 0:
                result = assignment, 0, None
            else:
                result = rsac_up_vectorized_search(clauses, assignment, remaining_vars)
            
            if result[0] is not None:
                final_assignment = result[0]
                self.stats['total_checks'] = result[1]
            else:
                final_assignment = None
                self.stats['total_checks'] = result[1]
        else:
            # Fall back to basic RSAC if still manageable
            if len(remaining_vars) <= 20:
                # Reconstruct clauses for remaining variables
                var_map = {v: i for i, v in enumerate(remaining_vars, 1)}
                mapped_clauses = []
                
                for clause in clauses:
                    mapped_clause = []
                    satisfied = False
                    
                    for lit in clause:
                        var = abs(lit)
                        if var in assignment:
                            val = assignment[var]
                            lit_val = val if lit > 0 else 1 - val
                            if lit_val == 1:
                                satisfied = True
                                break
                        else:
                            mapped_clause.append(lit if lit > 0 else -var_map[var] if var in var_map else lit)
                    
                    if not satisfied and mapped_clause:
                        mapped_clauses.append(mapped_clause)
                
                result = bucket_search_no_oracle(mapped_clauses, len(remaining_vars), eval_clause)
                
                if result[0] is not None:
                    # Map back to original variables
                    final_assignment = dict(assignment)
                    for i, val in enumerate(result[0]):
                        final_assignment[remaining_vars[i]] = val
                    self.stats['total_checks'] = result[1]
                else:
                    final_assignment = None
                    self.stats['total_checks'] = result[1]
            else:
                # Problem too large for RSAC
                final_assignment = None
                self.stats['total_checks'] = 2 ** len(remaining_vars)  # Would need full search
        
        self.stats['rsac_time'] = time.time() - rsac_start
        
        return final_assignment, self.stats
    
    def _preprocess(self, clauses: List[List[int]], n_vars: int) -> Tuple[Optional[List], Dict, int]:
        """Apply preprocessing techniques."""
        assignment = {}
        
        # Unit propagation
        clauses, unit_assignment = unit_propagate(clauses)
        if clauses is None:
            return None, None, n_vars  # UNSAT
        
        if unit_assignment:
            assignment.update(unit_assignment)
        
        # Pure literal elimination
        clauses, assignment = pure_literal_elim(clauses, assignment)
        if clauses is None:
            return None, None, n_vars  # UNSAT
        
        # Subsumption elimination (remove subsumed clauses)
        clauses = self._eliminate_subsumption(clauses)
        
        # Self-subsumption resolution
        clauses = self._self_subsumption(clauses)
        
        return clauses, assignment, n_vars
    
    def _eliminate_subsumption(self, clauses: List[List[int]]) -> List[List[int]]:
        """Remove clauses that are subsumed by others."""
        if not clauses:
            return clauses
        
        # Sort by length for efficiency
        sorted_clauses = sorted(clauses, key=len)
        result = []
        
        for i, clause1 in enumerate(sorted_clauses):
            subsumed = False
            clause1_set = set(clause1)
            
            for j in range(i):
                clause2_set = set(sorted_clauses[j])
                if clause2_set.issubset(clause1_set):
                    subsumed = True
                    break
            
            if not subsumed:
                result.append(clause1)
        
        return result
    
    def _self_subsumption(self, clauses: List[List[int]]) -> List[List[int]]:
        """Apply self-subsumption resolution."""
        if not clauses:
            return clauses
        
        changed = True
        while changed:
            changed = False
            new_clauses = []
            
            for i, clause1 in enumerate(clauses):
                modified = False
                new_clause = clause1[:]
                
                for j, clause2 in enumerate(clauses):
                    if i == j:
                        continue
                    
                    # Check if clause2 can resolve with clause1
                    for lit in clause2:
                        if -lit in new_clause and len(clause2) < len(new_clause):
                            # Remove -lit from new_clause
                            new_clause = [l for l in new_clause if l != -lit]
                            modified = True
                            changed = True
                            break
                
                new_clauses.append(new_clause if modified else clause1)
            
            clauses = new_clauses
        
        return clauses


def benchmark_hybrid_solver(test_cases, max_vars=20):
    """Benchmark the hybrid solver against basic RSAC."""
    results = []
    solver = HybridRSACSolver(max_vars_for_rsac=max_vars)
    
    print("="*80)
    print("HYBRID RSAC SOLVER BENCHMARK")
    print("="*80)
    
    for i, test_case in enumerate(test_cases):
        if test_case['n_vars'] > max_vars:
            continue
            
        print(f"\n[{i+1}] Testing: {test_case['name']}")
        print(f"  Variables: {test_case['n_vars']}, Clauses: {len(test_case['clauses'])}")
        
        # Test basic RSAC
        start_time = time.time()
        basic_result = bucket_search_no_oracle(test_case['clauses'], test_case['n_vars'], eval_clause)
        basic_time = time.time() - start_time
        basic_status = 'SAT' if basic_result[0] else 'UNSAT'
        
        # Test hybrid RSAC
        start_time = time.time()
        hybrid_result, hybrid_stats = solver.solve(test_case['clauses'], test_case['n_vars'])
        hybrid_time = time.time() - start_time
        hybrid_status = 'SAT' if hybrid_result else 'UNSAT'
        
        # Calculate improvements
        speedup = basic_result[1] / hybrid_stats['total_checks'] if hybrid_stats['total_checks'] > 0 else float('inf')
        time_speedup = basic_time / hybrid_time if hybrid_time > 0 else float('inf')
        
        print(f"  Basic RSAC: {basic_status} in {basic_result[1]} checks ({basic_time:.3f}s)")
        print(f"  Hybrid RSAC: {hybrid_status} in {hybrid_stats['total_checks']} checks ({hybrid_time:.3f}s)")
        print(f"  Preprocessing eliminated: {hybrid_stats['variables_eliminated']} vars, {hybrid_stats['clauses_eliminated']} clauses")
        print(f"  Check speedup: {speedup:.2f}x, Time speedup: {time_speedup:.2f}x")
        
        results.append({
            'name': test_case['name'],
            'n_vars': test_case['n_vars'],
            'n_clauses': len(test_case['clauses']),
            'basic_checks': basic_result[1],
            'basic_time': basic_time,
            'hybrid_checks': hybrid_stats['total_checks'],
            'hybrid_time': hybrid_time,
            'vars_eliminated': hybrid_stats['variables_eliminated'],
            'clauses_eliminated': hybrid_stats['clauses_eliminated'],
            'check_speedup': speedup,
            'time_speedup': time_speedup,
            'correctness': 'CORRECT' if basic_status == hybrid_status else 'INCORRECT'
        })
    
    return results


if __name__ == '__main__':
    # Quick test
    from ..sat import gen_random_kcnf
    import random
    
    # Generate test case
    rng = random.Random(42)
    clauses = gen_random_kcnf(12, 36, 3, rng)
    
    solver = HybridRSACSolver()
    result, stats = solver.solve(clauses, 12)
    
    print(f"Result: {'SAT' if result else 'UNSAT'}")
    print(f"Stats: {stats}")