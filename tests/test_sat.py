"""SAT-specific tests for RSAC."""

import unittest
import random
from src.rsac.sat import (
    gen_random_kcnf, eval_clause, sat_bruteforce,
    unit_propagate, pure_literal_elim, simplify_with_assignment
)
from src.rsac.bucket_search import bucket_search_no_oracle, rsac_up_vectorized_search


class TestSATGeneration(unittest.TestCase):
    def test_gen_random_kcnf_structure(self):
        """Test that generated CNF has correct structure."""
        rng = random.Random(42)
        clauses = gen_random_kcnf(5, 10, 3, rng)
        
        self.assertEqual(len(clauses), 10)
        for clause in clauses:
            self.assertEqual(len(clause), 3)
            for lit in clause:
                self.assertIsInstance(lit, int)
                self.assertNotEqual(lit, 0)
                self.assertTrue(1 <= abs(lit) <= 5)
    
    def test_reproducible_generation(self):
        """Test that same seed produces same CNF."""
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        
        cnf1 = gen_random_kcnf(4, 6, 3, rng1)
        cnf2 = gen_random_kcnf(4, 6, 3, rng2)
        
        self.assertEqual(cnf1, cnf2)


class TestClauseEvaluation(unittest.TestCase):
    def test_eval_clause_basic(self):
        """Test basic clause evaluation."""
        # Clause: x1 OR NOT x2 OR x3
        clause = [1, -2, 3]
        
        # Assignment: x1=1, x2=0, x3=0 -> should satisfy (x1=1)
        bits = (1, 0, 0)
        self.assertEqual(eval_clause(clause, bits), 1)
        
        # Assignment: x1=0, x2=1, x3=0 -> should satisfy (NOT x2 = NOT 1 = 0, but x1=0, x3=0)
        bits = (0, 1, 0)
        self.assertEqual(eval_clause(clause, bits), 0)
        
        # Assignment: x1=0, x2=0, x3=1 -> should satisfy (x3=1)
        bits = (0, 0, 1)
        self.assertEqual(eval_clause(clause, bits), 1)
    
    def test_eval_clause_all_negative(self):
        """Test clause with all negative literals."""
        clause = [-1, -2, -3]  # NOT x1 OR NOT x2 OR NOT x3
        
        # All variables true -> clause false
        bits = (1, 1, 1)
        self.assertEqual(eval_clause(clause, bits), 0)
        
        # One variable false -> clause true
        bits = (0, 1, 1)
        self.assertEqual(eval_clause(clause, bits), 1)


class TestBruteForce(unittest.TestCase):
    def test_satisfiable_instance(self):
        """Test brute force on known satisfiable instance."""
        # Simple satisfiable: (x1) AND (x2)
        clauses = [[1], [2]]
        solution, checks = sat_bruteforce(clauses, 2)
        
        self.assertIsNotNone(solution)
        self.assertEqual(solution, (1, 1))  # x1=1, x2=1
        self.assertLessEqual(checks, 4)  # Should find it quickly
    
    def test_unsatisfiable_instance(self):
        """Test brute force on unsatisfiable instance."""
        # Unsatisfiable: (x1) AND (NOT x1)
        clauses = [[1], [-1]]
        solution, checks = sat_bruteforce(clauses, 1)
        
        self.assertIsNone(solution)
        self.assertEqual(checks, 2)  # Must check all 2 assignments


class TestUnitPropagation(unittest.TestCase):
    def test_simple_unit_propagation(self):
        """Test basic unit propagation."""
        # CNF: (x1) AND (x1 OR x2) -> should propagate x1=1
        clauses = [[1], [1, 2]]
        result_clauses, assignment = unit_propagate(clauses)
        
        self.assertIsNotNone(result_clauses)
        self.assertEqual(assignment, {1: 1})
        self.assertEqual(result_clauses, [])  # All clauses satisfied
    
    def test_unit_propagation_conflict(self):
        """Test unit propagation with conflict."""
        # CNF: (x1) AND (NOT x1) -> should detect conflict
        clauses = [[1], [-1]]
        result_clauses, assignment = unit_propagate(clauses)
        
        self.assertIsNone(result_clauses)
        self.assertIsNone(assignment)
    
    def test_no_unit_clauses(self):
        """Test unit propagation when no unit clauses exist."""
        clauses = [[1, 2], [2, 3], [-1, -2]]
        result_clauses, assignment = unit_propagate(clauses)
        
        self.assertEqual(result_clauses, clauses)  # No change
        self.assertEqual(assignment, {})


class TestPureLiteralElimination(unittest.TestCase):
    def test_pure_literal_basic(self):
        """Test basic pure literal elimination."""
        # CNF: (x1 OR x2) AND (x1 OR x3) -> x1, x2, x3 are all pure positive
        clauses = [[1, 2], [1, 3]]
        result_clauses, assignment = pure_literal_elim(clauses, {})
        
        self.assertIsNotNone(result_clauses)
        self.assertEqual(assignment, {1: 1, 2: 1, 3: 1})  # All pure literals assigned
        self.assertEqual(result_clauses, [])  # All clauses satisfied
    
    def test_no_pure_literals(self):
        """Test when no pure literals exist."""
        # CNF: (x1 OR x2) AND (NOT x1 OR NOT x2) -> both x1 and x2 appear positive and negative
        clauses = [[1, 2], [-1, -2]]
        result_clauses, assignment = pure_literal_elim(clauses, {})
        
        self.assertEqual(result_clauses, clauses)  # No change
        self.assertEqual(assignment, {})


class TestRSACCorrectness(unittest.TestCase):
    def test_rsac_finds_correct_solutions(self):
        """Test that RSAC finds same solutions as brute force."""
        rng = random.Random(42)
        
        for n in [3, 4, 5]:  # Small instances for exhaustive testing
            for _ in range(5):  # Multiple random instances
                clauses = gen_random_kcnf(n, n+2, 3, rng)
                
                # Get brute force solution
                bf_solution, bf_checks = sat_bruteforce(clauses, n)
                
                # Get RSAC solution
                rsac_solution, rsac_checks, key = bucket_search_no_oracle(clauses, n, eval_clause)
                
                # Both should agree on satisfiability
                if bf_solution is None:
                    self.assertIsNone(rsac_solution, f"BF says UNSAT but RSAC found solution for n={n}")
                else:
                    self.assertIsNotNone(rsac_solution, f"BF found solution but RSAC says UNSAT for n={n}")
                    
                    # Verify RSAC solution actually satisfies the clauses
                    for clause in clauses:
                        self.assertEqual(eval_clause(clause, rsac_solution), 1,
                                       f"RSAC solution doesn't satisfy clause {clause}")


if __name__ == '__main__':
    unittest.main()