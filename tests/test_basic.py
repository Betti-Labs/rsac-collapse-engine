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

"""Basic tests for RSAC functionality."""

import pytest
import random
from src.rsac.sat import gen_random_kcnf, sat_bruteforce, eval_clause
from src.rsac.bucket_search import bucket_search_no_oracle
from src.rsac.collapse import digital_root, extended_signature_from_seq
from src.rsac.utils import bits_to_seq


def test_digital_root():
    """Test digital root calculation."""
    assert digital_root(0) == 0
    assert digital_root(9) == 9
    assert digital_root(10) == 1
    assert digital_root(19) == 1
    assert digital_root(38) == 2


def test_bits_to_seq():
    """Test bit sequence conversion."""
    result = bits_to_seq((1, 0, 1))
    expected = [2, 9, 4]  # positions 0,1,2 -> values 2,3,4, with False->9
    assert result == expected


def test_signature_generation():
    """Test signature generation for simple cases."""
    # Test with a simple bit pattern
    bits = (1, 0, 1, 0)
    seq = bits_to_seq(bits)
    signature = extended_signature_from_seq(seq)
    
    # Should return a tuple with 4 components
    assert isinstance(signature, tuple)
    assert len(signature) == 4


def test_clause_evaluation():
    """Test clause evaluation."""
    # Simple clause: [1, -2] means (x1 OR NOT x2)
    clause = [1, -2]
    
    # Test case: x1=True, x2=False -> True OR True = True
    assert eval_clause(clause, (1, 0)) == 1
    
    # Test case: x1=False, x2=True -> False OR False = False  
    assert eval_clause(clause, (0, 1)) == 0
    
    # Test case: x1=True, x2=True -> True OR False = True
    assert eval_clause(clause, (1, 1)) == 1


def test_random_kcnf_generation():
    """Test random k-CNF generation."""
    rng = random.Random(42)
    clauses = gen_random_kcnf(4, 6, 3, rng)
    
    assert len(clauses) == 6  # 6 clauses
    for clause in clauses:
        assert len(clause) == 3  # 3 literals per clause
        for lit in clause:
            assert abs(lit) <= 4  # Variables 1-4


def test_brute_force_sat():
    """Test brute force SAT solver."""
    # Simple satisfiable formula: (x1) AND (x1 OR x2)
    clauses = [[1], [1, 2]]
    result, checks = sat_bruteforce(clauses, 2)
    
    assert result is not None  # Should be satisfiable
    assert result[0] == 1  # x1 must be True
    assert checks <= 4  # Should find solution quickly


def test_rsac_basic_functionality():
    """Test basic RSAC functionality."""
    # Simple satisfiable formula
    clauses = [[1, 2], [-1, 2]]  # (x1 OR x2) AND (NOT x1 OR x2)
    
    result = bucket_search_no_oracle(clauses, 2, eval_clause)
    assignment, checks, signature = result
    
    assert assignment is not None  # Should find solution
    assert checks > 0  # Should have done some checks
    assert signature is not None  # Should have a signature


def test_rsac_vs_brute_force_correctness():
    """Test that RSAC and brute force agree on satisfiability."""
    rng = random.Random(123)
    
    for _ in range(5):  # Test 5 random instances
        clauses = gen_random_kcnf(4, 8, 3, rng)
        
        # Brute force result
        bf_result, _ = sat_bruteforce(clauses, 4)
        bf_satisfiable = bf_result is not None
        
        # RSAC result  
        rsac_result = bucket_search_no_oracle(clauses, 4, eval_clause)
        rsac_satisfiable = rsac_result[0] is not None
        
        # Should agree on satisfiability
        assert bf_satisfiable == rsac_satisfiable


def test_empty_clauses():
    """Test edge case with no clauses (should be satisfiable)."""
    clauses = []
    result, checks = sat_bruteforce(clauses, 2)
    
    assert result is not None  # Empty formula is satisfiable
    assert checks == 1  # Should find solution immediately


def test_unsatisfiable_formula():
    """Test unsatisfiable formula."""
    # (x1) AND (NOT x1) - clearly unsatisfiable
    clauses = [[1], [-1]]
    
    bf_result, _ = sat_bruteforce(clauses, 1)
    rsac_result = bucket_search_no_oracle(clauses, 1, eval_clause)
    
    assert bf_result is None  # Brute force: unsatisfiable
    assert rsac_result[0] is None  # RSAC: unsatisfiable


if __name__ == "__main__":
    pytest.main([__file__])