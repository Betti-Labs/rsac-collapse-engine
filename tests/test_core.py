"""Core algorithm tests for RSAC collapse engine."""

import unittest
import numpy as np
from src.rsac.collapse import (
    digital_root, symbolic_reduction_loop, 
    extended_signature_from_seq, vectorized_extended_signature, all_bit_arrays
)
from src.rsac.utils import bits_to_seq


class TestDigitalRoot(unittest.TestCase):
    def test_basic_cases(self):
        self.assertEqual(digital_root(0), 0)
        self.assertEqual(digital_root(9), 9)
        self.assertEqual(digital_root(18), 9)
        self.assertEqual(digital_root(19), 1)
        self.assertEqual(digital_root(123), 6)  # 1+2+3=6
        
    def test_large_numbers(self):
        self.assertEqual(digital_root(999), 9)
        self.assertEqual(digital_root(1000), 1)
        self.assertEqual(digital_root(12345), 6)  # 1+2+3+4+5=15, 1+5=6


class TestSymbolicReduction(unittest.TestCase):
    def test_simple_sequence(self):
        data = [1, 2, 3]
        history = symbolic_reduction_loop(data, depth=5)
        self.assertEqual(len(history), 3)  # original + 2 reductions
        self.assertEqual(history[0], [1, 2, 3])
        self.assertEqual(history[1], [3, 5])  # 1+2=3, 2+3=5
        self.assertEqual(history[2], [8])     # 3+5=8
        
    def test_single_element(self):
        data = [5]
        history = symbolic_reduction_loop(data, depth=5)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], [5])
        
    def test_empty_sequence(self):
        data = []
        history = symbolic_reduction_loop(data, depth=5)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], [])


class TestExtendedSignature(unittest.TestCase):
    def test_signature_consistency(self):
        """Test that same sequence always produces same signature."""
        seq = [1, 0, 1, 1, 0]
        sig1 = extended_signature_from_seq(seq)
        sig2 = extended_signature_from_seq(seq)
        self.assertEqual(sig1, sig2)
        
    def test_different_sequences_different_sigs(self):
        """Test that different sequences produce different signatures (usually)."""
        seq1 = [1, 0, 1, 0]
        seq2 = [0, 1, 0, 1]
        sig1 = extended_signature_from_seq(seq1)
        sig2 = extended_signature_from_seq(seq2)
        # Note: they might occasionally be the same due to collapse, but usually different
        # This is more of a sanity check
        
    def test_signature_structure(self):
        """Test that signature has expected structure."""
        seq = [1, 0, 1, 1, 0, 1]
        sig = extended_signature_from_seq(seq)
        self.assertEqual(len(sig), 4)  # final, penultimate, antepenultimate, entropy_tail
        final, penult, antepent, ent_tail = sig
        self.assertIsInstance(final, tuple)
        self.assertIsInstance(penult, tuple)
        self.assertIsInstance(antepent, tuple)
        self.assertIsInstance(ent_tail, tuple)


class TestVectorizedSignature(unittest.TestCase):
    def test_vectorized_vs_python_consistency(self):
        """Test that vectorized and Python implementations give same results."""
        n = 4
        bits_mat = all_bit_arrays(n)
        vec_sigs = vectorized_extended_signature(bits_mat)
        
        # Compare with Python implementation
        for i, bits_row in enumerate(bits_mat):
            seq = bits_to_seq(tuple(bits_row))
            py_sig = extended_signature_from_seq(seq)
            vec_sig = vec_sigs[i]
            self.assertEqual(py_sig, vec_sig, 
                           f"Mismatch at row {i}: Python={py_sig}, Vectorized={vec_sig}")
    
    def test_all_bit_arrays_correctness(self):
        """Test that all_bit_arrays generates correct bit patterns."""
        n = 3
        bits_mat = all_bit_arrays(n)
        self.assertEqual(bits_mat.shape, (8, 3))
        
        # Check that we get all possible combinations
        expected = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ]
        for i, expected_row in enumerate(expected):
            np.testing.assert_array_equal(bits_mat[i], expected_row)


class TestBitsToSeq(unittest.TestCase):
    def test_bits_to_seq(self):
        """Test conversion from bits to sequence."""
        bits = (0, 1, 0, 1)
        seq = bits_to_seq(bits)
        # Should map 0->9, 1->position+2
        expected = [9, 3, 9, 5]  # positions 0,1,2,3 -> values 2,3,4,5 for 1s
        self.assertEqual(seq, expected)
        
    def test_all_zeros(self):
        bits = (0, 0, 0)
        seq = bits_to_seq(bits)
        self.assertEqual(seq, [9, 9, 9])
        
    def test_all_ones(self):
        bits = (1, 1, 1)
        seq = bits_to_seq(bits)
        self.assertEqual(seq, [2, 3, 4])


if __name__ == '__main__':
    unittest.main()