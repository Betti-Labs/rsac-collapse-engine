import itertools


def bits_to_seq(bits):
    # True -> index+2 (2..n+1), False -> 9
    return [(i + 2) if b == 1 else 9 for i, b in enumerate(bits)]


def iter_all_bits(n):
    return itertools.product([0, 1], repeat=n)
