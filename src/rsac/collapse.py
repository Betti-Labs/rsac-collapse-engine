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

import numpy as np

def digital_root(n: int) -> int:
    return n if n == 0 else 9 if n % 9 == 0 else n % 9

def symbolic_reduction_loop(data, depth=16):
    history = [data.copy()]
    current = data.copy()
    for _ in range(depth):
        if len(current) < 2:
            break
        reduced = [digital_root(current[i] + current[i+1]) for i in range(len(current)-1)]
        current = reduced
        history.append(current.copy())
    return history

def extended_signature_from_seq(seq: list) -> tuple:
    """Generate extended signature as described in the paper: final, penultimate, third-from-last layers plus entropy tail."""
    hist = symbolic_reduction_loop(seq, depth=16)
    L = len(hist)
    
    # Extract layers (final, penultimate, antepenultimate)
    final_layer = tuple(hist[-1]) if L >= 1 else ()
    penultimate = tuple(hist[-2]) if L >= 2 else ()
    antepenultimate = tuple(hist[-3]) if L >= 3 else ()
    
    # Entropy tail: unique elements count for last 5 stages (or all if fewer)
    if L >= 5:
        ent_tail = tuple(len(set(stage)) for stage in hist[-5:])
    else:
        ent_tail = tuple(len(set(stage)) for stage in hist)
    
    return (final_layer, penultimate, antepenultimate, ent_tail)

# ---------- Vectorized helpers ----------
def all_bit_arrays(m: int) -> np.ndarray:
    N = 1 << m
    arr = np.zeros((N, m), dtype=np.uint8)
    for j in range(m):
        block = 1 << j
        reps = N // (block << 1)
        pattern = np.concatenate([np.zeros(block, dtype=np.uint8), np.ones(block, dtype=np.uint8)])
        arr[:, j] = np.tile(pattern, reps)
    return arr

def vectorized_extended_signature(bits_mat: np.ndarray) -> list:
    R, m = bits_mat.shape
    true_vals = np.arange(2, m+2, dtype=np.int64)  # 2..m+1
    seq = np.where(bits_mat == 1, true_vals, 9).astype(np.int64)

    entropies = []
    layers = [seq]
    cur = seq
    for width in range(m, 1, -1):
        cur = 1 + ((cur[:, :-1] + cur[:, 1:] - 1) % 9)
        layers.append(cur)
        entropies.append(np.array([len(np.unique(row)) for row in cur], dtype=np.int16))

    final_layer = layers[-1]               # (R,1)
    penultimate = layers[-2] if m >= 2 else None  # (R,2)
    antepenultimate = layers[-3] if m >= 3 else None  # (R,3)

    if len(entropies) >= 5:
        tail = entropies[-5:]
    else:
        tail = entropies
    ent_tail = np.stack(tail, axis=0).T if len(tail) > 0 else np.zeros((R,0), dtype=np.int16)

    keys = []
    for i in range(R):
        fin = int(final_layer[i, 0])
        pen = tuple(int(x) for x in penultimate[i]) if penultimate is not None else tuple()
        ant = tuple(int(x) for x in antepenultimate[i]) if antepenultimate is not None else tuple()
        ent_tuple = tuple(int(x) for x in ent_tail[i]) if ent_tail.shape[1] > 0 else tuple()
        keys.append((fin, pen, ant, ent_tuple))
    return keys
