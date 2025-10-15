# (Keep the l3_hash_rtx3060.py file exactly as it was before)
import numpy as np

def get_l3_set_index_rtx3060_64k(gpu_va):
    """Calculates the L3 set index for an RTX 3060 (Consumer Ampere)."""
    mask = ((1 << (46 - 20 + 1)) - 1) << 20
    relevant_bits = (gpu_va & mask) >> 20
    xor_lines = [
        [0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19],
        [4, 12, 20],    [5, 13, 21],    [6, 14, 22],    [7, 15, 23],
    ]
    set_index = 0
    for i, line in enumerate(xor_lines):
        xor_result = 0
        for bit_pos_relative in line:
            if 0 <= bit_pos_relative <= 26:
                if (relevant_bits >> bit_pos_relative) & 1:
                    xor_result ^= 1
        if xor_result: set_index |= (1 << i)
    return set_index