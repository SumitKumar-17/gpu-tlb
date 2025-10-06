#!/usr/bin/python3

import subprocess
import time
import pagemap
import sys
from run_util import *

prepare(["./fill"])

# extract PTEs from the pagemap
ptes = pagemap.retrieve_ptes(tmp_path + 'pagemap')

print(f"{hex(target_va)}: {ptes[target_va]}")
print(f"{hex(dummy_va)}: {ptes[dummy_va]}")

pte_pa = ptes[target_va][1]
pte_val = pagemap.make_pte_value(ptes[dummy_va][0])

print(f"pte_pa: {hex(pte_pa)}")
print(f"pte_val: {hex(pte_val)}")

# do experiments
launch_eviction(["./fill"], pte_pa, pte_val)


