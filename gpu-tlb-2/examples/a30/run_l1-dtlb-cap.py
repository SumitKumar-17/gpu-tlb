#!/usr/bin/python3

import subprocess
import time
import pagemap
import os
import sys
from run_util import *



if len(sys.argv) > 1:
  SMID0 = int(sys.argv[1])
  SMID1 = int(sys.argv[2])
  cmd = ["./l1-dtlb-cap", str(SMID0), str(SMID1)]
  prepare(cmd)

  # extract PTEs from the pagemap
  ptes = pagemap.retrieve_ptes(tmp_path + "pagemap")
  pte_pa = ptes[target_va][1]
  pte_val = pagemap.make_pte_value(ptes[dummy_va][0])

  print(f"{hex(target_va)}: {ptes[target_va]}")
  print(f"{hex(dummy_va)}: {ptes[dummy_va]}")
  print(f"pte_pa: {hex(pte_pa)}")
  print(f"pte_val: {hex(pte_val)}")
  # b_out = subprocess.Popen(["nvidia-smi", "-f", "/dev/null"])
  # time.sleep(1)
  # b_out.kill()
  print(f"[*] testing: {SMID0} {SMID1}")
  cmd[1] = str(SMID0)
  cmd[2] = str(SMID1)
  launch_eviction(cmd, pte_pa, pte_val, sleep_time=10)
  exit(0)

root_elements = list(range(0, 15, 2))
queue = [i for i in range(1, 56) if i not in root_elements]
topology = {}
print(f"root_elements: {root_elements}")
print(f"queue: {queue}")

# topology = {
# 0: [1, 32, 33, 48, 49],
# 2: [3, 34, 35, 50, 51],
# 4: [5, 36, 37, 52, 53],
# 6: [7, 38, 39, 54, 55],
# 8: [9, 16, 17, 24, 25, 40, 41],
# 10: [11, 18, 19, 26, 27, 42, 43],
# 12: [13, 20, 21, 28, 29, 44, 45],
# 14: [15, 22, 23, 30, 31, 46, 47],
# }

cmd = ["./l1-dtlb-cap", "0", "1"]
for SMID0 in root_elements:
  for SMID1 in queue:
  # for SMID1 in range(SMID0 + 1, 56, 2):
    if SMID0 == SMID1:
      continue
    print(f"[*] testing: {SMID0} {SMID1}")
    cmd[1] = str(SMID0)
    cmd[2] = str(SMID1)
    time.sleep(3)
    prepare(cmd)
    ptes = pagemap.retrieve_ptes(tmp_path + "pagemap")
    pte_pa = ptes[target_va][1]
    pte_val = pagemap.make_pte_value(ptes[dummy_va][0])
    b_out = subprocess.Popen(["nvidia-smi", "-f", "/dev/null"])
    time.sleep(1)
    b_out.kill()
    if launch_eviction(cmd, pte_pa, pte_val, sleep_time=10):
      if SMID0 not in topology:
        topology[SMID0] = [SMID1]
      else:
        topology[SMID0].append(SMID1)
      print(topology)
  for element in topology[SMID0]:
    queue.remove(element)

print(topology)
