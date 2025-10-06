#!/usr/bin/python3

import subprocess
import time
import pagemap
from run_util import *

prepare(["./l1-itlb-cap", "0x7fffe1200000"], sleep_time=3, dump_size="0x20000000")

# extract PTEs from the pagemap
ptes = pagemap.retrieve_ptes(tmp_path + "pagemap")
code_va = None
for key, val in ptes.items():
  if val[0] == code_pa:
    code_va = key
    break
print("code is at " + hex(code_va))
target_va = code_va
pte_pa = ptes[target_va][1]
pte_val = pagemap.make_pte_value(ptes[dummy_va][0])

print(f"{hex(target_va)}: {ptes[target_va]}")
print(f"{hex(dummy_va)}: {ptes[dummy_va]}")
print(f"pte_pa: {hex(pte_pa)}")
print(f"pte_val: {hex(pte_val)}")
print(" ".join(["sudo", modifier_path, hex(pte_pa), hex(pte_val)]))

a_out = subprocess.Popen(["./l1-itlb-cap", hex(code_va)])
b_out = subprocess.Popen(["nvidia-smi", "-f", "/dev/null"])
b_out.wait() # in case the driver warms up very slowly
time.sleep(10)
subprocess.Popen(["sudo", modifier_path, hex(pte_pa), hex(pte_val)])
time.sleep(6)

a_out.kill()
if a_out.poll() != None:
  print("eviction successful")
else:
  print("no eviction")


