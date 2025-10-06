#!/usr/bin/python3

import sys
import os

from run_util import *
from eviction_sets import get_eviction_set, get_slice

if len(sys.argv) > 1:
    num_entries = int(sys.argv[1])
else:
    num_entries = 18

os.system("make")

base_addr = 0x700000000000
eviction_set_addr = get_slice(get_eviction_set(18, n=100, indexed=False))[:num_entries]
eviction_set = []
for addr in eviction_set_addr:
    eviction_set.append((addr ^ base_addr) >> 20)
target = eviction_set[0]
eviction_set = list(map(str, eviction_set))

cmd = ["./tlb-timing"] + eviction_set
print(f"[*] cmd: {' '.join(cmd)}")
print(f"[*] size: {len(eviction_set)}")

sudo = subprocess.Popen(["sudo", "echo", ""])
sudo.wait()
a_out = subprocess.Popen(cmd)
b_out = subprocess.Popen(["nvidia-smi", "-f", "/dev/null"])
b_out.wait() # in case the driver warms up very slowly
for _ in range(50):
    time.sleep(1)
    if a_out.poll() is not None:
        break
a_out.kill()
