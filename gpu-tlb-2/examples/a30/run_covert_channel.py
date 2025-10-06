#!/usr/bin/python3

import sys
import os

from run_util import *
from eviction_sets import *

num_entries_rcv = 10
num_entries_snd = 20

if len(sys.argv) > 1:
    num_entries_rcv = int(sys.argv[1])
    num_entries_snd = int(sys.argv[2]) if len(sys.argv) > 2 else 20

os.system("make")

base_addr_rcv = 0x700000000000
base_addr_snd = 0x702000000000

# eviction_set_rcv = get_slice(get_eviction_set(18, n=100, indexed=False, base_address=base_addr_rcv), indexed=True)[:num_entries_rcv]
eviction_set_rcv = get_eviction_set(18, n=100, indexed=True, base_address=base_addr_rcv)[:num_entries_rcv]
target = eviction_set_rcv[0]
target_va = base_addr_rcv + target * (1 * 1024 * 1024)
slice_id = get_slice_address(target_va)
write_to_file(eviction_set_rcv, "./out/eviction_set_rcv.txt", base_address=base_addr_rcv)

eviction_set_snd = list(map(str, get_slice_set(18, n=num_entries_snd, indexed=True, base_address=base_addr_snd, slice_id=slice_id)))
write_to_file(eviction_set_snd, "./out/eviction_set_snd.txt", base_address=base_addr_snd)

#eviction_set_receiver = list(map(str, get_eviction_set(18, n=100, indexed=True, base_address=base_addr_receiver)[:10]))

# cmd_sender = ["./sender"] + eviction_set_sender[1:]
# cmd_receiver = ["./receiver"] + eviction_set_sender[:10]
#
# print(f"[*] cmd sender: {' '.join(cmd_sender)}")
# print(f"[*] cmd receiver: {' '.join(cmd_receiver)}")
# print(f"[*] size: {len(eviction_set_sender)}")
#
# sudo = subprocess.Popen(["sudo", "echo", ""])
# sudo.wait()
#
# a_out = subprocess.Popen(cmd_sender,
#     env={"CUDA_VISIBLE_DEVICES": "MIG-f1ededa2-530e-5c7b-9a72-cdab47b7e9e3"})
#
# b_out = subprocess.Popen(["nvidia-smi", "-f", "/dev/null"])
# b_out.wait()
#
# c_out = subprocess.Popen(cmd_receiver,
#     env={"CUDA_VISIBLE_DEVICES": "MIG-7d82bf30-4052-5c13-81e8-8bdef9be314c"})
# # env={"CUDA_VISIBLE_DEVICES": "MIG-f1ededa2-530e-5c7b-9a72-cdab47b7e9e3"})


# time.sleep(100)
# a_out.kill()
# b_out.kill()