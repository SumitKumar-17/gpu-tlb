#!/usr/bin/python3

import subprocess
import time
import os



dumper_path = "../../dumper/dumper"
extractor_path = "../../extractor/extractor"
modifier_path = "../../modifier/modifier"
tmp_path = "/tmp/"

target_va = 0x700000000000
dummy_va = 0x7f0000000000
# target_va = 0x100000000000
# dummy_va =  0x200000000000
code_pa = 0x0007c00000


def prepare(argv, sleep_time=2, dump_size="0x10000000"):
    os.system("make")
    #===============================================================================
    # modify PTE for target_va to make it point to the dummy_va"s page frame
    #===============================================================================
    sudo = subprocess.Popen(["sudo", "echo", ""])
    sudo.wait()
    a_out = subprocess.Popen(argv)
    b_out = subprocess.Popen(["nvidia-smi", "-f", "/dev/null"])
    b_out.wait() # in case the driver warms up very slowly
    time.sleep(sleep_time)
    tmp_path = "/tmp/"
    dump = subprocess.Popen([dumper_path, "-b", dump_size, "-o", tmp_path + "dump"])
    dump.wait()
    pagemap_file = open(tmp_path + "pagemap", "w")
    extract = subprocess.Popen([extractor_path, tmp_path + "dump"], stdout=pagemap_file)
    extract.wait()
    pagemap_file.close()
    a_out.kill()
    time.sleep(2)

def launch_eviction(cmd, pte_pa, pte_val, sleep_time=8):
    a_out = subprocess.Popen(cmd)
    b_out = subprocess.Popen(["nvidia-smi", "-f", "/dev/null"])
    b_out.wait() # in case the driver warms up very slowly
    time.sleep(2)
    subprocess.Popen(["sudo", modifier_path, hex(pte_pa), hex(pte_val)])
    print("[*] modification done")
    for i in range(sleep_time):
        time.sleep(1)
        if a_out.poll() != None:
            print("eviction successful")
            return True
    else:
        print("no eviction")
        a_out.kill()
        return False


