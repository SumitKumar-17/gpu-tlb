import sys
import subprocess


def main(pagemap):
    with open(pagemap, 'r') as f:
        lines = f.readlines()

    ptes = dict()
    tabs = [0, 0]
    pas = []
    for ent in lines:
        parts = ent.strip().split('-->')
        if 'Page' in ent:
            idx = int(parts[0])
            parts = parts[1].split('@')
            addrs = parts[1].split('VA: ')
            pa = int(addrs[0], 16)
            addrs[1] = addrs[1].split("\t")[0]
            va = int(addrs[1], 16)
            epa = tabs[0] + 16 * idx if '2MB' in ent else tabs[1] + 8 * idx
            ptes[va] = [pa, epa]
            print(f"pa: {hex(pa)}, va: {hex(va)}")
            pas.append(pa)

    found = False
    pas.insert(0, 0x0)
    for pa in pas[:]:
        dump = subprocess.Popen(["../dumper/dumper", '-s', hex(pa), '-b', '0x10000000', '-o', f'dump_{hex(pa)}'])
        dump.wait()
        with open(f'dump_{hex(pa)}', 'rb') as f1:
            data = f1.read()
        chunks = [data[i:i+16] for i in range(0, len(data), 16)]
        for i, chunk in enumerate(chunks):
            # if chunk != b"\x00" * 16:
            #     print(chunk)
            # 0x00000a0000017a02
            # 0x003fde0000000f00
            # 0x0000000000027805
            # 0x003fde0000015000
            # if chunk == b"\x02\x7a\x01\x00\x00\x0a\x00\x00\x00\x0f\x00\x00\x00\xde\x3f\x00\x05x\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00P\x01\x00\x00\xde?\x00\x00\x00\x00\x00\x00\x00\x00\x00":
            if chunk == b"\x05x\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00P\x01\x00\x00\xde?\x00\x00\x00\x00\x00\x00\x00\x00\x00":
                print(f"[*] found at {hex(pa)}, {hex(i)}")
                found = True
                break
            # else:
            #     print(chunk)
        if found:
            break

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pagemap")