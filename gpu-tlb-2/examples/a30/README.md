# A30 TLB Tests

## Description

The examples given here are specifically tuned for running on A30 GPUs.

## Run

First of all, turn off the ASLR to make the process repeatable.
```
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space
```

If you also want to permanently turn if off, write to a file under /etc/sysctl.d/.

```
echo "kernel.randomize_va_space=0" | sudo tee /etc/sysctl.d/01-aslr.conf 
```

### Fill

This test checks how many pages are needed to flush the TLB hierarchy.

```
./run_fill.py
```

Result: 6000 pages are enough to flush the TLB hierarchy.

### L1-dTLB

This test checks how many entries can be filled in the L1-dTLB. It iterates PAGE0_NUM (17) on SM X and PAGE1_NUM (6000) on SM Y.

```
# ./run_l1-dtlb-cap.py <SM X> <SM Y>
./run_l1-dtlb-cap.py 0 1
```


### L2-uTLB

You can use the L1-dTLB tests to find the sets of SM which shares the L2-uTLB:

```
./run_l1-dtlb-cap.py.py
```

Alternatively, you can also use:

```
./run_l2-utlb-set-topology.py
```

that is using an eviction set for L2-uTLB from SM Y to evict a target in L2-uTLB of SM X.

### L3-uTLB

This test performs self eviction from the L3-uTLB on SM 0:

```
# ./run_l3-utlb-set-static.py <number of entries in eviction set>
./run_l3-utlb-set-static.py 18
```

Passing `18` creates an eviction set of 17 entries + 1 target to evict.

## Results

I didn't find that the L1-dTLB was shared between SMs, and these are the sets of SMs which share the L2-uTLB:

```
{0, 1, 32, 33, 48, 49},
{2: 3, 34, 35, 50, 51},
{4, 5, 36, 37, 52, 53},
{6, 7, 38, 39, 54, 55},
{8, 9, 16, 17, 24, 25, 40, 41},
{10, 11, 18, 19, 26, 27, 42, 43},
{12, 13, 20, 21, 28, 29, 44, 45},
{14, 15, 22, 23, 30, 31, 46, 47}
```

The L3-uTLB is shared across all SMs, and it can be verified by changing:

```
eviction_set = get_eviction_set(18, n=6, indexed=True)
```

to:

```
eviction_set = get_eviction_set(18, n=10, indexed=True)
```

in [./run_l2-utlb-set-topology.py](./run_l2-utlb-set-topology.py)