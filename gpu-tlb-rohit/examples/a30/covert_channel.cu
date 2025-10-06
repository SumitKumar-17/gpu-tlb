#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include "common.h"

#define CHUNK0_SIZE (64L * 1024L * 1024L * 1024L * 1024L + 0x55554000000L)
#define CHUNK1_SIZE (41L * 1024L * 1024L * 1024L * 1024L + 0x0ffc8000000L)
#define STRIDE_SIZE (1L * 1024L * 1024L)

#define BASE_ADDR_RECEIVER  0x700000000000
#define BASE_ADDR_SENDER    0x702000000000
#define DUMMY_ADDR 0x7F0000000000

#define PAGE_NUM_RECEIVER 20000
#define PAGE_NUM_SENDER   80000
#define PAGE_NUM_L1  16
#define PAGE_FILL_NUM  6000
#define WAIT_TIME 500000000L
#define WAIT_TIME2 100000000L
#define ITERATIONS 5

#define BLK_NUM 100
#define SHARED_MEM (96 * 1024)
#define SMID0 0
#define SMID1 20


__device__ void rest(uint64_t wait_time) {
    uint64_t clk0 = 0;
    uint64_t clk1 = 0;
    clk0 = clock64();
    clk1 = 0;
    while (clk1 < wait_time)
        clk1 = clock64() - clk0;
}

__device__ uint64_t iterate_pages(volatile uint64_t *page_l2, volatile uint64_t *page_l1,
                                  uint32_t sm) {
    uint64_t clk0 = 0;
    uint64_t clk1 = 0;
    uint64_t cycle_count[ITERATIONS + 1] = {0};
    uint32_t count = 0;
    volatile uint64_t *ptr;
    volatile uint64_t *evt;

    while (count < ITERATIONS) {
        asm volatile("membar.cta;");
        clk0 = clock64();
        ptr = (uint64_t *)page_l2[0];
        clk1 = clock64();
        asm volatile("membar.cta;");
        cycle_count[count] = clk1 - clk0;

        // Fill L2 and put target in L3
        for (; ptr != page_l2; ptr = (uint64_t *)ptr[0]) {
            ++ptr[2];
        }

        // Fill L1-dTLB
        for (evt = (uint64_t *)page_l1[0]; evt != page_l1; evt = (uint64_t *)evt[0]) {
            atomicAdd((unsigned long long int *)(&evt[2]), 1);
        }

        count++;

        rest(WAIT_TIME);
    }
    // Not median
    return cycle_count[1];
}

__device__ void receiver(volatile uint64_t *page_l2, volatile uint64_t *page_l1, uint32_t len) {
    uint64_t cycle_count = 0;
    printf("[rcv] page_l2: 0x%lx, page_l2[0]: 0x%lx\n", (uint64_t)page_l2, (uint64_t)page_l2[0]);
    uint32_t message[10] = {0};
    // for (int i = 0; i < 100; i++) {
    //     rest(WAIT_TIME);
    // }
    for (int i = 0; i < len; ++i) {
        printf("[rcv] page_l2[0]: 0x%lx\n", (uint64_t)page_l2[0]);
        // asm volatile("membar.cta;");
        cycle_count = iterate_pages(page_l2, page_l1, SMID0);
        if (cycle_count < 1100) {
            message[i] = 0;
            printf("[!] RECEIVED: 0\n");
        } else {
            message[i] = 1;
            printf("[!] RECEIVED: 1\n");
        }
        printf("[-] receiver | cycle_count: %lu\n", cycle_count);
    }
    for (int i = 0; i < len; ++i) {
        printf("[rcv] message[%u]: %u\n", i, message[i]);
    }
}

__device__ void sender(volatile uint64_t *page_l2, volatile uint64_t *page_l1, uint32_t *to_send, uint32_t len) {
    printf("[snd] page_l2: 0x%lx, page_l2[0]: 0x%lx\n", (uint64_t)page_l2, (uint64_t)page_l2[0]);
    for (int i = 0; i < len; ++i) {
        if (to_send[i] == 1) {
            printf("[!] sender | 1\n");
            iterate_pages(page_l2, page_l1, SMID1);
        } else {
            printf("[!] sender | 0\n");
            for (int j = 0; j < ITERATIONS; ++j) {
                rest(WAIT_TIME);
            }
        }
    }
}

__device__ uint32_t barrier = 0;

__global__ void loop(volatile uint64_t *page_sender, volatile uint64_t *page_receiver, volatile uint64_t *page_l1,
                     volatile uint64_t *page_fill) {
    uint32_t smid;
    uint32_t to_send[] = {1, 0, 0, 1, 1, 0};
    uint32_t len = sizeof(to_send) / sizeof(to_send[0]);
    uint32_t message[10] = {0};
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    if (smid != SMID0 && smid != SMID1)
        return;

    for (uint32_t i = 0; i < len; i++) {

        for (uint64_t *ptr = (uint64_t *)page_fill[0]; ptr != page_fill; ptr = (uint64_t *)ptr[0]) {
            ++ptr[2];
        }

        atomicAdd(&barrier, 1);
        while (atomicOr(&barrier, 0) != 2) {
            ;
        }
        __threadfence();
        atomicExch(&barrier, 0);

        if (smid == SMID0) {
            // receiver(page_receiver, page_l1, len);
            uint64_t cycle_count = iterate_pages(page_receiver, page_l1, smid);
            printf("[rcv] cycle_count: %lu\n", cycle_count);
            if (cycle_count < 1100) {
                message[i] = 0;
            } else {
                message[i] = 1;
            }
        } else {
            if (to_send[i] == 1) {
                printf("[!] sender | 1\n");
                iterate_pages(page_sender, page_l1, SMID1);
            } else {
                printf("[!] sender | 0\n");
                for (int j = 0; j < ITERATIONS; ++j) {
                    rest(WAIT_TIME);
                }
            }
            // sender(page_sender, page_l1, to_send, len);
        }
    }
    if (smid == SMID0) {
        for (int i = 0; i < len; ++i) {
            printf("[rcv] message[%u]: %u\n", i, message[i]);
        }
    }
}

int main(int argc, char *argv[]) {
    uint8_t *chunk0 = NULL;
    uint8_t *chunk1 = NULL;
    uint64_t **list_rcv;
    uint64_t **list_snd;
    uint64_t **list_l1;
    uint64_t **list_fill;
    int aim = -1;

    struct __eviction_set *es_rcv = get_eviction_set(18, BASE_ADDR_RECEIVER, 9);
    aim = es_rcv->indexes[0];

    struct __eviction_set *es_snd = get_slice_set(18, BASE_ADDR_SENDER, 16, 0);

    cudaDeviceReset();
    cudaFuncSetAttribute(loop, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM);

    // hoard a large address space
    cudaMallocManaged(&chunk0, CHUNK0_SIZE);
    cudaMallocManaged(&chunk1, CHUNK1_SIZE);


    list_rcv = create_list(PAGE_NUM_RECEIVER, (uint8_t *)BASE_ADDR_RECEIVER, STRIDE_SIZE, es_rcv);
    list_snd = create_list(PAGE_NUM_SENDER, (uint8_t *)BASE_ADDR_SENDER, STRIDE_SIZE, es_snd);
    list_l1 = create_list(PAGE_NUM_L1, (uint8_t *)(idx_to_addr(aim, BASE_ADDR_RECEIVER) + STRIDE_SIZE), STRIDE_SIZE, NULL);
    list_fill = create_list(PAGE_FILL_NUM, (uint8_t *)DUMMY_ADDR + STRIDE_SIZE, STRIDE_SIZE, NULL);

    uint64_t *dummy = (uint64_t *)DUMMY_ADDR;
    put<<<1, 1>>>(dummy, 0, 0);

    cudaDeviceSynchronize();

    printf("Done hoarding, aim: 0x%x, page: 0x%lx\n", aim, (uint64_t)list_rcv[aim]);
    loop<<<BLK_NUM, 1, SHARED_MEM>>>(list_snd[es_snd->indexes[0]], list_rcv[aim], list_l1[0], list_fill[0]);
    // loop<<<BLK_NUM, 1, SHARED_MEM>>>(list_snd[0], list_rcv[aim], list_l1[0]);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaFree(chunk0);
    cudaFree(chunk1);

    destroy_eviction_set(es_rcv);
    destroy_eviction_set(es_snd);
    free(list_rcv);
    free(list_snd);
    free(list_l1);
    free(list_fill);
}
