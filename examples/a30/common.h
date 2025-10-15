#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>

#ifndef COMMON_H
#define COMMON_H

int xored[8][4] = {
    {44, 36, 28, 20},
    {45, 37, 29, 21},
    {46, 38, 30, 22},
    {39, 31, 23, -1},
    {40, 32, 24, -1},
    {41, 33, 25, -1},
    {42, 34, 26, -1},
    {43, 35, 27, -1}
};

int slice_xor[] = {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46};

struct __eviction_set {
    uint8_t *base_address;
    uint32_t *indexes;
    uint32_t count;
};

uint64_t *_malloc(size_t size) {
    uint64_t *ptr = (uint64_t *)malloc(size);
    if (ptr == NULL) {
        perror("malloc");
        exit(-1);
    }
    return ptr;
}

__device__ void sort(uint64_t *arr, uint32_t n) {
    for (uint32_t i = 0; i < n - 1; ++i) {
        for (uint32_t j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                uint64_t temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

__device__ uint64_t median(uint64_t *cycle_count, uint32_t count) {
    sort(cycle_count, count);
    return cycle_count[(uint32_t)count / 2];
}

__global__ void put(uint64_t *page, uint64_t x1, uint64_t x2) {
    page[0] = x1;
    page[1] = x2;
}

void parse_eviction_set(const char *filename, struct __eviction_set *es) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("fopen");
        exit(-1);
    }
    fscanf(fp, "%lx", (uint64_t *)&es->base_address);
    fscanf(fp, "%u", &es->count);
    uint32_t number = 0;
    uint32_t count = 0;
    es->indexes = (uint32_t *)malloc(sizeof(uint32_t) * es->count);
    while (fscanf(fp, "%u", &number) == 1) {
        es->indexes[count++] = number;
    }
    // printf("base_address: 0x%lx\n", (uint64_t)es->base_address);
    // printf("count: %u\n", es->count);
    // for (int i = 0; i < es->count; ++i) {
    //     printf("indexes[%u]: %u\n", i, es->indexes[i]);
    // }
    assert(es->count == count);
    fclose(fp);
}

uint64_t **create_list(uint64_t page_num, uint8_t *base_addr, uint64_t stride, struct __eviction_set *es) {
    uint64_t **list = (uint64_t **)malloc(page_num * sizeof(uint64_t *));
    for (uint64_t i = 0; i < page_num; i++) {
        list[i] = (uint64_t *)(base_addr + i * stride);
    }
    if (es) {
        for (int i = 0; i < es->count; i++) {
            int m = es->indexes[i];
            int n = es->indexes[(i + 1) % es->count];
            put<<<1, 1>>>(list[m], (uint64_t)list[n], 0xdeadbeef);
        }
    } else {
        for (int i = 0; i < page_num; i++) {
            put<<<1, 1>>>(list[i], (uint64_t)list[(i + 1) % page_num], 0xdeadbeef);
        }
    }
    return list;
}

void write_to_file(uint64_t* eviction_set, int length, const char* filename, uint64_t base_address) {
    FILE* f = fopen(filename, "w");
    if (f == NULL) {
        perror("Failed to open file");
        return;
    }

    fprintf(f, "0x%lx\n", base_address);
    fprintf(f, "%d\n", length);

    for (int i = 0; i < length; i++) {
        fprintf(f, "%lu\n", eviction_set[i]);
    }

    fclose(f);
}

int get_virtual_address(uint64_t address) {
    int cache_set[8] = {0};
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4 && xored[i][j] != -1; j++) {
            int v = xored[i][j];
            uint64_t mask = 1ULL << v;
            cache_set[i] ^= ((address & mask) >> v);
        }
    }

    // Convert binary array to integer
    int result = 0;
    for (int i = 0; i < 8; i++) {
        result = result * 2 + cache_set[i];
    }

    return result;
}

uint64_t idx_to_addr(uint64_t idx, uint64_t base_address) {
    return base_address ^ (idx << 20);
}

uint64_t addr_to_idx(uint64_t addr, uint64_t base_address) {
    return (addr ^ base_address) >> 20;
}

int get_slice_address(uint64_t addr) {
    int slice_id = 0;
    for (int i = 0; i < 22; i++) {
        int sx = slice_xor[i];
        slice_id = slice_id ^ ((addr >> sx) & 1);
    }
    return slice_id;
}

struct __eviction_set *get_eviction_set(int cache_set, uint64_t base_address, int n) {
    struct __eviction_set *es = (struct __eviction_set *)malloc(sizeof(struct __eviction_set));
    es->indexes = (uint32_t*)malloc(n * sizeof(uint32_t));
    es->count = n;
    es->base_address = (uint8_t *)base_address;
    uint64_t start_addr = base_address;
    uint64_t* addresses = (uint64_t*)malloc(n * sizeof(uint64_t));
    int count = 0;

    while (count < n) {
        start_addr += 0x100000;
        if (get_virtual_address(start_addr) == cache_set) {
            addresses[count++] = start_addr;
        }
    }

    for (int i = 0; i < n; i++) {
        es->indexes[i] = addr_to_idx(addresses[i], base_address);
    }
    free(addresses);
    return es;
}

struct __eviction_set* get_slice_set(int cache_set, uint64_t base_address, int n, int slice_id) {
    uint64_t start_addr = base_address;
    struct __eviction_set *es = (struct __eviction_set *)malloc(sizeof(struct __eviction_set));
    es->indexes = (uint32_t*)malloc(n * sizeof(uint32_t));
    es->count = n;
    es->base_address = (uint8_t *)base_address;

    uint64_t* addresses = (uint64_t*)malloc(n * sizeof(uint64_t));
    int count = 0;

    while (count < n) {
        start_addr += 0x100000;
        if (get_virtual_address(start_addr) == cache_set) {
            if (get_slice_address(start_addr) == slice_id) {
                addresses[count++] = start_addr;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        es->indexes[i] = addr_to_idx(addresses[i], base_address);
    }
    free(addresses);
    return es;
}

void print_eviction_set(struct __eviction_set *es, uint8_t index_only) {
    for (int i = 0; i < es->count; i++) {
        if (index_only) {
            printf("%u\n", es->indexes[i]);
        } else {
            printf("0x%lx\n", idx_to_addr(es->indexes[i], (uint64_t)es->base_address));
        }
    }
}

void destroy_eviction_set(struct __eviction_set *es) {
    free(es->indexes);
    free(es);
}

#endif
