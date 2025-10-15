#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#define SECRET 34
#define MAXSECRET 64

char table[MAXSECRET * 64];

void secretfunction(void) {
    char value = table[SECRET * 64];
}

int64_t timeguess(int guess) {
    uint64_t oldtsc, newtsc;
    asm volatile("clflush (%0);" // flush address from cache
                 "mfence;"       // make sure no prior memory operations get in the way
                 "rdtscp;"       // read timestamp into edx:eax
                 "shl $32,%%rdx;"
                 "or %%rdx,%%rax;"         // combine output in rax
                 : "=a"(oldtsc)            // output of tsc
                 : "r"(&table[guess * 64]) // parameter for clflush
                 : "%rcx", "%rdx"          // clobbered
    );
    secretfunction();
    asm volatile("mfence;" // make sure operation is complete
                 "rdtscp;"
                 "shl $32,%%rdx;"
                 "or %%rdx,%%rax;" // read timestamp, store in rax
                 : "=a"(newtsc)    // output of tsc
                 :                 // no input
                 : "%rcx", "%rdx"  // clobbered
    );
    // printf("Time difference for secret %d: %" PRIu64 "\n",guess,newtsc-oldtsc);
    return newtsc - oldtsc;
}

int findsecret() {
    uint64_t maxdiff = 0, sum = 0;
    int likelysecret = -1;
    for (int i = 0; i < MAXSECRET; i++) {
        uint64_t diff = timeguess(i);
        sum += diff;
        if (maxdiff < diff) {
            maxdiff = diff;
            likelysecret = i;
        }
    }
    printf("Most likely secret is %d, with a time difference of %" PRIu64
           ", compared to the average of %" PRIu64 "\n",
           likelysecret, maxdiff, sum / MAXSECRET);
    return likelysecret;
}

int main() {
    int secrets[MAXSECRET] = {0};
    for (int i = 0; i < 100; i++) {
        secrets[findsecret()]++;
    }
    int likelysecret = 0;
    for (int i = 0; i < MAXSECRET; i++) {
        if (secrets[likelysecret] < secrets[i])
            likelysecret = i;
    }
    printf("Final guess: %d, with %d out of 100 trials confirming it\n", likelysecret,
           secrets[likelysecret]);
    return 0;
}