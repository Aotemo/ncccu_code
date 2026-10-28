/* Ignore only NUMA's automatic binding suggestion after verifying our thread's
 * explicit role mask. Ordinary sched/pthread affinity APIs remain untouched. */
#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

struct bitmask;

static int explicit_mask(cpu_set_t *mask)
{
    const char *p = getenv("NCCCU_NATIVE_CPU_MASK");
    CPU_ZERO(mask);
    if (p == NULL || *p == '\0')
        return -1;
    for (;;) {
        int cpu = 0;
        if (*p < '0' || *p > '9' ||
            (*p == '0' && p[1] >= '0' && p[1] <= '9'))
            return -1;
        do {
            cpu = cpu * 10 + (*p++ - '0');
            if (cpu >= CPU_SETSIZE)
                return -1;
        } while (*p >= '0' && *p <= '9');
        if (CPU_ISSET(cpu, mask))
            return -1;
        CPU_SET(cpu, mask);
        if (*p == '\0')
            return 0;
        if (*p++ != ',' || *p == '\0')
            return -1;
    }
}

int numa_sched_setaffinity(pid_t pid, struct bitmask *suggestion)
{
    cpu_set_t expected, actual;
    if (pid != 0 && (long)pid != syscall(SYS_gettid)) {
        errno = EPERM;
        return -1;
    }
    if (suggestion == NULL || explicit_mask(&expected) != 0) {
        errno = EINVAL;
        return -1;
    }
    CPU_ZERO(&actual);
    if (sched_getaffinity(0, sizeof(actual), &actual) != 0)
        return -1;
    if (!CPU_EQUAL(&actual, &expected)) {
        errno = EINVAL;
        return -1;
    }
    return 0;
}
