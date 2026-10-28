"""Build and CPU-test the narrow NUMA affinity guard in a verified CPU task."""
import argparse
import errno
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

CHECK_ERRNOS = {'online_request': 0, 'different16_request': 0, 'null_pointer': errno.EINVAL,
                'missing_env': errno.EINVAL, 'bad_csv': errno.EINVAL,
                'duplicate_cpu': errno.EINVAL, 'negative_cpu': errno.EINVAL,
                'out_of_range': errno.EINVAL, 'mask_mismatch': errno.EINVAL,
                'worker_self': 0, 'worker_other_tid': errno.EPERM}

# Direct extern calls link against libnuma's versioned symbol, as HIP's PLT does.
# No HIP, Torch, images, model construction or forward is involved.
FIXTURE_C = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

struct bitmask { unsigned long size; unsigned long *maskp; };
extern int numa_sched_setaffinity(pid_t, struct bitmask *);
static cpu_set_t expected;
static char expected_csv[CPU_SETSIZE * 12];
static unsigned long words[CPU_SETSIZE / (8 * sizeof(unsigned long))];
static struct bitmask suggestion = { CPU_SETSIZE, words };
static int worker_passed;

static int csv(const cpu_set_t *set, char *buffer, size_t size)
{
    size_t used = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, set)) continue;
        int count = snprintf(buffer + used, size - used, "%s%d", used ? "," : "", cpu);
        if (count < 0 || (size_t)count >= size - used) return -1;
        used += (size_t)count;
    }
    return used ? 0 : -1;
}

static int request(const char *text)
{
    memset(words, 0, sizeof(words));
    while (*text) {
        char *end;
        long cpu = strtol(text, &end, 10);
        if (end == text || cpu < 0 || cpu >= CPU_SETSIZE || (*end && *end != ',')) return -1;
        words[cpu / (8 * sizeof(unsigned long))] |= 1UL << (cpu % (8 * sizeof(unsigned long)));
        text = *end ? end + 1 : end;
    }
    return 0;
}

static int probe(const char *name, pid_t pid, struct bitmask *mask, int wanted_errno)
{
    cpu_set_t before, after;
    char actual_csv[CPU_SETSIZE * 12] = {0};
    CPU_ZERO(&before);
    CPU_ZERO(&after);
    if (sched_getaffinity(0, sizeof(before), &before) != 0) return 0;
    errno = 0;
    int result = numa_sched_setaffinity(pid, mask);
    int error = errno;
    int passed = sched_getaffinity(0, sizeof(after), &after) == 0 &&
        CPU_EQUAL(&before, &expected) && CPU_EQUAL(&after, &expected) &&
        csv(&after, actual_csv, sizeof(actual_csv)) == 0 &&
        result == (wanted_errno ? -1 : 0) && error == wanted_errno;
    printf("{\"name\":\"%s\",\"passed\":%s,\"return_code\":%d,\"error\":%d,"
           "\"cpu_mask\":\"%s\",\"provider_verified\":true}\n",
           name, passed ? "true" : "false", result, error, actual_csv);
    return passed;
}

static void *worker(void *unused)
{
    (void)unused;
    worker_passed = probe("worker_self", (pid_t)syscall(SYS_gettid), &suggestion, 0);
    worker_passed &= probe("worker_other_tid", getpid(), &suggestion, EPERM);
    return NULL;
}

int main(int argc, char **argv)
{
    Dl_info provider;
    char actual_library[PATH_MAX], expected_library[PATH_MAX];
    if (argc != 5 || sched_getaffinity(0, sizeof(expected), &expected) != 0 ||
        csv(&expected, expected_csv, sizeof(expected_csv)) != 0 || strcmp(expected_csv, argv[1]) != 0)
        return 2;
    void *symbol = dlsym(RTLD_DEFAULT, "numa_sched_setaffinity");
    if (!symbol || !dladdr(symbol, &provider) || !realpath(provider.dli_fname, actual_library) ||
        !realpath(argv[2], expected_library) || strcmp(actual_library, expected_library) != 0)
        return 3;  /* Never risk calling the original NUMA implementation. */
    int passed = 1;
    if (request(argv[3]) != 0) return 4;
    passed &= probe("online_request", 0, &suggestion, 0);
    if (request(argv[4]) != 0) return 4;
    passed &= probe("different16_request", 0, &suggestion, 0);
    passed &= probe("null_pointer", 0, NULL, EINVAL);
    if (unsetenv("NCCCU_NATIVE_CPU_MASK") != 0) return 5;
    passed &= probe("missing_env", 0, &suggestion, EINVAL);
    if (setenv("NCCCU_NATIVE_CPU_MASK", "0,", 1) != 0) return 5;
    passed &= probe("bad_csv", 0, &suggestion, EINVAL);
    char invalid[64];
    int first = (int)strtol(expected_csv, NULL, 10);
    snprintf(invalid, sizeof(invalid), "%d,%d", first, first);
    if (setenv("NCCCU_NATIVE_CPU_MASK", invalid, 1) != 0) return 5;
    passed &= probe("duplicate_cpu", 0, &suggestion, EINVAL);
    if (setenv("NCCCU_NATIVE_CPU_MASK", "-1", 1) != 0) return 5;
    passed &= probe("negative_cpu", 0, &suggestion, EINVAL);
    snprintf(invalid, sizeof(invalid), "%d", CPU_SETSIZE);
    if (setenv("NCCCU_NATIVE_CPU_MASK", invalid, 1) != 0) return 5;
    passed &= probe("out_of_range", 0, &suggestion, EINVAL);
    if (setenv("NCCCU_NATIVE_CPU_MASK", argv[4], 1) != 0) return 5;
    passed &= probe("mask_mismatch", 0, &suggestion, EINVAL);
    if (setenv("NCCCU_NATIVE_CPU_MASK", expected_csv, 1) != 0) return 5;
    pthread_t thread;
    if (pthread_create(&thread, NULL, worker, NULL) != 0 || pthread_join(thread, NULL) != 0) return 6;
    passed &= worker_passed;
    cpu_set_t after;
    CPU_ZERO(&after);
    if (sched_getaffinity(0, sizeof(after), &after) != 0 || !CPU_EQUAL(&after, &expected)) return 7;
    return passed ? 0 : 1;
}
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_budget():
    path = Path(__file__).resolve().with_name('cpu_budget.py')
    spec = importlib.util.spec_from_file_location('native_guard_cpu_budget', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_context():
    require(all(os.environ.get(key) == '' for key in
                ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')),
            'native guard build requires hidden accelerators')
    path = Path(os.environ.get('NCCCU_CPU_ALLOCATION', ''))
    digest = os.environ.get('NCCCU_CPU_ALLOCATION_SHA256', '')
    require(path.is_absolute() and not path.is_symlink() and path.is_file()
            and re.fullmatch('[0-9a-f]{64}', digest), 'explicit CPU allocation path/SHA required')
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == digest, 'CPU allocation SHA differs')
    allocation = json.loads(content)
    budget = load_budget()
    require(allocation.get('guard_sha256') == sha(budget.__file__), 'allocated CPU guard source differs')
    identity = budget.identity()  # Includes actual hostname, interpreter and nice 19.
    policy = budget.load_resource_policy(allocation.get('resource_policy', {}))
    require(os.environ.get('NCCCU_RESOURCE_POLICY') == policy['path']
            and os.environ.get('NCCCU_RESOURCE_POLICY_SHA256') == policy['sha256'], 'CPU policy env differs')
    original = allocation['resources']
    budget.validation_pool(original)
    cpuset = budget.parse_cpuset(Path('/sys/fs/cgroup/cpuset.cpus.effective').read_text())
    maximum = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    require(len(maximum) == 2 and int(maximum[1]) > 0
            and (maximum[0] == 'max' or int(maximum[0]) > 0), 'invalid live CPU quota')
    affinity = original['affinity']
    require(isinstance(affinity, list) and affinity and affinity == sorted(set(affinity))
            and all(type(cpu) is int and cpu >= 0 for cpu in affinity), 'invalid original CPU affinity')
    available = sorted(set(affinity) & set(cpuset))
    capacity = Fraction(len(available))
    if maximum[0] != 'max':
        capacity = min(capacity, Fraction(int(maximum[0]), int(maximum[1])))
    total = int(capacity * Fraction(policy['value']['cpu_fractions']['validation']))
    live = {'affinity': affinity, 'cpuset': cpuset, 'available': available, 'cpu_max': maximum,
            'effective_cpu_fraction': str(capacity), 'effective_cpus': float(capacity),
            'max_total_threads': total, 'validation_mask': available[:total], 'resource_policy': policy}
    owner = os.environ.get('NCCCU_EXECUTOR_ID', '')
    granted = budget.validate_allocation(allocation, live, identity, owner, now=time.time())
    mask = json.loads(os.environ.get('NCCCU_CPU_MASK', 'null'))
    threads = os.environ.get('NCCCU_CPU_THREADS', '')
    require(isinstance(mask, list) and all(type(cpu) is int for cpu in mask)
            and mask == granted['mask'] == sorted(os.sched_getaffinity(0))
            and threads == str(granted['threads'])
            and all(os.environ.get(key) == threads for key in budget.THREAD_ENV)
            and all(os.environ.get(key) == value
                    for key, value in policy['value']['native_affinity_environment'].items()),
            'actual inherited CPU mask/thread limits differ from the allocation')
    return {'identity': identity, 'nice': os.getpriority(os.PRIO_PROCESS, 0), 'mask': mask,
            'threads': int(threads), 'executor_id': owner,
            'allocation': {'path': str(path), 'sha256': digest}, 'resource_policy': policy,
            'thread_environment': {key: os.environ[key] for key in budget.THREAD_ENV},
            'native_affinity_environment': {key: os.environ[key]
                                            for key in policy['value']['native_affinity_environment']},
            'original_resources': original, 'live_resources_from_original_affinity': live,
            'online': budget.parse_cpuset(Path('/sys/devices/system/cpu/online').read_text()),
            'gpu_visibility': {key: os.environ[key] for key in
                               ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}}


def reserve_output(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and not path.is_symlink(),
            'explicit nonsymlink absolute output required')
    parts = path.resolve().parts
    require(any(parts[index:index + 2] == ('exp-v3', 'technical-prep')
                and index + 2 < len(parts) for index in range(len(parts) - 1)),
            'output must be a new directory under exp-v3/technical-prep')
    require(not path.exists(), 'output exists; never overwrite or retry a build directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path.resolve()


def validate_fixture_output(text, mask):
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    require(len(records) == len(CHECK_ERRNOS)
            and {item.get('name') for item in records} == set(CHECK_ERRNOS), 'incomplete native fixture checks')
    expected = ','.join(map(str, mask))
    for item in records:
        error = CHECK_ERRNOS[item['name']]
        require(item.get('passed') is True and item.get('provider_verified') is True
                and item.get('cpu_mask') == expected and type(item.get('return_code')) is int
                and item['return_code'] == (-1 if error else 0)
                and type(item.get('error')) is int and item['error'] == error,
                'native fixture did not prove unchanged allocation/strict refusal: ' + item['name'])
    return records


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def build(args):
    require(args.source.is_absolute() and '..' not in args.source.parts
            and not args.source.is_symlink() and args.source.is_file() and args.source.suffix == '.c',
            'explicit regular absolute C source required')
    source = args.source.resolve(strict=True)
    cpu = cpu_context()
    output = reserve_output(args.output)
    artifact = output / 'libncccu_numa_affinity.so'
    fixture_source, fixture = output / 'NUMA-FIXTURE.c', output / 'numa-fixture'
    report = {'schema_version': 1, 'passed': False, 'scope': 'compute_pod_native_affinity_guard',
              'source': {'path': str(source), 'sha256': sha(source)},
              'artifact': {'path': str(artifact), 'sha256': None}, 'cpu': cpu, 'checks': [],
              'compiler': None, 'commands': [], 'gpu_initialized': False, 'forward_calls': 0,
              'training_lock_acquired': False, 'formal_clock_created': False}
    write_json(output / 'STARTED.json', report)
    started = time.time()

    def command(argv, name, environment, timeout=60):
        with (output / (name + '.log')).open('x') as log:
            result = subprocess.run(argv, cwd=output, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
        report['commands'].append({'command': argv, 'exit_code': result.returncode,
                                   'log': str(output / (name + '.log'))})
        require(result.returncode == 0, name + ' failed; see its actual log')
        return (output / (name + '.log')).read_text()

    try:
        compiler = shutil.which('cc')
        require(compiler is not None, 'a Pod C compiler is required')
        compiler = str(Path(compiler).resolve(strict=True))
        environment = os.environ.copy()
        require(not environment.get('LD_PRELOAD'), 'build channel must start without an unverified preload')
        report['compiler'] = {'path': compiler, 'version': command([compiler, '--version'], 'COMPILER', environment)}
        shared_command = [compiler, '-shared', '-fPIC', '-O2', '-Wall', '-Wextra', '-Werror',
                          str(source), '-ldl', '-o', str(artifact)]
        command(shared_command, 'BUILD-GUARD', environment)
        require(sha(source) == report['source']['sha256'], 'C source changed during compilation')
        report['compiler']['command'] = shared_command
        report['artifact']['sha256'] = sha(artifact)
        fixture_source.write_text(FIXTURE_C)
        fixture_command = [compiler, '-O2', '-Wall', '-Wextra', '-Werror', '-fPIE', '-pie',
                           str(fixture_source), '-Wl,-l:libnuma.so.1', '-pthread', '-ldl', '-o', str(fixture)]
        command(fixture_command, 'BUILD-FIXTURE', environment)
        report['fixture'] = {'source': {'path': str(fixture_source), 'sha256': sha(fixture_source)},
                             'artifact': {'path': str(fixture), 'sha256': sha(fixture)},
                             'compiler_command': fixture_command}
        online, mask = cpu['online'], cpu['mask']
        different = online[:16] if online[:16] != mask else online[-16:]
        require(len(different) == 16 and different != mask, 'a different online 16-CPU request is required')
        csv = lambda values: ','.join(map(str, values))
        environment.update(LD_PRELOAD=str(artifact), NCCCU_NATIVE_CPU_MASK=csv(mask))
        text = command([str(fixture), csv(mask), str(artifact), csv(online), csv(different)],
                       'NATIVE-FIXTURE', environment, timeout=15)
        report['checks'] = validate_fixture_output(text, mask)
        report['cpu_after'] = cpu_context()
        require(report['cpu_after'] == cpu, 'actual CPU allocation/context changed during the build')
        require(report['artifact']['sha256'] == sha(artifact), 'native artifact changed during its CPU check')
        report['passed'] = True
        report['exit_code'] = 0
    except Exception as error:
        report['error'] = repr(error)
        report['exit_code'] = 1
    report['wall_seconds'] = time.time() - started
    write_json(output / 'NATIVE-AFFINITY.json', report)
    print(json.dumps({'receipt': str(output / 'NATIVE-AFFINITY.json'),
                      'sha256': sha(output / 'NATIVE-AFFINITY.json'), 'exit_code': report['exit_code']}, indent=2))
    return report['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return build(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
