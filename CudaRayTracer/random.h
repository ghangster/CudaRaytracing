#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "vec3.h"

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}


__device__ vec3 random_unit_vector(curandState* local_rand_state) {
    return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
    while (true) {
        float x = curand_uniform(local_rand_state) * 2.0f - 1.0f;
        float y = curand_uniform(local_rand_state) * 2.0f - 1.0f;
        auto p = vec3(x, y, 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}