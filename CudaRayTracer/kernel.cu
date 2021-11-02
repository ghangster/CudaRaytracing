
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "windows.h"
#include "shellapi.h"

//#include "rtweekend.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "camera.h"
#include "ray.h"

#include "curand_kernel.h"

#include <iostream>
#include <math.h>
//#include <cmath>



#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n" << cudaGetErrorString(result) << '\n' << cudaGetErrorName(result) << '\n';
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ bool hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0 * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius * radius;
    auto discriminant = b * b - 4 * a * c;
    return (discriminant > 0);
}

__device__ vec3 Color(const ray& r, hittable** world) {

    hit_record rec;

    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    }

    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__device__ vec3 Color_iter(const ray& r, hittable** world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = color(1, 1, 1);

    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001, FLT_MAX, rec)) {

            ray scattered;
            color attenuation;

            if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation = cur_attenuation * attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }

        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            
            return cur_attenuation * c;
        }
    }
    return color(0, 0, 0);
}



__global__ void render_init(int max_x, int max_y, curandState* rand_state) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;

    curand_init(1991, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, curandState* rand_state) {

    curand_init(0, 0, 0, rand_state);

    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = rand_state[0];
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
            new lambertian(vec3(0.5, 0.5, 0.5)));

        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = curand_uniform(rand_state);
                vec3 center(a + curand_uniform(rand_state), 0.2, b + curand_uniform(rand_state));
                if (choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                        new lambertian(vec3(curand_uniform(rand_state) * curand_uniform(rand_state), curand_uniform(rand_state) * curand_uniform(rand_state), curand_uniform(rand_state) * curand_uniform(rand_state))));
                }
                else if (choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                        new metal(vec3(0.5f * (1.0f + curand_uniform(rand_state)), 0.5f * (1.0f + curand_uniform(rand_state)), 0.5f * (1.0f + curand_uniform(rand_state))), 0.5f * curand_uniform(rand_state)));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

        *rand_state = local_rand_state;
        *d_world = new hittable_list(d_list, 22 * 22 + 1 + 3);


        const float aspect_ratio = 16.0 / 9.0;
        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            aspect_ratio,
            aperture,
            dist_to_focus);

        /// <summary>
        /// 
        /// <returns></returns>

        //material* diffuseMat = new lambertian(color(.5f, .5f, .5f));
        //material* metalMat = new metal(color(.6, .4, .2), .3f);
        //material* dielectricMat = new dielectric(1.5f);

        //*(d_list) = new sphere(vec3(0.f, 0.f, 0.f), 0.5f, dielectricMat);
        //*(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100, diffuseMat);
        //*(d_list + 2) = new sphere(vec3(-1.f, 0.f, -1.f), .5f, metalMat);
        //*(d_list + 3) = new sphere(vec3(1.f, 0.f, -1.f), .5f, diffuseMat);
        //*d_world = new hittable_list(d_list, 4);


        //const float aspect_ratio = 16.0 / 9.0;
        //point3 lookfrom(0, 0, 5);
        //point3 lookat(0, 0, -1);
        //vec3 vup(0, 1, 0);
        //auto dist_to_focus = 5.0;
        //auto aperture = 0.1;

        //*d_camera = new camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
    }
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hittable ** world, curandState* rand_state) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;

    curandState local_rand_state = rand_state[pixel_index];
    
    color col(0.0f, 0.0f, 0.0f);


    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        
        col += Color_iter(r, world, &local_rand_state);
     }


    col /= float(ns);
    color gamma_corrected(sqrt(col.x()), sqrt(col.y()), sqrt(col.z()));
    fb[pixel_index] = gamma_corrected;



}


int main(void)
{

    // Image
    vec3* image_data;
    const float aspect_ratio = 16.0 / 9.0;
    const int image_width = 2560;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int image_size = image_width * image_height;
    const int num_channels = 3;
    const int num_bytes = image_size * 3;
    const char* filename = "render.bmp";
    const int num_samples = 50;

    // Camera

    float viewport_height = 2.0f;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0f;

    vec3 origin = vec3(0.f, 0.f, 0.f);
    vec3 horizontal = vec3(viewport_width, 0.f, 0.f);
    vec3 vertical = vec3(0, viewport_height, 0);
    vec3 lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);


    // World

    hittable** d_list;
    int num_hittables = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void**) &d_list, num_hittables * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**) &d_world, sizeof(hittable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));




    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, image_size * sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

    // Allocate Unified Memory – accessible from CPU or GPU
    checkCudaErrors(cudaMallocManaged(&image_data, image_size * sizeof(vec3)));



    int tx = 8;
    int ty = 8;

    dim3 blocks(image_width / tx+1, image_height / ty+1);
    dim3 threads(tx, ty);

    render_init << <blocks, threads >> > (image_width, image_height, d_rand_state);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    create_world << <1, 1 >> > (d_list, d_world, d_camera, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render << <blocks, threads >> > (image_data, image_width, image_height, num_samples, d_camera, d_world, d_rand_state);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Save as bmp and show image in windows
    unsigned char* image_bmp = (unsigned char*)malloc(image_size * sizeof(unsigned char) * num_channels);

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t idx = j * image_width + i;
            size_t bmp_idx = ((image_height-j-1) * image_width + i) * 3; // turn downside up
            image_bmp[bmp_idx] = (unsigned char)(255.999 * image_data[idx].x());
            image_bmp[bmp_idx + 1] = (unsigned char)(255.999 * image_data[idx].y());
            image_bmp[bmp_idx + 2] = (unsigned char)(255.999 * image_data[idx].z());
        }
    }

    int result = stbi_write_bmp(filename, image_width, image_height, num_channels, image_bmp);
    ShellExecute(NULL, "open", "render.bmp", NULL, NULL, SW_NORMAL);


    // Free memory
    checkCudaErrors(cudaFree(image_data));

    return 0;
}