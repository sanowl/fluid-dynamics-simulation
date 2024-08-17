#include "fluid_simulation.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define IX(i,j,k) ((i) + (N+2)*(j) + (N+2)*(N+2)*(k))
#define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}
#define MAX_VELOCITY 200.0f

__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__global__ void advect3D(int N, float* d, float* d0, float* u, float* v, float* w, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > N+1 || j > N+1 || k > N+1) return;

    float x = i - dt*N*u[IX(i,j,k)];
    float y = j - dt*N*v[IX(i,j,k)];
    float z = k - dt*N*w[IX(i,j,k)];

    if (x < 0.5f) x = 0.5f; if (x > N + 0.5f) x = N + 0.5f;
    if (y < 0.5f) y = 0.5f; if (y > N + 0.5f) y = N + 0.5f;
    if (z < 0.5f) z = 0.5f; if (z > N + 0.5f) z = N + 0.5f;

    int i0 = (int)x, i1 = i0 + 1;
    int j0 = (int)y, j1 = j0 + 1;
    int k0 = (int)z, k1 = k0 + 1;

    float s1 = x - i0, s0 = 1 - s1;
    float t1 = y - j0, t0 = 1 - t1;
    float u1 = z - k0, u0 = 1 - u1;

    d[IX(i,j,k)] = 
        s0 * (t0 * (u0 * d0[IX(i0,j0,k0)] + u1 * d0[IX(i0,j0,k1)]) +
              t1 * (u0 * d0[IX(i0,j1,k0)] + u1 * d0[IX(i0,j1,k1)])) +
        s1 * (t0 * (u0 * d0[IX(i1,j0,k0)] + u1 * d0[IX(i1,j0,k1)]) +
              t1 * (u0 * d0[IX(i1,j1,k0)] + u1 * d0[IX(i1,j1,k1)]));
}

__global__ void project3D(int N, float* u, float* v, float* w, float* p, float* div)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > N || j > N || k > N) return;

    div[IX(i,j,k)] = -0.5f*(
        u[IX(i+1,j,k)] - u[IX(i-1,j,k)] +
        v[IX(i,j+1,k)] - v[IX(i,j-1,k)] +
        w[IX(i,j,k+1)] - w[IX(i,j,k-1)]
    ) / N;
    p[IX(i,j,k)] = 0;
    __syncthreads();

    for (int l = 0; l < 20; l++) {
        p[IX(i,j,k)] = (div[IX(i,j,k)] +
            p[IX(i-1,j,k)] + p[IX(i+1,j,k)] +
            p[IX(i,j-1,k)] + p[IX(i,j+1,k)] +
            p[IX(i,j,k-1)] + p[IX(i,j,k+1)]) / 6;
        __syncthreads();
    }

    u[IX(i,j,k)] -= 0.5f * N * (p[IX(i+1,j,k)] - p[IX(i-1,j,k)]);
    v[IX(i,j,k)] -= 0.5f * N * (p[IX(i,j+1,k)] - p[IX(i,j-1,k)]);
    w[IX(i,j,k)] -= 0.5f * N * (p[IX(i,j,k+1)] - p[IX(i,j,k-1)]);
}

__global__ void addBuoyancy(int N, float* v, float* temp, float* density, float alpha, float beta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > N || j > N || k > N) return;

    float buoyancy = alpha * temp[IX(i,j,k)] - beta * density[IX(i,j,k)];
    v[IX(i,j,k)] += buoyancy * 0.01f; // Time step hardcoded for simplicity
}

__global__ void vorticityConfinement(int N, float* u, float* v, float* w, float* vorticity, float vorticityScale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < 1 || i > N || j < 1 || j > N || k < 1 || k > N) return;

    float dudy = (u[IX(i,j+1,k)] - u[IX(i,j-1,k)]) * 0.5f;
    float dudz = (u[IX(i,j,k+1)] - u[IX(i,j,k-1)]) * 0.5f;
    float dvdx = (v[IX(i+1,j,k)] - v[IX(i-1,j,k)]) * 0.5f;
    float dvdz = (v[IX(i,j,k+1)] - v[IX(i,j,k-1)]) * 0.5f;
    float dwdx = (w[IX(i+1,j,k)] - w[IX(i-1,j,k)]) * 0.5f;
    float dwdy = (w[IX(i,j+1,k)] - w[IX(i,j-1,k)]) * 0.5f;

    vorticity[IX(i,j,k)] = ((dwdy - dvdz) * (dwdy - dvdz) + 
                            (dudz - dwdx) * (dudz - dwdx) + 
                            (dvdx - dudy) * (dvdx - dudy));

    __syncthreads();

    float vort_dx = (vorticity[IX(i+1,j,k)] - vorticity[IX(i-1,j,k)]) * 0.5f;
    float vort_dy = (vorticity[IX(i,j+1,k)] - vorticity[IX(i,j-1,k)]) * 0.5f;
    float vort_dz = (vorticity[IX(i,j,k+1)] - vorticity[IX(i,j,k-1)]) * 0.5f;

    float len = sqrtf(vort_dx*vort_dx + vort_dy*vort_dy + vort_dz*vort_dz) + 1e-5f;
    vort_dx /= len; vort_dy /= len; vort_dz /= len;

    u[IX(i,j,k)] += vorticityScale * (vort_dy * (dwdy - dvdz) - vort_dz * (dvdx - dudy));
    v[IX(i,j,k)] += vorticityScale * (vort_dz * (dudz - dwdx) - vort_dx * (dwdy - dvdz));
    w[IX(i,j,k)] += vorticityScale * (vort_dx * (dvdx - dudy) - vort_dy * (dudz - dwdx));
}

__global__ void enforceBoundaries(int N, float* x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > N+1 || j > N+1 || k > N+1) return;

    x[IX(0,j,k)] = x[IX(1,j,k)];
    x[IX(N+1,j,k)] = x[IX(N,j,k)];
    x[IX(i,0,k)] = x[IX(i,1,k)];
    x[IX(i,N+1,k)] = x[IX(i,N,k)];
    x[IX(i,j,0)] = x[IX(i,j,1)];
    x[IX(i,j,N+1)] = x[IX(i,j,N)];
}

void FluidSimulation::step() {
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    advect3D<<<numBlocks, threadsPerBlock>>>(N, d_velocity_x, d_velocity_x0, d_velocity_x, d_velocity_y, d_velocity_z, dt);
    advect3D<<<numBlocks, threadsPerBlock>>>(N, d_velocity_y, d_velocity_y0, d_velocity_x, d_velocity_y, d_velocity_z, dt);
    advect3D<<<numBlocks, threadsPerBlock>>>(N, d_velocity_z, d_velocity_z0, d_velocity_x, d_velocity_y, d_velocity_z, dt);

    project3D<<<numBlocks, threadsPerBlock>>>(N, d_velocity_x, d_velocity_y, d_velocity_z, d_pressure, d_divergence);

    addBuoyancy<<<numBlocks, threadsPerBlock>>>(N, d_velocity_y, d_temperature, d_density, buoyancyAlpha, buoyancyBeta);

    vorticityConfinement<<<numBlocks, threadsPerBlock>>>(N, d_velocity_x, d_velocity_y, d_velocity_z, d_vorticity, vorticityScale);

    enforceBoundaries<<<numBlocks, threadsPerBlock>>>(N, d_velocity_x);
    enforceBoundaries<<<numBlocks, threadsPerBlock>>>(N, d_velocity_y);
    enforceBoundaries<<<numBlocks, threadsPerBlock>>>(N, d_velocity_z);

    advect3D<<<numBlocks, threadsPerBlock>>>(N, d_density, d_density0, d_velocity_x, d_velocity_y, d_velocity_z, dt);
    advect3D<<<numBlocks, threadsPerBlock>>>(N, d_temperature, d_temperature0, d_velocity_x, d_velocity_y, d_velocity_z, dt);

    // Add forces, sources, etc. here

    cudaDeviceSynchronize();

    // Use Thrust to find max velocity for adaptive time stepping
    thrust::device_ptr<float> d_ptr_vx(d_velocity_x);
    thrust::device_ptr<float> d_ptr_vy(d_velocity_y);
    thrust::device_ptr<float> d_ptr_vz(d_velocity_z);

    float max_velocity_x = *thrust::max_element(d_ptr_vx, d_ptr_vx + (N+2)*(N+2)*(N+2));
    float max_velocity_y = *thrust::max_element(d_ptr_vy, d_ptr_vy + (N+2)*(N+2)*(N+2));
    float max_velocity_z = *thrust::max_element(d_ptr_vz, d_ptr_vz + (N+2)*(N+2)*(N+2));

    float max_velocity = fmaxf(max_velocity_x, fmaxf(max_velocity_y, max_velocity_z));
    dt = fminf(MAX_VELOCITY / max_velocity, 1.0f) * 0.1f; // Adaptive time step
}

// Additional methods for initialization, boundary conditions, etc. would be implemented here