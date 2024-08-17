#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <memory>

class FluidSimulation {
public:
    FluidSimulation(int N, float dt);
    ~FluidSimulation();

    void step();
    void addImpulse(float2 position, float force = 5.0f);
    
    float* getDensity() const { return h_density.get(); }
    float3* getVelocityField() const { return h_velocity.get(); }
    float* getTemperature() const { return h_temperature.get(); }
    float* getPressure() const { return h_pressure.get(); }

private:
    int N;  // grid size
    float dt;  // time step
    
    // Host arrays (using smart pointers for automatic memory management)
    std::unique_ptr<float[]> h_density;
    std::unique_ptr<float3[]> h_velocity;
    std::unique_ptr<float[]> h_temperature;
    std::unique_ptr<float[]> h_pressure;
    
    // Device arrays
    float *d_density, *d_density_prev;
    float3 *d_velocity, *d_velocity_prev;
    float *d_temperature, *d_temperature_prev;
    float *d_pressure, *d_divergence;
    float3 *d_vorticity;
    
    // Simulation parameters
    struct SimParams {
        float viscosity;
        float diffusion;
        float buoyancyAlpha;
        float buoyancyBeta;
        float vorticityScale;
        float3 gravity;
    } params;

    cudaStream_t stream1, stream2;

    void allocateMemory();
    void freeMemory();
    void copyToHost();

    void advect();
    void diffuse();
    void project();
    void applyBuoyancy();
    void applyVorticityConfinement();
    void enforceBoundaries();

    void computeVorticity();
    void computeDivergence();
    void solvePressure();
};