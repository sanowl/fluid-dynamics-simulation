#include "fluid_simulation.h"
#include "renderer.h"
#include "config_parser.h"
#include "logger.h"
#include <iostream>
#include <chrono>
#include <thread>

int main(int argc, char** argv) {
    try {
        Logger::init("fluid_sim.log");
        LOG_INFO("Starting Fluid Simulation");

        ConfigParser config("config.json");
        const int N = config.getInt("grid_size", 128);
        const float dt = config.getFloat("time_step", 0.1f);
        const int maxFPS = config.getInt("max_fps", 60);
        
        FluidSimulation sim(N, dt);
        Renderer renderer(N, config.getString("window_title", "Advanced Fluid Simulation"));

        std::chrono::microseconds frameTime(1000000 / maxFPS);
        std::chrono::steady_clock::time_point lastTime = std::chrono::steady_clock::now();

        while (!renderer.shouldClose()) {
            auto startTime = std::chrono::steady_clock::now();

            sim.step();
            
            renderer.render(sim.getDensity(), sim.getVelocityField(), sim.getTemperature(), sim.getPressure());

            auto endTime = std::chrono::steady_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

            if (elapsedTime < frameTime) {
                std::this_thread::sleep_for(frameTime - elapsedTime);
            }

            auto currentTime = std::chrono::steady_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
            lastTime = currentTime;

            renderer.updateFPSCounter(1.0f / deltaTime);

            if (renderer.isKeyPressed(GLFW_KEY_SPACE)) {
                sim.addImpulse(renderer.getMousePosition());
            }
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("Fatal error: {}", e.what());
        return 1;
    }

    LOG_INFO("Simulation ended normally");
    return 0;
}