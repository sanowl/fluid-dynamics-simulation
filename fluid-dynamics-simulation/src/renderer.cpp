#include "renderer.h"
#include "shader.h"
#include "texture.h"
#include "particle_system.h"
#include "fluid_simulation.h"
#include "camera.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdexcept>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stb_image.h>

Renderer::Renderer(int N, const std::string& windowTitle) 
    : N(N), particleSystem(N * N), camera(glm::vec3(0.0f, 0.0f, 3.0f)), fluidSimulation(N) {
    initOpenGL();
    initImGui();
    initShaders();
    initTextures();
    initQuad();
    initFramebuffer();
    initParticleSystem();
    glfwSetWindowTitle(window, windowTitle.c_str());

    // Set up callback data
    glfwSetWindowUserPointer(window, this);
}

Renderer::~Renderer() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}

void Renderer::initOpenGL() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);  // Enable MSAA

    window = glfwCreateWindow(1280, 720, "Advanced Fluid Simulation", NULL, NULL);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    if (glewInit() != GLEW_OK) {
        throw std::runtime_error("Failed to initialize GLEW");
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Renderer::initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");
}

void Renderer::initShaders() {
    fluidShader = std::make_unique<Shader>("shaders/fluid.vert", "shaders/fluid.frag");
    postProcessShader = std::make_unique<Shader>("shaders/post_process.vert", "shaders/post_process.frag");
    particleShader = std::make_unique<Shader>("shaders/particle.vert", "shaders/particle.frag", "shaders/particle.geom");
}

void Renderer::initTextures() {
    textures.push_back(std::make_unique<Texture>(N, N, GL_RED, GL_FLOAT));   // Density
    textures.push_back(std::make_unique<Texture>(N, N, GL_RGB, GL_FLOAT));   // Velocity
    textures.push_back(std::make_unique<Texture>(N, N, GL_RED, GL_FLOAT));   // Temperature
    textures.push_back(std::make_unique<Texture>(N, N, GL_RED, GL_FLOAT));   // Pressure
    textures.push_back(std::make_unique<Texture>(N, N, GL_RGB, GL_FLOAT));   // Vorticity
    textures.push_back(std::make_unique<Texture>(N, N, GL_RGB, GL_FLOAT));   // Color

    // Load background texture
    int width, height, nrChannels;
    unsigned char *data = stbi_load("textures/background.jpg", &width, &height, &nrChannels, 0);
    if (data) {
        backgroundTexture = std::make_unique<Texture>(width, height, GL_RGB, GL_UNSIGNED_BYTE, data);
        stbi_image_free(data);
    } else {
        throw std::runtime_error("Failed to load background texture");
    }
}

void Renderer::initQuad() {
    float quadVertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
}

void Renderer::initFramebuffer() {
    glGenFramebuffers(1, &FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);

    // Create a color attachment texture
    glGenTextures(1, &textureColorbuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 1280, 720, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);

    // Create a renderbuffer object for depth and stencil attachment
    glGenRenderbuffers(1, &RBO);
    glBindRenderbuffer(GL_RENDERBUFFER, RBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 1280, 720);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Framebuffer is not complete!");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::initParticleSystem() {
    particleSystem.init();
}

void Renderer::updateSimulationParameters() {
    fluidSimulation.setViscosity(viscosity);
    fluidSimulation.setDiffusion(diffusion);
    fluidSimulation.setGravity(gravity);
    fluidSimulation.setVorticityConfinement(vorticityConfinement);
    fluidSimulation.setTemperatureBuoyancy(temperatureBuoyancy);
    fluidSimulation.setDensityBuoyancy(densityBuoyancy);
    fluidSimulation.setColorDiffusion(colorDiffusion);
    
    particleSystem.setParticleLifetime(particleLifetime);
    particleSystem.setParticleSize(particleSize);
    particleSystem.setParticleEmissionRate(particleEmissionRate);
}

void Renderer::run() {
    double lastTime = glfwGetTime();
    int frameCount = 0;

    while (!shouldClose()) {
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Limit the simulation to a fixed time step
        const float fixedTimeStep = 1.0f / 60.0f;
        float accumulator = 0.0f;
        accumulator += deltaTime;

        while (accumulator >= fixedTimeStep) {
            updateSimulationParameters();

            fluidSimulation.update(fixedTimeStep);
            particleSystem.update(fixedTimeStep, fluidSimulation.getVelocity());

            accumulator -= fixedTimeStep;
        }

        const float* density = fluidSimulation.getDensity();
        const float3* velocity = fluidSimulation.getVelocity();
        const float* temperature = fluidSimulation.getTemperature();
        const float* pressure = fluidSimulation.getPressure();
        const float3* vorticity = fluidSimulation.getVorticity();
        const float3* color = fluidSimulation.getColor();

        render(density, velocity, temperature, pressure, vorticity, color);

        updateUI();

        glfwSwapBuffers(window);
        glfwPollEvents();

        frameCount++;
        if (currentFrame - lastTime >= 1.0) {
            float fps = static_cast<float>(frameCount) / static_cast<float>(currentFrame - lastTime);
            updateFPSCounter(fps);
            frameCount = 0;
            lastTime = currentFrame;
        }

        handleInput();
    }
}

void Renderer::handleInput() {
    if (isKeyPressed(GLFW_KEY_R)) {
        resetSimulation();
    }

    if (isMousePressed) {
        glm::vec2 mousePos = getMousePosition();
        glm::vec2 simPos = screenToSimulationSpace(mousePos.x, mousePos.y);
        glm::vec2 mouseDelta = mousePos - lastMousePos;
        
        if (glm::length(mouseDelta) > 0.0f) {
            glm::vec2 impulse = glm::normalize(mouseDelta) * impulseStrength;
            fluidSimulation.addImpulse(simPos.x, simPos.y, impulse.x, impulse.y);
            fluidSimulation.addColor(simPos.x, simPos.y, brushColor.r, brushColor.g, brushColor.b);
            particleSystem.emitParticles(simPos.x, simPos.y, impulse.x, impulse.y);
        }
        
        lastMousePos = mousePos;
    }

    if (isKeyPressed(GLFW_KEY_W)) camera.processKeyboard(Camera::FORWARD, deltaTime);
    if (isKeyPressed(GLFW_KEY_S)) camera.processKeyboard(Camera::BACKWARD, deltaTime);
    if (isKeyPressed(GLFW_KEY_A)) camera.processKeyboard(Camera::LEFT, deltaTime);
    if (isKeyPressed(GLFW_KEY_D)) camera.processKeyboard(Camera::RIGHT, deltaTime);
}

void Renderer::updateUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Simulation Controls");
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::ColorEdit3("Brush Color", &brushColor[0]);
    ImGui::SliderFloat("Viscosity", &viscosity, 0.0f, 1.0f);
    ImGui::SliderFloat("Diffusion", &diffusion, 0.0f, 1.0f);
    ImGui::SliderFloat("Impulse Strength", &impulseStrength, 1.0f, 50.0f);
    ImGui::SliderFloat3("Gravity", &gravity[0], -10.0f, 10.0f);
    ImGui::SliderFloat("Vorticity Confinement", &vorticityConfinement, 0.0f, 1.0f);
    ImGui::SliderFloat("Temperature Buoyancy", &temperatureBuoyancy, 0.0f, 1.0f);
    ImGui::SliderFloat("Density Buoyancy", &densityBuoyancy, -1.0f, 1.0f);
    ImGui::SliderFloat("Color Diffusion", &colorDiffusion, 0.0f, 1.0f);
    ImGui::SliderFloat("Particle Lifetime", &particleLifetime, 0.1f, 10.0f);
    ImGui::SliderFloat("Particle Size", &particleSize, 0.1f, 10.0f);
    ImGui::SliderFloat("Particle Emission Rate", &particleEmissionRate, 1.0f, 1000.0f);
    if (ImGui::Button("Reset Simulation")) {
        resetSimulation();
    }
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

#include "renderer.h"
#include "shader.h"
#include "texture.h"
#include "particle_system.h"
#include "fluid_simulation.h"
#include "camera.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdexcept>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stb_image.h>

// ... [Previous code remains unchanged]

void Renderer::render(const float* density, const float3* velocity, const float* temperature, const float* pressure, const float3* vorticity, const float3* color) {
    updateTextures(density, velocity, temperature, pressure, vorticity, color);

    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderBackground();
    renderFluid();
    renderParticles();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    applyPostProcess();
}

void Renderer::updateTextures(const float* density, const float3* velocity, const float* temperature, const float* pressure, const float3* vorticity, const float3* color) {
    textures[0]->updateData(density);
    textures[1]->updateData(reinterpret_cast<const float*>(velocity));
    textures[2]->updateData(temperature);
    textures[3]->updateData(pressure);
    textures[4]->updateData(reinterpret_cast<const float*>(vorticity));
    textures[5]->updateData(reinterpret_cast<const float*>(color));
}

void Renderer::renderBackground() {
    glDisable(GL_DEPTH_TEST);
    postProcessShader->use();
    postProcessShader->setInt("screenTexture", 0);
    backgroundTexture->bind(0);

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}

void Renderer::renderFluid() {
    fluidShader->use();
    for (size_t i = 0; i < textures.size(); ++i) {
        fluidShader->setInt("texture" + std::to_string(i), i);
        textures[i]->bind(i);
    }
    fluidShader->setFloat("time", glfwGetTime());
    fluidShader->setVec2("resolution", glm::vec2(1280, 720));
    fluidShader->setVec3("fluidColor", brushColor);
    fluidShader->setFloat("viscosity", viscosity);
    fluidShader->setFloat("diffusion", diffusion);

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void Renderer::renderParticles() {
    particleShader->use();
    particleShader->setMat4("projection", camera.getProjectionMatrix());
    particleShader->setMat4("view", camera.getViewMatrix());
    particleSystem.render();
}

void Renderer::applyPostProcess() {
    postProcessShader->use();
    postProcessShader->setInt("screenTexture", 0);
    postProcessShader->setFloat("time", glfwGetTime());
    postProcessShader->setVec2("resolution", glm::vec2(1280, 720));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureColorbuffer);

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

bool Renderer::shouldClose() const {
    return glfwWindowShouldClose(window);
}

void Renderer::updateFPSCounter(float fps) {
    std::string title = "Advanced Fluid Simulation - FPS: " + std::to_string(static_cast<int>(fps));
    glfwSetWindowTitle(window, title.c_str());
}

bool Renderer::isKeyPressed(int key) const {
    return glfwGetKey(window, key) == GLFW_PRESS;
}

glm::vec2 Renderer::getMousePosition() const {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    return glm::vec2(xpos, ypos);
}

glm::vec2 Renderer::screenToSimulationSpace(double xpos, double ypos) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    return glm::vec2(xpos / width * N, (height - ypos) / height * N);
}

void Renderer::resetSimulation() {
    fluidSimulation.reset();
    particleSystem.reset();
    camera = Camera(glm::vec3(0.0f, 0.0f, 3.0f));
}

void Renderer::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    auto* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    glViewport(0, 0, width, height);
    renderer->camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
}

void Renderer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    auto* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

void Renderer::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    auto* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            renderer->isMousePressed = true;
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            renderer->lastMousePos = glm::vec2(xpos, ypos);
        } else if (action == GLFW_RELEASE) {
            renderer->isMousePressed = false;
        }
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            renderer->isRightMousePressed = true;
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            renderer->lastMousePos = glm::vec2(xpos, ypos);
        } else if (action == GLFW_RELEASE) {
            renderer->isRightMousePressed = false;
        }
    }
}

void Renderer::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    auto* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    if (renderer->isMousePressed) {
        glm::vec2 mousePos(xpos, ypos);
        glm::vec2 mouseDelta = mousePos - renderer->lastMousePos;
        glm::vec2 simPos = renderer->screenToSimulationSpace(xpos, ypos);
        
        glm::vec2 impulse = glm::normalize(mouseDelta) * renderer->impulseStrength;
        renderer->fluidSimulation.addImpulse(simPos.x, simPos.y, impulse.x, impulse.y);
        renderer->fluidSimulation.addColor(simPos.x, simPos.y, renderer->brushColor.r, renderer->brushColor.g, renderer->brushColor.b);
        renderer->particleSystem.emitParticles(simPos.x, simPos.y, impulse.x, impulse.y);

        renderer->lastMousePos = mousePos;
    }

    if (renderer->isRightMousePressed) {
        float xoffset = xpos - renderer->lastMousePos.x;
        float yoffset = renderer->lastMousePos.y - ypos;
        renderer->camera.processMouseMovement(xoffset, yoffset);
        renderer->lastMousePos = glm::vec2(xpos, ypos);
    }
}

void Renderer::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    auto* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    renderer->camera.processMouseScroll(static_cast<float>(yoffset));
}