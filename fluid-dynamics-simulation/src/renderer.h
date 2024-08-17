#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <memory>

class Shader;
class Texture;

class Renderer {
public:
    Renderer(int N, const std::string& windowTitle);
    ~Renderer();

    void render(const float* density, const float3* velocity, const float* temperature, const float* pressure);
    bool shouldClose() const;
    void updateFPSCounter(float fps);
    bool isKeyPressed(int key) const;
    glm::vec2 getMousePosition() const;

private:
    int N;
    GLFWwindow* window;
    std::unique_ptr<Shader> fluidShader;
    std::unique_ptr<Shader> postProcessShader;
    std::vector<std::unique_ptr<Texture>> textures;
    GLuint VAO, VBO;
    GLuint FBO, RBO;

    void initOpenGL();
    void initShaders();
    void initTextures();
    void initQuad();
    void initFramebuffer();
    void updateTextures(const float* density, const float3* velocity, const float* temperature, const float* pressure);
    void renderFluid();
    void applyPostProcess();

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
};