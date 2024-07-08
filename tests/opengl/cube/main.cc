#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <unistd.h> 
#include <string.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


#include "../common.h"

#define WIDTH 150
#define HEIGHT 100

static float position[] = {
  // FRONT
  -0.5, 0.5, -0.5,
  -0.5, -0.5, -0.5,
  0.5, -0.5, -0.5,
  -0.5, 0.5, -0.5,
  0.5, 0.5, -0.5,
  0.5, -0.5, -0.5,
  // BACK
  -0.5, 0.5, 0.5,
  -0.5, -0.5, 0.5,
  0.5, -0.5, 0.5,
  -0.5, 0.5, 0.5,
  0.5, 0.5, 0.5,
  0.5, -0.5, 0.5,
  // TOP
  -0.5, 0.5, 0.5,
  -0.5, 0.5, -0.5,
  0.5, 0.5, -0.5,
  -0.5, 0.5, 0.5,
  0.5, 0.5, 0.5,
  0.5, 0.5, -0.5,
  // BOTTOM
  -0.5, -0.5, 0.5,
  -0.5, -0.5, -0.5,
  0.5, -0.5, -0.5,
  -0.5, -0.5, 0.5,
  0.5, -0.5, 0.5,
  0.5, -0.5, -0.5,
  // LEFT
  -0.5, 0.5, -0.5,
  -0.5, -0.5, -0.5,
  -0.5, -0.5, 0.5,
  -0.5, 0.5, -0.5,
  -0.5, 0.5, 0.5,
  -0.5, -0.5, 0.5,
  // RIGHT
  0.5, 0.5, -0.5,
  0.5, -0.5, -0.5,
  0.5, -0.5, 0.5,
  0.5, 0.5, -0.5,
  0.5, 0.5, 0.5,
  0.5, -0.5, 0.5,
};

static float color[] = {
  // FRONT
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  // BACK
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  // TOP
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  // BOTTOM
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  // LEFT
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  // RIGHT
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
};

int main() {
  // Set up vertex buffer object
  uint8_t result[WIDTH][HEIGHT][4]; // RGBA8 32 bits x fragment
  GLuint vbo, program, framebuffer, colorbuffer, depthbuffer;
  GLint loc_position, loc_color;

  #ifdef HOSTDRIVER
  EGL_SETUP();
  #endif

  // Set Up Frame Context
  glGenFramebuffers(1, &framebuffer);
  glGenRenderbuffers(1, &colorbuffer);
  glGenRenderbuffers(1, &depthbuffer);

  glBindRenderbuffer(GL_RENDERBUFFER, colorbuffer);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA4, WIDTH, HEIGHT);

  glBindRenderbuffer(GL_RENDERBUFFER, depthbuffer);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, WIDTH, HEIGHT);

  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorbuffer);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthbuffer);

  glViewport(0, 0, WIDTH, HEIGHT); 

  // Set Up Program
  program = glCreateProgram();
  file_t file;
  #ifndef HOSTDRIVER
  read_file("kernel.pocl", &file);
  glProgramBinary(program, 0, file.data, file.size);
  #else
  GLuint vs, fs;
  GLchar *bin;
  GLint size;
  GLint success;

  vs = glCreateShader(GL_VERTEX_SHADER);
  fs = glCreateShader(GL_FRAGMENT_SHADER);

  read_file("kernel.vs", &file);
  bin = (char*)file.data;
  size = file.size;

  glShaderSource(vs, 1, &bin, &size);
  glCompileShader(vs);
  free(bin);

  glGetShaderiv(vs, GL_COMPILE_STATUS, &success);  
  if (success == GL_FALSE) {
    GLint logSize = 0;
    glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &logSize);

    GLchar* infoLog = (GLchar*) malloc(logSize);
    glGetShaderInfoLog(vs, logSize, &logSize, infoLog);
    printf("Fail compile vs: %s\n", infoLog);
    exit(-1);
  }
  
  read_file("kernel.fs", &file);
  bin = (char*)file.data;
  size = file.size;

  glShaderSource(fs, 1, &bin, &size);
  glCompileShader(fs);

  glGetShaderiv(fs, GL_COMPILE_STATUS, &success);  
  if (success == GL_FALSE) {
    GLint logSize = 0;
    glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &logSize);

    GLchar* infoLog = (GLchar*) malloc(logSize);
    glGetShaderInfoLog(fs, logSize, &logSize, infoLog);
    printf("Fail compile fs: %s\n", infoLog);
    exit(-1);
  }

  free(file.data);

  glAttachShader(program, vs);
  glAttachShader(program, fs);

  glLinkProgram(program);

  glDetachShader(program, vs);
  glDetachShader(program, fs);
  glDeleteShader(vs);
  glDeleteShader(fs);
  #endif
  glUseProgram(program);

  // Set Up Vertex Attributes
  loc_position  = glGetAttribLocation(program, "position");
  loc_color     = glGetAttribLocation(program, "in_color");
  
  // Client memory
  glVertexAttribPointer(loc_position, 3, GL_FLOAT, GL_FALSE, 0, &position);
  glEnableVertexAttribArray(loc_position); 

  // Server memory
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER,sizeof(color), &color, GL_STATIC_DRAW);
  glVertexAttribPointer(loc_color, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
  glEnableVertexAttribArray(loc_color); 

  // Uniforms
  GLint loc_perspective, loc_view, loc_model;
  glm::mat4 perspective(1), view(1), model(1);
  //glDepthRangef(0, 10);
  perspective = glm::perspective((float)M_PI / 4, (float)WIDTH/HEIGHT, 0.1f, 100.f);
  view = glm::lookAt(glm::vec3{-1,-1,-1}, glm::vec3{0,0,0},glm::vec3{0,1,0});
  model = glm::rotate(model, M_PIf/8, glm::vec3{1,1,1});

  loc_perspective = glGetUniformLocation(program, "perspective");
  loc_view = glGetUniformLocation(program, "view");
  loc_model = glGetUniformLocation(program, "model");

  printf("pers: %i , view: %i, model: %i\n", loc_perspective, loc_view, loc_model);

  glUniformMatrix4fv(loc_perspective, 1, GL_FALSE, &perspective[0][0]);
  glUniformMatrix4fv(loc_view, 1, GL_FALSE, &view[0][0]);
  glUniformMatrix4fv(loc_model, 1, GL_FALSE, &model[0][0]);

  glEnable(GL_DEPTH_TEST);
  // Draw
  printf("Color\n");
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  printf("DrawArrays\n");
  glDrawArrays(GL_TRIANGLES, 0, sizeof(position)/sizeof(float[3]));
  glFinish();
  
  #ifdef HOSTDRIVER
  glReadPixels(0,0,WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, result);
  #else
  glReadnPixels(0,0,WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, WIDTH*HEIGHT*4, result);
  #endif
  print_ppm("image.ppm", WIDTH, HEIGHT, (uint8_t*) result);

  return 0; 
}