#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <GLSC2/glsc2.h>
#include <unistd.h> 
#include <string.h>

#include "../common.c"

#define WIDTH 150
#define HEIGHT 100

typedef struct {
  uint8_t* bin;
  size_t size;
} kernel_data_t;

static float position[] = {
   0.0, 1.0, 0.0,
  -1.0,-1.0, 0.0,
   1.0,-1.0, 0.0
};
static float color[] = {
   1.0, 0.0, 0.0,
   0.0, 1.0, 0.0,
   0.0, 0.0, 1.0
};

int main() {
  // Set up vertex buffer object
  uint8_t result[WIDTH][HEIGHT][4]; // RGBA8 32 bits x fragment
  GLuint vbo, program, framebuffer, colorbuffer;
  GLint loc_position, loc_color;

  // Set Up Frame Context
  glGenFramebuffers(1, &framebuffer);
  glGenRenderbuffers(1, &colorbuffer);

  glBindRenderbuffer(GL_RENDERBUFFER, colorbuffer);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA4, WIDTH, HEIGHT);

  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorbuffer);

  glViewport(0, 0, WIDTH, HEIGHT); 

  // Set Up Program
  kernel_data_t kernel_data;
  read_kernel_file("kernel.pocl", &kernel_data.bin, &kernel_data.size);
  program = glCreateProgram();
  glProgramBinary(program, 0, kernel_data.bin, kernel_data.size);
  glUseProgram(program);

  // Set Up Vertex Attributes
  loc_position  = glGetAttribLocation(program, "position");
  loc_color     = glGetAttribLocation(program, "in_color");
  
  glVertexAttribPointer(loc_position, 3, GL_FLOAT, GL_FALSE, 0, &position);
  glEnableVertexAttribArray(loc_position); 

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER,sizeof(color), &color, GL_STATIC_DRAW);
  glVertexAttribPointer(loc_color, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
  glEnableVertexAttribArray(loc_color); 

  // Draw
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  glFinish();
  glReadnPixels(0,0,WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, WIDTH*HEIGHT*4, result);

  printPPM("image.ppm", WIDTH, HEIGHT, (uint8_t*) result);

  return 0; 
}