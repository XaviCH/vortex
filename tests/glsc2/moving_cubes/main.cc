#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <signal.h>
#include <chrono>

#include <unistd.h>
#include <string.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "../common.h"
#include <tests/glsc2/parameters.h>

static float position_fan_array[] {
   0, 0, 0,
  -1, 0, 0,
   0, 1, 0,
   1, 0, 0,
   0,-1, 0,
  -1,-0, 0,
};

static float color_fan_array[] {
  1, 1, 1,
  1, 0, 0,
  0, 1, 0,
  0, 0, 1,
  0, 0, 0,
  1, 0, 0,
};

static int fan_index[] {
  0, 1, 2, 3, 4, 1
};

GLushort INDEX_BUFFER_TRIANGLES[] = {
    0, 1, 3,
    3, 1, 2,
    1, 5, 2,
    2, 5, 6,
    5, 4, 6, 6, 4, 7,
    4, 0, 7, 7, 0, 3,
    3, 2, 7, 7, 2, 6,
    4, 5, 0, 0, 5, 1,
};

GLushort INDEX_BUFFER_TRIANGLE_FAN[] = {
    0, 1, 3, 3, 1, 2,
    1, 5, 2, 2, 5, 6,
    5, 4, 6, 6, 4, 7,
    4, 0, 7, 7, 0, 3,
    3, 2, 7, 7, 2, 6,
    4, 5, 0, 0, 5, 1,
};

GLushort INDEX_BUFFER_TRIANGLE_STRIP[] = {
    0, 1, 3, 3, 1, 2,
    1, 5, 2, 2, 5, 6,
    5, 4, 6, 6, 4, 7,
    4, 0, 7, 7, 0, 3,
    3, 2, 7, 7, 2, 6,
    4, 5, 0, 0, 5, 1,
};

float position_array[] = {
    -1, -1, -1,
    1, -1, -1,
    1, 1, -1, 
    -1, 1, -1,

    -1, -1, 1, 
    1, -1, 1, 
    1, 1, 1, 
    -1, 1, 1,
};

float color_array[] {
  1, 0, 0,
  0, 1, 0,
  0, 0, 1,
  1, 1, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 1,
  0, 0, 0
};

static float texCoord[] = {
  0, 0,
  0, 1,
  1, 0,
  1, 1,
  0, 1,
  0, 0,
  1, 0,
  1, 1,
};


typedef struct {
  glm::vec3 position, speed, axis;
  float angle;
} cube_world_data_t;

cube_world_data_t* cube_world_data;

EGLDisplay display;
EGLSurface surface;

GLint program;
GLint location_position, location_color;
GLint location_model, location_view, location_perspective;

volatile sig_atomic_t cntrl_c = 0;

void catch_cntrl_c(int sig){ // can be called asynchronously
  cntrl_c = 1; // set flag
};

static void init_cube_data()
{
  cube_world_data = (cube_world_data_t*) malloc(sizeof(cube_world_data_t[test_get_size()]));

  for(size_t cube_id = 0; cube_id < test_get_size(); ++cube_id)
  {
    cube_world_data_t& cube = cube_world_data[cube_id];

    cube.position = glm::vec3(
      rand() % 10 - 5,
      rand() % 10 - 5,
      rand() % 10 + 10
    );

    cube.speed = glm::vec3(
      (float) rand() / (RAND_MAX/2) - 1.0f, 
      (float) rand() / (RAND_MAX/2) - 1.0f, 
      (float) rand() / (RAND_MAX/2) - 1.0f 
    );

    cube.axis = glm::normalize(glm::vec3(
      rand(), 
      rand(), 
      rand()
    ));
  }
}

const glm::mat4 perspective = glm::perspective((float)M_PI / 4, (float)test_get_width()/test_get_height(), 0.1f, 100.f);
const glm::mat4 view = glm::lookAt(glm::vec3{0, 0, -2}, glm::vec3{0,0,0},glm::vec3{0,1,0});

static void update_cube_data(float time)
{
  for(size_t cube_id = 0; cube_id < test_get_size(); ++cube_id)
  {
    cube_world_data_t& cube = cube_world_data[cube_id];

    glm::vec4 tmp_position_4d = perspective * view * glm::vec4(cube.position + cube.speed*time, 1.0f);
    glm::vec3 tmp_position = glm::vec3(tmp_position_4d) / tmp_position_4d.w;

    if (glm::abs(tmp_position.x) > 1.0f || 
        glm::abs(tmp_position.y) > 1.0f ||
        glm::abs(tmp_position.z) > 1.0f)
    {
      cube.speed *= -1;
    }

    cube.position += cube.speed*time;
    cube.angle += glm::length(cube.speed)*time;
  }
}

static void draw_cubes()
{
  glEnable(GL_DEPTH_TEST);

  glVertexAttribPointer(location_position, 3, GL_FLOAT, GL_FALSE, 0, &position_array);
  glEnableVertexAttribArray(location_position);
  glVertexAttribPointer(location_color, 3, GL_FLOAT, GL_FALSE, 0, &color_array);
  glEnableVertexAttribArray(location_color);

  glUniformMatrix4fv(location_perspective, 1, GL_FALSE, &perspective[0][0]);
  glUniformMatrix4fv(location_view, 1, GL_FALSE, &view[0][0]);

  for (int cube_id = 0; cube_id < test_get_size(); ++cube_id) 
  {
    cube_world_data_t& cube = cube_world_data[cube_id];

    glm::mat4 model = glm::mat4(1);
    model = glm::translate(model, cube.position);
    model = glm::rotate(model, cube.angle, cube.axis);

    glUniformMatrix4fv(location_model, 1, GL_FALSE, &model[0][0]);

    const size_t count = sizeof(INDEX_BUFFER_TRIANGLES)/sizeof(INDEX_BUFFER_TRIANGLES[0]);
    glDrawRangeElements(GL_TRIANGLES, 0, sizeof(position_array)/sizeof(position_array[0])/3, count, GL_UNSIGNED_SHORT, &INDEX_BUFFER_TRIANGLES);
  }
}

static void egl_setup(EGLDisplay* display, EGLSurface* surface)
{
  *display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  
  EGLint major, minor;
  eglInitialize(*display, &major, &minor);

  eglBindAPI(EGL_OPENGL_API);
  EGLint configAttribs[] = {
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_RED_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE, 8,
    EGL_ALPHA_SIZE, 8,
    EGL_DEPTH_SIZE, 16,
    EGL_STENCIL_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_NONE
  };

  EGLConfig config;
  EGLint numConfigs;
  eglChooseConfig(*display, configAttribs, &config, 1, &numConfigs);

  EGLint contextAttribs[] = {
      EGL_CONTEXT_MAJOR_VERSION, 3,
      EGL_CONTEXT_MINOR_VERSION, 0,
      EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
      EGL_NONE
  };
  EGLContext eglContext = eglCreateContext(*display, config, EGL_NO_CONTEXT, contextAttribs);

  EGLint surfaceAttribs[] = {
      EGL_WIDTH, static_cast<int>(test_get_width()),
      EGL_HEIGHT, static_cast<int>(test_get_height()),
      EGL_NONE 
  };
  
  *surface = eglCreatePbufferSurface(*display, config, surfaceAttribs);

  eglMakeCurrent(*display, *surface, *surface, eglContext); 
}

static void offline_setup(GLuint *framebuffer, GLuint *colorbuffer, GLuint *depthbuffer)
{
  // Set Up Frame Context
  glGenFramebuffers(1, framebuffer);
  glGenTextures(1, colorbuffer);
  glGenRenderbuffers(1, depthbuffer);

  glBindTexture(GL_TEXTURE_2D, *colorbuffer);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, test_get_width(), test_get_height());

  glBindRenderbuffer(GL_RENDERBUFFER, *depthbuffer);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, test_get_width(), test_get_height());

  glBindFramebuffer(GL_FRAMEBUFFER, *framebuffer);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *colorbuffer, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, *depthbuffer);
}

int main() 
{
  signal(SIGINT, catch_cntrl_c);
  
  GLuint framebuffer, colorbuffer, depthbuffer;
  // Set Up Window
  egl_setup(&display, &surface);
  offline_setup(&framebuffer, &colorbuffer, &depthbuffer);

  glViewport(0, 0, test_get_width(), test_get_height()); 

  // Set Up Program
  file_t file;
  read_file("kernel.cl.o", &file);
  
  program = glCreateProgram();
  glProgramBinary(program, 0, file.data, file.size);
  glUseProgram(program);

  // Set Up Vertex Attributes
  location_position  = glGetAttribLocation(program, "position");
  location_color     = glGetAttribLocation(program, "color");

  if (location_position == -1 || location_color == -1)
  {
    fprintf(stderr, "ERROR: attribute locations: position=%d, color=%d\n", location_position, location_color);
    exit(1);
  }

  // Uniforms
  location_perspective  = glGetUniformLocation(program, "perspective");
  location_view         = glGetUniformLocation(program, "view");
  location_model        = glGetUniformLocation(program, "model");
  
  if (location_perspective == -1 || location_view == -1 || location_model == -1)
  {
    fprintf(stderr, "ERROR: uniform locations: perspective=%d, view=%d, model=%d\n", location_perspective, location_view, location_model);
    exit(1);
  }

  init_cube_data();
  
  auto begin = std::chrono::high_resolution_clock::now();  
  auto last_goal_clock = begin;

  int last_goal_frame_count = 0;
  int seconds_goal = 1;

  auto start = begin;
  int frame_count = 0;
  while (cntrl_c == 0) {
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_microsenconds = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
    update_cube_data(elapsed_microsenconds / 1e6);
    
    glClearColor(0.2f, 0.2f, 0.2f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    draw_cubes();

    if (not test_is_offline_rendering()) {
      eglSwapBuffers(display, surface);
    }
    
    frame_count += 1;
    last_goal_frame_count += 1;

    if (seconds_goal <= std::chrono::duration_cast<std::chrono::seconds>(end-start).count()) {
      double elapsed_microsenconds = std::chrono::duration_cast<std::chrono::microseconds>(end-last_goal_clock).count();

      printf("PERF: fps=%.1f.\n", last_goal_frame_count / (elapsed_microsenconds / 1e6));
      seconds_goal += 1;
      last_goal_clock = end;
      last_goal_frame_count = 0;
    }

    begin = end;
  }
  // clean out line
  glFinish();

  printf("\n");
  
  auto end = std::chrono::high_resolution_clock::now();
  double total_time_microseconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

  printf("PERF: fps=%.1f.\n", frame_count / (total_time_microseconds / 1e6));
  printf("PERF: frame_time=%.3f ms.\n", total_time_microseconds / frame_count / 1e3);

  uint8_t* result = (uint8_t*) malloc(sizeof(uint8_t[test_get_width()][test_get_height()][4]));

  #ifdef C_OPENGL_HOST
  glReadPixels(0,0,test_get_width(), test_get_height(), GL_RGBA, GL_UNSIGNED_BYTE, result);
  #else
  glReadnPixels(0,0,test_get_width(), test_get_height(), GL_RGBA, GL_UNSIGNED_BYTE, test_get_width()*test_get_height()*4, result);
  #endif

  print_ppm("image.ppm", test_get_width(), test_get_height(), (uint8_t*) result);
  free(result);
  
  EGL_DESTROY();

  return 0; 
}