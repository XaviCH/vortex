#include <stdio.h>
#include <stdlib.h>
#ifdef HOSTDRIVER
#include <EGL/egl.h>
#define EGL_EGLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#else
#include <GLSC2/glsc2.h>
#endif


#ifdef HOSTDRIVER
#define EGL_SETUP() \
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY); \
    EGLint major; \
    EGLint minor; \
    eglInitialize(display, &major, &minor); \
    eglBindAPI(EGL_OPENGL_API); \
    EGLint configAttribs[] = { \
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, \
        EGL_RED_SIZE, 8, \
        EGL_GREEN_SIZE, 8, \
        EGL_BLUE_SIZE, 8, \
        EGL_ALPHA_SIZE, 8, \
        EGL_DEPTH_SIZE, 16, \
        EGL_STENCIL_SIZE, 8, \
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, \
        EGL_NONE \
    }; \
    EGLConfig config; \
    EGLint numConfigs; \
    eglChooseConfig(display, configAttribs, &config, 1, &numConfigs); \
    EGLint contextAttribs[] = { \
        EGL_CONTEXT_MAJOR_VERSION, 3, \
        EGL_CONTEXT_MINOR_VERSION, 0, \
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, \
        EGL_NONE \
    }; \
    EGLContext eglContext = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs); \
    EGLint surfaceAttribs[] = { \
        EGL_WIDTH, static_cast<int>(WIDTH), \
        EGL_HEIGHT, static_cast<int>(HEIGHT), \
        EGL_NONE \
    }; \
    EGLSurface surface = eglCreatePbufferSurface(display, config, surfaceAttribs); \
    eglMakeCurrent(display, surface, surface, eglContext); 

#define EGL_DESTROY() \
    eglDestroyContext(display, eglContext); \
    eglDestroySurface(display, surface); \
    eglTerminate(display);
#endif

static void print_ppm(const char* filename, size_t width, size_t height, const uint8_t *data) {
  FILE *f = fopen(filename, "wb");
  fprintf(f, "P6\n%i %i 255\n", width, height);
  for (int y=0; y<height; y++) {
      for (int x=0; x<width; x++) {
          fputc(data[0], f); 
          fputc(data[1], f); // 0 .. 255
          fputc(data[2], f);  // 0 .. 255
          data += 4;
      }
  }
  fclose(f);
}

typedef struct {
    void* data;
    size_t size;
} file_t;

static int read_file(const char* filename, file_t* file) {
  if (NULL == filename || NULL == file)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load the file.");
    return -1;
  }
  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);
  
  file->data = malloc(fsize);
  file->size = fread(file->data, 1, fsize, fp);
  
  fclose(fp);
  
  return 0;
}