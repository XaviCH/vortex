#pragma once
#include <stdio.h>
#include <stdlib.h>

#ifdef HOSTGPU
typedef struct {
    void* data;
    size_t size;
} file_t;

static int read_file(const char* filename, file_t* file) {
  if (NULL == filename || NULL == file)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load the file. %s\n", filename);
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
#endif

