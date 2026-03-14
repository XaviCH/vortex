#ifndef COMPILER_COMMON_H
#define COMPILER_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include <config.device.h>

static int read_file(const char* filename, uint8_t** data, size_t* size) {
    if (NULL == filename || NULL == data || 0 == size)
        return -1;

    FILE* fp = fopen(filename, "r");
    if (NULL == fp) {
        fprintf(stderr, "Failed to read file %s.", filename);
        return -1;
    }
    fseek(fp , 0 , SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);

    *data = (uint8_t*)malloc(fsize);
    *size = fread(*data, 1, fsize, fp);
  
    fclose(fp);
  
    return 0;
}

#endif // COMPILER_COMMON_H