#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include "esUtil.h"
#include "../common.h"
#define BENCHMARKING

#ifndef BENCHMARKING

#define GL_ERROR() { \
	GLenum error=glGetError(); \
	switch(error) \
	{\
		case GL_NO_ERROR: \
			break;\
		case GL_INVALID_ENUM: \
			printf("GL_INVALID_ENUM at %d:%s\n", __LINE__, __FILE__);\
			exit(-1);\
			break;\
		case GL_INVALID_VALUE: \
			printf("GL_INVALID_VALUE at %d:%s\n", __LINE__, __FILE__);\
			exit(-1);\
			break;\
		case GL_INVALID_OPERATION: \
			printf("GL_INVALID_OPERATION at %d:%s\n", __LINE__, __FILE__);\
			exit(-1);\
			break;\
		case GL_INVALID_FRAMEBUFFER_OPERATION: \
			printf("GL_INVALID_FRAMEBUFFER_OPERATION at %d:%s\n", __LINE__, __FILE__);\
			exit(-1);\
			break;\
		case GL_OUT_OF_MEMORY: \
			printf("GL_OUT_OF_MEMORY at %d:%s\n", __LINE__, __FILE__);\
			exit(-1);\
			break;\
	}\
}

#else

#define GL_ERROR()  

#endif /*BENCHMARKING*/

unsigned char out[4*2048*2048];
unsigned char texture_d[4*2048*2048];
GLuint texture[2];

#define DEBUG

unsigned long frame_n=0;
unsigned long period_n=0;
#define FRAMEBUFFER_SIZE (16.0)

#define xstr(s) str(s)
#define str(s) #s
#ifdef DEBUG
	#define REVERT_IMAGE_FOR_DISPLAY
#endif
#define RECT
#define BYTE_TEXTURE 0
#define SHIFT_32 0
#define U_UNION_32 0
#define U_32_NO_CONVERSION 0
#define S_UNION_32 0
#define S_32_NO_CONVERSION 0
#define FP_32 1
#define NO_CPU_ENCODING 0

#define OUTPUT_TEXTURE_0 3 

#if SHIFT_32 || U_UNION_32 || U_32_NO_CONVERSION
	#define UINT_TEXTURE 0
	#define UINT_TEXTURE_VECTORISED 1
#elif S_UNION_32 || S_32_NO_CONVERSION
	#define SINT_TEXTURE 0
	#define SINT_TEXTURE_VECTORISED 1
#endif

//#define PRINT_COMPONENT
#ifdef PRINT_COMPONENT
	#define READ_COMPONENT 2
#endif

#define MIN(x,y) ((x>y)?y:x)
#define MAX(x,y) ((x>y)?x:y)

GLuint texture_target[2];
GLuint FBO[2];

struct timeval t1, t2;
struct timezone tz;

typedef struct
{
   // Handle to a program object
   GLuint programObject;

} UserData;

///
// Initialize the shader and program object
//
int Init ( ESContext *esContext )
{
   esContext->userData = malloc(sizeof(UserData));

   UserData *userData = (UserData*) esContext->userData;
   
   GLuint programObject;
   GLint linked;

   file_t file;
   read_file("./kernel.ocl", &file);

   // Create the program object
   programObject = glCreateProgram ( );
   
   if ( programObject == 0 )
      return 0;

   glProgramBinary(programObject, 0, file.data, file.size);

   // Bind vPosition to attribute 0   
   // glBindAttribLocation ( programObject, 0, "vPosition" );
   // GL_ERROR();

   // Link the program
   // glLinkProgram ( programObject );

   // Check the link status
   glGetProgramiv ( programObject, GL_LINK_STATUS, &linked );

   if ( !linked ) 
   {
      GLint infoLen = 0;

      printf ( "Error linking program.\n%s\n");            

      return GL_FALSE;
   }

   GLint texcoordLoc = glGetAttribLocation( programObject, "vPosition" );
   GL_ERROR();
   printf("vPosition location:%d\n", texcoordLoc);

   texcoordLoc = glGetAttribLocation( programObject, "texCoord0" );
   GL_ERROR();
   printf("texCoord0 location:%d\n", texcoordLoc);

   texcoordLoc = glGetAttribLocation( programObject, "texCoord1" );
   GL_ERROR();
   printf("texCoord1 location:%d\n", texcoordLoc);


   // Store the program object
   userData->programObject = programObject;

   glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
   return GL_TRUE;
}

///
// Draw a triangle using the shader pair created in Init()
//
void Draw ( ESContext *esContext )
{
   int i, j;
   UserData *userData = (UserData*) esContext->userData;

   //bottom-left triangle
   GLfloat vVertices[] = { -1.0f,  1.0f, 0.0f, 
                           -1.0f, -1.0f, 0.0f,
                            1.0f, -1.0f, 0.0f /*};

   //upper-right triangle
   GLfloat vVertices[] = {*/, -1.0f,  1.0f, 0.0f, 
                            1.0f,  1.0f, 0.0f,
                            1.0f, -1.0f, 0.0f };

   //bottom-left triangle
   GLfloat uvPtr[] =     {  0.0f,  1.0f, 0.0f, 
                            0.0f,  0.0f, 0.0f,
                            1.0f,  0.0f, 0.0f /*};

   //upper-right triangle
   GLfloat uvPtr[] =     {*/,  0.0f,  1.0f, 0.0f, 
                            1.0f,  1.0f, 0.0f,
                            1.0f,  0.0f, 0.0f };

   glGenTextures(2, texture);
   GL_ERROR();

   int I=0;
   for(I=0; I<2; I++){
   glActiveTexture(GL_TEXTURE0+I);
   glBindTexture(GL_TEXTURE_2D, texture[0+I]);
   GL_ERROR();

   //we create a texture with the same size as the framebuffer
   for(i=0; i<esContext->height; i++)
   {
   	for(j=0; j<esContext->width; j++)
	{
#if NO_CPU_ENCODING  
		//just write a 4 byte quantity without conversion
		((unsigned int*)texture_d)[i*esContext->width + j] = j;
#elif BYTE_TEXTURE
		//Just pass the values as they are, because they fit in a byte
		texture_d[i*4*esContext->width + 4*j +0]=0;
		texture_d[i*4*esContext->width + 4*j +1]=0;
		texture_d[i*4*esContext->width + 4*j +2]=0;
		texture_d[i*4*esContext->width + 4*j +3]=0;
#elif SHIFT_32 
		//split a 32 bit integer value to 4 components
		texture_d[i*4*esContext->width + 4*j +0]=(j)&0xFF;
		texture_d[i*4*esContext->width + 4*j +1]=(j>>8)&0xFF;
		texture_d[i*4*esContext->width + 4*j +2]=(j>>16)&0xFF;
		texture_d[i*4*esContext->width + 4*j +3]=(j>>24)&0xFF;
#elif U_UNION_32 || S_UNION_32
		union {	
			unsigned int i;
			int s;
			struct{
			  char x;
			  char y;
			  char z;
			  char w;};
		}__attribute__ ((packed)) u;
	#if U_UNION_32
		//From the number 2^24 +1 up to 2^32-1 only the powers of two are represented  
		u.i= (j+125)<<24;
//printf("i.x=%u i.y=%u i.z=%u i.w=%u\n", u.x, u.y, u.z, u.w);
	#elif S_UNION_32
		//with the signed version between -1 and (some value), only the powers of 2 can be represented  
		//the same with the numbers between 2^24 + 1 up to 2^31-1
		u.s= -(j<<8);
		#ifdef DEBUG
		printf("i.x=%u i.y=%u i.z=%u i.w=%u\n", u.x, u.y, u.z, u.w);
		#endif
	#endif

		texture_d[i*4*esContext->width + 4*j +0] = u.x;
		texture_d[i*4*esContext->width + 4*j +1] = u.y;
		texture_d[i*4*esContext->width + 4*j +2] = u.z;
		texture_d[i*4*esContext->width + 4*j +3] = u.w;
#elif U_32_NO_CONVERSION
		((unsigned int*)texture_d)[i*esContext->width + j] = j;
#elif S_32_NO_CONVERSION
		((int*)texture_d)[i*esContext->width + j] = -(j << 24);
#elif FP_32
		union {	
			float f;
			struct{
			  unsigned int mant:23;
			  unsigned int exp:8;
			  unsigned int sign:1;
			};
			//better uint32_t here
			unsigned int u32;
		}__attribute__ ((packed)) f;
		assert(sizeof(float) == sizeof(f));

		f.f= j;

		//copy the entire number
		*((float*) (texture_d + i*4*esContext->width + 4*j)) = f.f;
		//so now we only have to swap a bit between those two bytes
		texture_d[i*4*esContext->width + 4*j +3] = f.exp;
		texture_d[i*4*esContext->width + 4*j +2] = (f.sign<<7) | (f.mant >> 16);
	#ifdef PRINT_COMPONENT
		#if READ_COMPONENT==0
		printf("%d ", f.exp);
		#elif READ_COMPONENT==1
//		printf("sign:%d\n", f.sign);
		printf("%d ", (unsigned char)(texture_d[i*4*esContext->width + 4*j +1]));
		#elif READ_COMPONENT==2
		printf("%d ", (unsigned char)(texture_d[i*4*esContext->width + 4*j +2]));
		#elif READ_COMPONENT==3
		printf("%d ", texture_d[i*4*esContext->width + 4*j +3]);
		#endif
	#endif
#endif /*FP_32*/
	}
	#ifdef PRINT_COMPONENT
	printf("\n");
	#endif
   }
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, esContext->width, esContext->height);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, esContext->width, esContext->height, GL_RGBA, GL_UNSIGNED_BYTE, texture_d);
   }


printf("allocate render texture\n ");
   glGenTextures(2, texture_target);
   GL_ERROR();

   glGenFramebuffers(2, FBO);
   GL_ERROR();

   int K=I;
   printf("About to setup the texture to be rendered, K=%d\n", K);
   glActiveTexture(GL_TEXTURE0+K+1);
   GL_ERROR();
   glBindTexture(GL_TEXTURE_2D, texture_target[K-I]);
   GL_ERROR();
printf("bind render texture %d to unit %d\n", K-I, K+1);

   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   GL_ERROR();
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   GL_ERROR();
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   GL_ERROR();
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   GL_ERROR();

   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, esContext->width, esContext->height);
   GL_ERROR();
   glBindFramebuffer(GL_FRAMEBUFFER, FBO[K-I]);
   GL_ERROR();

   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_target[(K-I)?1:0], 0);
   GL_ERROR();

   // Set the viewport
   glViewport ( 0, 0, esContext->width, esContext->height );
   GL_ERROR();

   glClear ( GL_COLOR_BUFFER_BIT );
   GL_ERROR();

   // Use the program object
   glUseProgram ( userData->programObject );
   GL_ERROR();

   // Load the vertex data
   glVertexAttribPointer ( 0, 3, GL_FLOAT, GL_FALSE, 0, vVertices );
   GL_ERROR();

   glEnableVertexAttribArray ( 0 );
   GL_ERROR();

#define texture_setup(num) do{\
   GLint uvAttrib##num = glGetAttribLocation( userData->programObject, "texCoord"#num ); \
GL_ERROR(); \
printf("textureCoord%d location:%d\n", num, uvAttrib##num); \
glVertexAttribPointer(uvAttrib##num, 3, GL_FLOAT, GL_FALSE, 0, uvPtr); \
GL_ERROR(); \
    glEnableVertexAttribArray(uvAttrib##num); \
GL_ERROR(); \
   GLint texUnitLoc  = glGetUniformLocation( userData->programObject, "textureUnit"#num );\
printf("texture%d location:%d\n", num, texUnitLoc);\
   glUniform1i(texUnitLoc, num);\
GL_ERROR(); \
}while(0);

   texture_setup(0);
   texture_setup(1);

   glActiveTexture(GL_TEXTURE0);
   GL_ERROR();
   glBindTexture(GL_TEXTURE_2D, texture[0]);
   GL_ERROR();

   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, texture[1]);
   GL_ERROR();

   glDrawArrays ( GL_TRIANGLES, 0, 6 );
   GL_ERROR();

glReadnPixels(0,
 	0,
	esContext->width,
	esContext->height,
 	GL_RGBA,
 	GL_UNSIGNED_BYTE,
      esContext->width*esContext->height*4, // sizeof(out),
 	out);
GL_ERROR();
#ifdef DEBUG
//The image is flipped, (0,0) is on top
#ifndef REVERT_IMAGE_FOR_DISPLAY
for(i=0; i<esContext->height; i++)
#else
for(i=esContext->height-1; i>=0; i--)
#endif /*REVERT_IMAGE_FOR_DISPLAY*/
{
	for(j=0; j<esContext->width; j++)
	{	
#ifdef PRINT_COMPONENT
		//we just print the R component
		printf("%u ", out[i*4*esContext->width + 4*j + READ_COMPONENT]);
#else
	#if UINT_TEXTURE || UINT_TEXTURE_VECTORISED
		printf("%u ", ((unsigned int*)out)[i*esContext->width + j]);

		assert(((unsigned int*)out)[i*esContext->width + j] == (j));
	#elif BYTE_TEXTURE
		printf("%u ", out[4*i*esContext->width + 4*j]);
	#elif SINT_TEXTURE || SINT_TEXTURE_VECTORISED
		printf("%d ", ((int*)out)[i*esContext->width + j]);
		//it is 25 because we add -(j<<24) + -(j<<24)
		assert(((int*)out)[i*esContext->width + j] == -(j<<25));
	#elif FP_32
		union {	
			float f;
			struct{
			  unsigned int mant:23;
			  unsigned int exp:8;
			  unsigned int sign:1;
			};
			//better use uint32_t
			unsigned int u32;
		}__attribute__ ((packed)) f;

		//copy the entire number
		f.f = *((float*) (out + i*4*esContext->width + 4*j));

		assert(sizeof(float) == sizeof(f));
#ifndef BENCHMARKING
		//only those fields need to be corrected
		f.exp=out[i*4*esContext->width + 4*j +3];
//printf("exponent:%d\n", f.exp);
		register byte2 = out[i*4*esContext->width + 4*j +2];
		f.mant = (byte2 & 0x7F) << 16 ;
//printf("mant:%d\n", (out[i*4*esContext->width + 4*j +1]& 0x7F));
//printf("mant:%d\n", f.mant);
		f.sign= (byte2 & 0x80)  >> 7;
//printf("sign:%d\n", f.sign);
//printf("mant:%d\n", f.mant);
//		printf("%.1f ", f.f);
#else
		//better uint32_t here
		register u32=*((unsigned int*) (out + i*4*esContext->width + 4*j));
		f.u32 = ((u32 << 8 ) & 0x80000000) |  (u32 >> 1) & 0x7F800000 | (u32 & 0x7FFFFF);
		printf("%.1f ", f.f );
#endif

		//assert(f.f == (float)j);
	#endif
#endif
	}
//#ifndef BENCHMARKING
	printf("\n");
//#endif
}
#endif
printf("read pixels, exiting\n");
           gettimeofday(&t2, &tz);
           float time = (float)(t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1e-6);

           float flops=(float)FRAMEBUFFER_SIZE*FRAMEBUFFER_SIZE;
           printf("Computation time: %f seconds for %f MFlops\n", time, (float)(flops*1e-6));
           printf("%f MFlops/s\n", (float)(flops)/(time*1e6));

	   exit(0);
}

int main ( int argc, char *argv[] )
{
   ESContext esContext;
   UserData  userData;

   gettimeofday(&t1, &tz);

   //esInitContext ( &esContext );
   esContext.userData = &userData;
   esContext.width = (int)FRAMEBUFFER_SIZE;
   esContext.height = (int)FRAMEBUFFER_SIZE;
   //esCreateWindow ( &esContext, "Hello Vector Addition GPGPU", (int)FRAMEBUFFER_SIZE, (int)FRAMEBUFFER_SIZE, ES_WINDOW_RGB|ES_WINDOW_ALPHA);

   if ( !Init ( &esContext ) )
      return 0;

   Draw(&esContext);

}
