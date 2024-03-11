#include <EGL/egl.h>

EGLDisplay eglGetDisplay(EGLNativeDisplayType
display_id){
    if (display_id == EGL_DEFAULT_DISPLAY){
        FILE* file_display = fopen("out","w");
        return (void *) file_display;
    }
    return EGL_NO_DISPLAY;
}

EGLBoolean eglInitialize(EGLDisplay dpy, EGLint
*major, EGLint *minor){
    if (!dpy) return EGL_FALSE;
    
    //iniciar context aqui? (quien sabe)
    

    if(major) *major=1;
    if (minor) *minor=5;
    return EGL_TRUE;
}