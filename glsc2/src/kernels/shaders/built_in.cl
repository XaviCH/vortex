

float4 mul(float16 mat, float4 vec) {
    
    float4 result = 0;

    for(int i=0; i<16; ++i) {
        result[i%4] += mat[i]*vec[i/4]; 
    }

    return result;
    /*

    float4 *p = (float4*) &mat;
    return (float4){
        dot(p[0], vec),
        dot(p[1], vec),
        dot(p[2], vec),
        dot(p[3], vec),
    };
    */
}

float mod(float x, float y) {
    return x - y * floor(x/y);
}