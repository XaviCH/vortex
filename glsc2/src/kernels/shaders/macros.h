/**
 * Generic macros
 * 
 * TODO: Simplify macros to more generic ones.
 */

#ifndef KERNELS_SHADERS_MACROS_H
#define KERNELS_SHADERS_MACROS_H

// A preprocessor argument counter
#define COUNT(...) COUNT_I(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1,)
#define COUNT_I(_9,_8,_7,_6,_5,_4,_3,_2,_1,X,...) X
// Preprocessor paster
#define GLUE(A,B) GLUE_I(A,B)
#define GLUE_I(A,B) A##B
// chained caller
#define NAMED_VALUES(...) GLUE(NAMED_VALUES_,COUNT(__VA_ARGS__))(__VA_ARGS__)
// chain
#define NAMED_VALUES_1(a) #a,a
#define NAMED_VALUES_2(a,...) #a,a,NAMED_VALUES_1(__VA_ARGS__)
#define NAMED_VALUES_3(a,...) #a,a,NAMED_VALUES_2(__VA_ARGS__)
#define NAMED_VALUES_4(a,...) #a,a,NAMED_VALUES_3(__VA_ARGS__)
#define NAMED_VALUES_5(a,...) #a,a,NAMED_VALUES_4(__VA_ARGS__)
#define NAMED_VALUES_6(a,...) #a,a,NAMED_VALUES_5(__VA_ARGS__)
#define NAMED_VALUES_7(a,...) #a,a,NAMED_VALUES_6(__VA_ARGS__)
#define NAMED_VALUES_8(a,...) #a,a,NAMED_VALUES_7(__VA_ARGS__)
#define NAMED_VALUES_9(a,...) #a,a,NAMED_VALUES_8(__VA_ARGS__)

// 
#define JOIN_CHAIN(value, ...) GLUE(JOIN_CHAIN_,COUNT(__VA_ARGS__))(value, __VA_ARGS__)
// chain
#define JOIN_CHAIN_1(value, a) value##a
#define JOIN_CHAIN_2(value, a,...) value##a, JOIN_CHAIN_1(value, __VA_ARGS__)
#define JOIN_CHAIN_3(value, a,...) value##a, JOIN_CHAIN_2(value, __VA_ARGS__)
#define JOIN_CHAIN_4(value, a,...) value##a, JOIN_CHAIN_3(value, __VA_ARGS__)
#define JOIN_CHAIN_5(value, a,...) value##a, JOIN_CHAIN_4(value, __VA_ARGS__)
#define JOIN_CHAIN_6(value, a,...) value##a, JOIN_CHAIN_5(value, __VA_ARGS__)
#define JOIN_CHAIN_7(value, a,...) value##a, JOIN_CHAIN_6(value, __VA_ARGS__)
#define JOIN_CHAIN_8(value, a,...) value##a, JOIN_CHAIN_7(value, __VA_ARGS__)
#define JOIN_CHAIN_9(value, a,...) value##a, JOIN_CHAIN_8(value, __VA_ARGS__)

// chained caller
#define STRUCT_CHAIN(type, ...) GLUE(STRUCT_CHAIN_,COUNT(__VA_ARGS__))(type, __VA_ARGS__)
// chain
#define STRUCT_CHAIN_1(type, a) type a;
#define STRUCT_CHAIN_2(type, a,...) type a; STRUCT_CHAIN_1(type, __VA_ARGS__)
#define STRUCT_CHAIN_3(type, a,...) type a; STRUCT_CHAIN_2(type, __VA_ARGS__)
#define STRUCT_CHAIN_4(type, a,...) type a; STRUCT_CHAIN_3(type, __VA_ARGS__)
#define STRUCT_CHAIN_5(type, a,...) type a; STRUCT_CHAIN_4(type, __VA_ARGS__)
#define STRUCT_CHAIN_6(type, a,...) type a; STRUCT_CHAIN_5(type, __VA_ARGS__)
#define STRUCT_CHAIN_7(type, a,...) type a; STRUCT_CHAIN_6(type, __VA_ARGS__)
#define STRUCT_CHAIN_8(type, a,...) type a; STRUCT_CHAIN_7(type, __VA_ARGS__)
#define STRUCT_CHAIN_9(type, a,...) type a; STRUCT_CHAIN_8(type, __VA_ARGS__)

// add comma and type to all parameters

#define COMMA_CHAIN(type, ...) GLUE(COMMA_CHAIN_,COUNT(__VA_ARGS__))(type, __VA_ARGS__)
// chain
#define COMMA_CHAIN_1(type, a) type a,
#define COMMA_CHAIN_2(type, a,...) type a, COMMA_CHAIN_1(type, __VA_ARGS__)
#define COMMA_CHAIN_3(type, a,...) type a, COMMA_CHAIN_2(type, __VA_ARGS__)
#define COMMA_CHAIN_4(type, a,...) type a, COMMA_CHAIN_3(type, __VA_ARGS__)
#define COMMA_CHAIN_5(type, a,...) type a, COMMA_CHAIN_4(type, __VA_ARGS__)
#define COMMA_CHAIN_6(type, a,...) type a, COMMA_CHAIN_5(type, __VA_ARGS__)
#define COMMA_CHAIN_7(type, a,...) type a, COMMA_CHAIN_6(type, __VA_ARGS__)
#define COMMA_CHAIN_8(type, a,...) type a, COMMA_CHAIN_7(type, __VA_ARGS__)
#define COMMA_CHAIN_9(type, a,...) type a, COMMA_CHAIN_8(type, __VA_ARGS__)

#define PARAM_SAMPLER_CHAIN(type0, type1, ...) GLUE(PARAM_SAMPLER_CHAIN_,COUNT(__VA_ARGS__))(type0, type1, __VA_ARGS__)
#define PARAM_SAMPLER_CHAIN_1(type0, type1, a) type0 a, type1 gl_image_##a,
#define PARAM_SAMPLER_CHAIN_2(type0, type1, a,...) type0 a, type1 gl_image_##a, PARAM_SAMPLER_CHAIN_1(type0, type1, __VA_ARGS__)
#define PARAM_SAMPLER_CHAIN_3(type0, type1, a,...) type0 a, type1 gl_image_##a, PARAM_SAMPLER_CHAIN_2(type0, type1, __VA_ARGS__)
#define PARAM_SAMPLER_CHAIN_4(type0, type1, a,...) type0 a, type1 gl_image_##a, PARAM_SAMPLER_CHAIN_3(type0, type1, __VA_ARGS__)
#define PARAM_SAMPLER_CHAIN_5(type0, type1, a,...) type0 a, type1 gl_image_##a, PARAM_SAMPLER_CHAIN_4(type0, type1, __VA_ARGS__)
#define PARAM_SAMPLER_CHAIN_6(type0, type1, a,...) type0 a, type1 gl_image_##a, PARAM_SAMPLER_CHAIN_5(type0, type1, __VA_ARGS__)
#define PARAM_SAMPLER_CHAIN_7(type0, type1, a,...) type0 a, type1 gl_image_##a, PARAM_SAMPLER_CHAIN_6(type0, type1, __VA_ARGS__)
#define PARAM_SAMPLER_CHAIN_8(type0, type1, a,...) type0 a, type1 gl_image_##a, PARAM_SAMPLER_CHAIN_7(type0, type1, __VA_ARGS__)
#define PARAM_SAMPLER_CHAIN_9(type0, type1, a,...) type0 a, type1 gl_image_##a, PARAM_SAMPLER_CHAIN_8(type0, type1, __VA_ARGS__)

#define EQ_STRUCT_CHAIN(name, ...) GLUE(EQ_STRUCT_CHAIN_,COUNT(__VA_ARGS__))(name, __VA_ARGS__)
// chain
#define EQ_STRUCT_CHAIN_1(name, a) name.a=a;
#define EQ_STRUCT_CHAIN_2(name, a,...) name.a=a; EQ_STRUCT_CHAIN_1(name, __VA_ARGS__)
#define EQ_STRUCT_CHAIN_3(name, a,...) name.a=a; EQ_STRUCT_CHAIN_2(name, __VA_ARGS__)
#define EQ_STRUCT_CHAIN_4(name, a,...) name.a=a; EQ_STRUCT_CHAIN_3(name, __VA_ARGS__)
#define EQ_STRUCT_CHAIN_5(name, a,...) name.a=a; EQ_STRUCT_CHAIN_4(name, __VA_ARGS__)
#define EQ_STRUCT_CHAIN_6(name, a,...) name.a=a; EQ_STRUCT_CHAIN_5(name, __VA_ARGS__)
#define EQ_STRUCT_CHAIN_7(name, a,...) name.a=a; EQ_STRUCT_CHAIN_6(name, __VA_ARGS__)
#define EQ_STRUCT_CHAIN_8(name, a,...) name.a=a; EQ_STRUCT_CHAIN_7(name, __VA_ARGS__)
#define EQ_STRUCT_CHAIN_9(name, a,...) name.a=a; EQ_STRUCT_CHAIN_8(name, __VA_ARGS__)

/**
 * Kernel setters
 */
#define SET_ATTRIBUTE_CHAIN(...) GLUE(SET_ATTRIBUTE_CHAIN_,COUNT(__VA_ARGS__))(__VA_ARGS__)

#define SET_ATTRIBUTE_CHAIN_1(a)     set_attribute_from_kernel(_##a,&a);
#define SET_ATTRIBUTE_CHAIN_2(a,...) set_attribute_from_kernel(_##a,&a); SET_ATTRIBUTE_CHAIN_1(__VA_ARGS__)
#define SET_ATTRIBUTE_CHAIN_3(a,...) set_attribute_from_kernel(_##a,&a); SET_ATTRIBUTE_CHAIN_2(__VA_ARGS__)
#define SET_ATTRIBUTE_CHAIN_4(a,...) set_attribute_from_kernel(_##a,&a); SET_ATTRIBUTE_CHAIN_3(__VA_ARGS__)
#define SET_ATTRIBUTE_CHAIN_5(a,...) set_attribute_from_kernel(_##a,&a); SET_ATTRIBUTE_CHAIN_4(__VA_ARGS__)
#define SET_ATTRIBUTE_CHAIN_6(a,...) set_attribute_from_kernel(_##a,&a); SET_ATTRIBUTE_CHAIN_5(__VA_ARGS__)
#define SET_ATTRIBUTE_CHAIN_7(a,...) set_attribute_from_kernel(_##a,&a); SET_ATTRIBUTE_CHAIN_6(__VA_ARGS__)
#define SET_ATTRIBUTE_CHAIN_8(a,...) set_attribute_from_kernel(_##a,&a); SET_ATTRIBUTE_CHAIN_7(__VA_ARGS__)
#define SET_ATTRIBUTE_CHAIN_9(a,...) set_attribute_from_kernel(_##a,&a); SET_ATTRIBUTE_CHAIN_8(__VA_ARGS__)

#define SET_UNIFORM_CHAIN(...) GLUE(SET_UNIFORM_CHAIN_,COUNT(__VA_ARGS__))(__VA_ARGS__)

#define SET_UNIFORM_CHAIN_1(a)     a = *_##a;
#define SET_UNIFORM_CHAIN_2(a,...) a = *_##a; SET_UNIFORM_CHAIN_1(__VA_ARGS__)
#define SET_UNIFORM_CHAIN_3(a,...) a = *_##a; SET_UNIFORM_CHAIN_2(__VA_ARGS__)
#define SET_UNIFORM_CHAIN_4(a,...) a = *_##a; SET_UNIFORM_CHAIN_3(__VA_ARGS__)
#define SET_UNIFORM_CHAIN_5(a,...) a = *_##a; SET_UNIFORM_CHAIN_4(__VA_ARGS__)
#define SET_UNIFORM_CHAIN_6(a,...) a = *_##a; SET_UNIFORM_CHAIN_5(__VA_ARGS__)
#define SET_UNIFORM_CHAIN_7(a,...) a = *_##a; SET_UNIFORM_CHAIN_6(__VA_ARGS__)
#define SET_UNIFORM_CHAIN_8(a,...) a = *_##a; SET_UNIFORM_CHAIN_7(__VA_ARGS__)
#define SET_UNIFORM_CHAIN_9(a,...) a = *_##a; SET_UNIFORM_CHAIN_8(__VA_ARGS__)

/**
 * Vertex buffer getters
 */
#define SET_VARYING_F2_CHAIN(...) GLUE(SET_VARYING_F2_CHAIN_,COUNT(__VA_ARGS__))(__VA_ARGS__)

#define SET_VARYING_F2_CHAIN_1(a)     a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xy;
#define SET_VARYING_F2_CHAIN_2(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xy; SET_VARYING_F2_CHAIN_1(__VA_ARGS__)
#define SET_VARYING_F2_CHAIN_3(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xy; SET_VARYING_F2_CHAIN_2(__VA_ARGS__)
#define SET_VARYING_F2_CHAIN_4(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xy; SET_VARYING_F2_CHAIN_3(__VA_ARGS__)
#define SET_VARYING_F2_CHAIN_5(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xy; SET_VARYING_F2_CHAIN_4(__VA_ARGS__)
#define SET_VARYING_F2_CHAIN_6(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xy; SET_VARYING_F2_CHAIN_5(__VA_ARGS__)
#define SET_VARYING_F2_CHAIN_7(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xy; SET_VARYING_F2_CHAIN_6(__VA_ARGS__)
#define SET_VARYING_F2_CHAIN_8(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xy; SET_VARYING_F2_CHAIN_7(__VA_ARGS__)
#define SET_VARYING_F2_CHAIN_9(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xy; SET_VARYING_F2_CHAIN_8(__VA_ARGS__)

#define SET_VARYING_F3_CHAIN(...) GLUE(SET_VARYING_F3_CHAIN_,COUNT(__VA_ARGS__))(__VA_ARGS__)

#define SET_VARYING_F3_CHAIN_1(a)     a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xyz;
#define SET_VARYING_F3_CHAIN_2(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xyz; SET_VARYING_F3_CHAIN_1(__VA_ARGS__)
#define SET_VARYING_F3_CHAIN_3(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xyz; SET_VARYING_F3_CHAIN_2(__VA_ARGS__)
#define SET_VARYING_F3_CHAIN_4(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xyz; SET_VARYING_F3_CHAIN_3(__VA_ARGS__)
#define SET_VARYING_F3_CHAIN_5(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xyz; SET_VARYING_F3_CHAIN_4(__VA_ARGS__)
#define SET_VARYING_F3_CHAIN_6(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xyz; SET_VARYING_F3_CHAIN_5(__VA_ARGS__)
#define SET_VARYING_F3_CHAIN_7(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xyz; SET_VARYING_F3_CHAIN_6(__VA_ARGS__)
#define SET_VARYING_F3_CHAIN_8(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xyz; SET_VARYING_F3_CHAIN_7(__VA_ARGS__)
#define SET_VARYING_F3_CHAIN_9(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer).xyz; SET_VARYING_F3_CHAIN_8(__VA_ARGS__)

#define SET_VARYING_CHAIN(...) GLUE(SET_VARYING_CHAIN_,COUNT(__VA_ARGS__))(__VA_ARGS__)

#define SET_VARYING_CHAIN_1(a)     a = interpolate_varying(varying, vert_idx, bary, vertex_buffer);
#define SET_VARYING_CHAIN_2(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer); SET_VARYING_CHAIN_1(__VA_ARGS__)
#define SET_VARYING_CHAIN_3(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer); SET_VARYING_CHAIN_2(__VA_ARGS__)
#define SET_VARYING_CHAIN_4(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer); SET_VARYING_CHAIN_3(__VA_ARGS__)
#define SET_VARYING_CHAIN_5(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer); SET_VARYING_CHAIN_4(__VA_ARGS__)
#define SET_VARYING_CHAIN_6(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer); SET_VARYING_CHAIN_5(__VA_ARGS__)
#define SET_VARYING_CHAIN_7(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer); SET_VARYING_CHAIN_6(__VA_ARGS__)
#define SET_VARYING_CHAIN_8(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer); SET_VARYING_CHAIN_7(__VA_ARGS__)
#define SET_VARYING_CHAIN_9(a,...) a = interpolate_varying(varying, vert_idx, bary, vertex_buffer); SET_VARYING_CHAIN_8(__VA_ARGS__)
/**
 * Struct setters 
 */

// To name.a = (float4){a,value}; 
#define SET_STRUCT_CHAIN(name, ...) GLUE(SET_STRUCT_CHAIN_,COUNT(__VA_ARGS__))(name, __VA_ARGS__)

#define SET_STRUCT_CHAIN_1(name, a)     name.a=a;
#define SET_STRUCT_CHAIN_2(name, a,...) name.a=a; SET_STRUCT_CHAIN_1(name, __VA_ARGS__)
#define SET_STRUCT_CHAIN_3(name, a,...) name.a=a; SET_STRUCT_CHAIN_2(name, __VA_ARGS__)
#define SET_STRUCT_CHAIN_4(name, a,...) name.a=a; SET_STRUCT_CHAIN_3(name, __VA_ARGS__)
#define SET_STRUCT_CHAIN_5(name, a,...) name.a=a; SET_STRUCT_CHAIN_4(name, __VA_ARGS__)
#define SET_STRUCT_CHAIN_6(name, a,...) name.a=a; SET_STRUCT_CHAIN_5(name, __VA_ARGS__)
#define SET_STRUCT_CHAIN_7(name, a,...) name.a=a; SET_STRUCT_CHAIN_6(name, __VA_ARGS__)
#define SET_STRUCT_CHAIN_8(name, a,...) name.a=a; SET_STRUCT_CHAIN_7(name, __VA_ARGS__)
#define SET_STRUCT_CHAIN_9(name, a,...) name.a=a; SET_STRUCT_CHAIN_8(name, __VA_ARGS__)

// To name.a = (float4){a,value0, value1}; 
#define SET_STRUCT_F4_F2_CHAIN(name, value0, value1, ...) GLUE(SET_STRUCT_F4_F2_CHAIN_,COUNT(__VA_ARGS__))(name, value0, value1, __VA_ARGS__)

#define SET_STRUCT_F4_F2_CHAIN_1(name, value0, value1, a)     name.a= (float4){a,value0,value1};
#define SET_STRUCT_F4_F2_CHAIN_2(name, value0, value1, a,...) name.a= (float4){a,value0,value1}; SET_STRUCT_F4_F2_CHAIN_1(name, value0, value1, __VA_ARGS__)
#define SET_STRUCT_F4_F2_CHAIN_3(name, value0, value1, a,...) name.a= (float4){a,value0,value1}; SET_STRUCT_F4_F2_CHAIN_2(name, value0, value1, __VA_ARGS__)
#define SET_STRUCT_F4_F2_CHAIN_4(name, value0, value1, a,...) name.a= (float4){a,value0,value1}; SET_STRUCT_F4_F2_CHAIN_3(name, value0, value1, __VA_ARGS__)
#define SET_STRUCT_F4_F2_CHAIN_5(name, value0, value1, a,...) name.a= (float4){a,value0,value1}; SET_STRUCT_F4_F2_CHAIN_4(name, value0, value1, __VA_ARGS__)
#define SET_STRUCT_F4_F2_CHAIN_6(name, value0, value1, a,...) name.a= (float4){a,value0,value1}; SET_STRUCT_F4_F2_CHAIN_5(name, value0, value1, __VA_ARGS__)
#define SET_STRUCT_F4_F2_CHAIN_7(name, value0, value1, a,...) name.a= (float4){a,value0,value1}; SET_STRUCT_F4_F2_CHAIN_6(name, value0, value1, __VA_ARGS__)
#define SET_STRUCT_F4_F2_CHAIN_8(name, value0, value1, a,...) name.a= (float4){a,value0,value1}; SET_STRUCT_F4_F2_CHAIN_7(name, value0, value1, __VA_ARGS__)
#define SET_STRUCT_F4_F2_CHAIN_9(name, value0, value1, a,...) name.a= (float4){a,value0,value1}; SET_STRUCT_F4_F2_CHAIN_8(name, value0, value1, __VA_ARGS__)

// To name.a = (float4){a,value}; 
#define SET_STRUCT_F4_F3_CHAIN(name, value, ...) GLUE(SET_STRUCT_F4_F3_CHAIN_,COUNT(__VA_ARGS__))(name, value, __VA_ARGS__)

#define SET_STRUCT_F4_F3_CHAIN_1(name, value, a)     name.a= (float4){a,value};
#define SET_STRUCT_F4_F3_CHAIN_2(name, value, a,...) name.a= (float4){a,value}; SET_STRUCT_F4_F3_CHAIN_1(name, value, __VA_ARGS__)
#define SET_STRUCT_F4_F3_CHAIN_3(name, value, a,...) name.a= (float4){a,value}; SET_STRUCT_F4_F3_CHAIN_2(name, value, __VA_ARGS__)
#define SET_STRUCT_F4_F3_CHAIN_4(name, value, a,...) name.a= (float4){a,value}; SET_STRUCT_F4_F3_CHAIN_3(name, value, __VA_ARGS__)
#define SET_STRUCT_F4_F3_CHAIN_5(name, value, a,...) name.a= (float4){a,value}; SET_STRUCT_F4_F3_CHAIN_4(name, value, __VA_ARGS__)
#define SET_STRUCT_F4_F3_CHAIN_6(name, value, a,...) name.a= (float4){a,value}; SET_STRUCT_F4_F3_CHAIN_5(name, value, __VA_ARGS__)
#define SET_STRUCT_F4_F3_CHAIN_7(name, value, a,...) name.a= (float4){a,value}; SET_STRUCT_F4_F3_CHAIN_6(name, value, __VA_ARGS__)
#define SET_STRUCT_F4_F3_CHAIN_8(name, value, a,...) name.a= (float4){a,value}; SET_STRUCT_F4_F3_CHAIN_7(name, value, __VA_ARGS__)
#define SET_STRUCT_F4_F3_CHAIN_9(name, value, a,...) name.a= (float4){a,value}; SET_STRUCT_F4_F3_CHAIN_8(name, value, __VA_ARGS__)




#endif

