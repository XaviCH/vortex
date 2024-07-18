#version 330 core

in vec3 position;
in vec2 in_texCoord;

out vec2 out_texCoord;

void main()
{
    gl_Position = vec4(position, 1);
    out_texCoord = in_texCoord;
}
