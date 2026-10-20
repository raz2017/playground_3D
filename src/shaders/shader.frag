#version 460

#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in flat uint inTextureIndex;
layout(location = 4) in flat vec4 inMaterialBaseColor;
layout(location = 0) out vec4 fragColor;

// Matches SceneGpu's descriptor set: one array of combined image/sampler pairs.
layout(set = 0, binding = 0) uniform sampler2D textures[];

void main()
{
    // Interpolated normals may no longer have unit length, so normalize again.
    vec3 nNormal = normalize(inNormal);
    vec3 lightDirection = normalize(vec3(-0.75, -1, 0));
    // Lambert diffuse: surfaces facing the light brighten; facing away yields 0.
    float d = max(dot(nNormal, -lightDirection), 0);
    // Different invocations can choose different textures. nonuniformEXT tells
    // the compiler this descriptor index need not be uniform across invocations.
    vec4 texColor = texture(textures[nonuniformEXT(inTextureIndex)], inUV);

    // Hemisphere ambient maps normal Y from [-1, 1] to [0, 1], blending ground
    // and sky colors to approximate indirect light without a lighting simulation.
	vec3 skyColor = vec3(0.15, 0.18, 0.25);
	vec3 groundColor = vec3(0.05, 0.03, 0.02);
	float t = nNormal.y * 0.5 + 0.5;
	vec3 hemiAmbient = mix(groundColor, skyColor, t);

    // Multiply vertex tint, sampled texture, and material factor before lighting.
    // sRGB textures are decoded during sampling; an sRGB target encodes on write.
    vec3 finalColor = inColor * texColor.rgb * inMaterialBaseColor.rgb;
	vec3 litColor = finalColor * d + finalColor * hemiAmbient;
	fragColor = vec4(litColor, texColor.a);
}
