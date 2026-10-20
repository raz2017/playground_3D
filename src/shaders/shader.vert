#version 460

// Buffer references let us read GPU memory via addresses passed from Rust.
// Scalar layout gives vec3 4-byte alignment and 12-byte size, matching scene.rs
// instead of requiring 16-byte alignment. Explicit int64 represents the addresses.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Order and size must match SceneGpu::draw's three u64 push constants.
layout(push_constant, scalar) uniform FrameConstants
{
    uint64_t vertexBufferAddress;
    uint64_t materialBufferAddress;
    uint64_t renderItemBufferAddress;
} frameConsts;

// These structs mirror Vertex, Material, and RenderItem in scene.rs.
struct Vertex
{
    vec3 position;
    vec3 color;
    vec3 normal;
    vec2 uv;
};

layout(buffer_reference, scalar) readonly buffer VertexPtr
{
    Vertex vertices[];
};

struct Material
{
    vec4 baseColor;
    uint colorTextureIndex;
};

layout(buffer_reference, scalar) readonly buffer MaterialPtr
{
    Material materials[];
};

struct RenderItem
{
    mat4x4 wvp;
    mat4x4 worldMatrix;
    uint materialIndex;
};

layout(buffer_reference, scalar) readonly buffer RenderItemPtr
{
    RenderItem renderItems[];
};


// Locations connect vertex outputs to fragment inputs. Smooth outputs interpolate
// across triangles; `flat` keeps material values constant across a triangle.
layout (location = 0) out vec3 outColor;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUV;
layout (location = 3) out flat uint outTextureIndex;
layout (location = 4) out flat vec4 outMaterialBaseColor;

void main()
{
    // Vertex pulling replaces fixed vertex-attribute bindings. For indexed draws,
    // gl_VertexIndex already includes the index-buffer value plus vertex_offset.
    VertexPtr vBuffer = VertexPtr(frameConsts.vertexBufferAddress);
    Vertex v = vBuffer.vertices[gl_VertexIndex];

    // Each indirect command draws one instance; its first_instance chooses the
    // record through gl_InstanceIndex, even across batches of indirect draws.
    RenderItemPtr riBuffer = RenderItemPtr(frameConsts.renderItemBufferAddress);
    RenderItem ri = riBuffer.renderItems[gl_InstanceIndex];

    MaterialPtr matBuff = MaterialPtr(frameConsts.materialBufferAddress);
    Material material = matBuff.materials[ri.materialIndex];

    // wvp = projection * view * world. The GPU divides clip XYZ by W afterwards.
    gl_Position = ri.wvp * vec4(v.position, 1.0);
    outColor = v.color;
    // Inverse-transpose preserves perpendicular normals under nonuniform scale.
    // Taking mat3 drops translation; normalize after fragment interpolation.
    outNormal = mat3x3(transpose(inverse(ri.worldMatrix))) * v.normal;
    outUV = v.uv;
    outTextureIndex = material.colorTextureIndex;
    outMaterialBaseColor = material.baseColor;
}
