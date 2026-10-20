"""Generate the original, self-contained lesson 02 glTF demo (stdlib only)."""
import base64
import json
import math
from pathlib import Path
import struct
import zlib

vertices = []
normals = []
uvs = []
indices = []
# Four counter-clockwise vertices per face, viewed from outside the cube.
faces = [
    ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
    ((0, 0, -1), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)]),
    ((1, 0, 0), [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)]),
    ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),
    ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)]),
    ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
]
# Separate face vertices keep edges sharp: corner positions repeat, but normals
# and UVs differ per face. Each quad becomes two indexed triangles.
for normal, positions in faces:
    start = len(vertices)
    vertices.extend(tuple(c * 0.5 for c in p) for p in positions)
    normals.extend([normal] * 4)
    uvs.extend([(0, 1), (1, 1), (1, 0), (0, 0)])
    indices.extend(start + i for i in [0, 1, 2, 0, 2, 3])

data = bytearray()
views = []
accessors = []
def attribute(values, fmt, shape, component=5126, **extra):
    # A bufferView selects bytes; an accessor describes their numeric type/shape.
    # glTF data is little-endian, and each attribute starts at a 4-byte boundary.
    while len(data) % 4:
        data.append(0)
    offset = len(data)
    for value in values:
        data.extend(struct.pack('<' + fmt, *(value if isinstance(value, tuple) else (value,))))
    views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(data) - offset})
    accessors.append({'bufferView': len(views) - 1, 'componentType': component,
                      'count': len(values), 'type': shape, **extra})
    return len(accessors) - 1

pos = attribute(vertices, '3f', 'VEC3', min=[-0.5] * 3, max=[0.5] * 3)
norm = attribute(normals, '3f', 'VEC3')
uv = attribute(uvs, '2f', 'VEC2')
idx_a = attribute(indices[:18], 'H', 'SCALAR', 5123)
idx_b = attribute(indices[18:], 'H', 'SCALAR', 5123)

def chunk(kind, payload):
    # PNG chunks store a big-endian length, type, payload, and CRC checksum.
    return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
pixels = bytearray()
for y in range(8):
    pixels.append(0)  # PNG filter: none.
    for x in range(8):
        pixels.extend((245, 245, 245, 255) if (x // 2 + y // 2) % 2 else (35, 80, 180, 255))
png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 8, 8, 8, 6, 0, 0, 0))
       + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))
uri = lambda mime, payload: 'data:' + mime + ';base64,' + base64.b64encode(payload).decode()
# A Y-axis quaternion is (0, sin(angle/2), 0, cos(angle/2)). Both child nodes
# share mesh 0 and inherit this parent rotation, exercising scene traversal.
angle = math.radians(25) / 2
scene = {
    'asset': {'version': '2.0', 'generator': 'hello_rust original demo'},
    'scene': 0, 'scenes': [{'nodes': [0]}],
    'nodes': [
        {'name': 'Parent rotation', 'rotation': [0, math.sin(angle), 0, math.cos(angle)], 'children': [1, 2, 3]},
        {'name': 'Checker cube', 'mesh': 0, 'translation': [-0.65, 0, 0]},
        {'name': 'Shared mesh child', 'mesh': 0, 'translation': [0.65, 0.25, 0], 'scale': [0.7, 0.7, 0.7]},
        # Camera looks down toward the cubes; its parent rotation is inherited.
        {'name': 'Demo camera', 'camera': 0, 'translation': [0, 1, 3],
         'rotation': [math.sin(-math.atan(1 / 3) / 2), 0, 0, math.cos(-math.atan(1 / 3) / 2)]},
    ],
    'cameras': [{'type': 'perspective', 'perspective': {
        'yfov': math.radians(65), 'znear': 0.1, 'zfar': 100}}],
    'meshes': [{'name': 'Two material cube', 'primitives': [
        {'attributes': {'POSITION': pos, 'NORMAL': norm, 'TEXCOORD_0': uv}, 'indices': idx_a, 'material': 0},
        {'attributes': {'POSITION': pos, 'NORMAL': norm, 'TEXCOORD_0': uv}, 'indices': idx_b, 'material': 1},
    ]}],
    'materials': [
        {'name': 'Checker', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}},
        {'name': 'Warm solid (white fallback)', 'pbrMetallicRoughness': {'baseColorFactor': [0.9, 0.35, 0.08, 1]}},
    ],
    'textures': [{'source': 0, 'sampler': 0}],
    'samplers': [{'magFilter': 9728, 'minFilter': 9728, 'wrapS': 10497, 'wrapT': 10497}],
    'images': [{'uri': uri('image/png', png)}],
    'buffers': [{'uri': uri('application/octet-stream', data), 'byteLength': len(data)}],
    'bufferViews': views, 'accessors': accessors,
}
output = Path(__file__).resolve().parents[1] / 'assets' / 'demo.gltf'
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(scene, indent=2) + '\n', encoding='utf-8')
