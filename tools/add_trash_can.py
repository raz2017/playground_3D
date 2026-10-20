"""Add a low-poly trash can to the modular scene using only the Python stdlib.

Geometry lives in a separate embedded buffer, preserving the upstream binary.
Running this script again leaves an existing Trash can node unchanged.
"""
import base64
import json
import math
from pathlib import Path
import struct


SCENE = Path(__file__).resolve().parents[1] / "assets/modular-demo/modular-demo.gltf"


def main():
    gltf = json.loads(SCENE.read_text(encoding="utf-8"))
    if any(node.get("name") == "Trash can" for node in gltf["nodes"]):
        print("Trash can already exists.")
        return

    data = bytearray()
    buffer_index = len(gltf["buffers"])

    def accessor(values, fmt, shape, component=5126, target=34962, **extra):
        while len(data) % 4:
            data.append(0)
        offset = len(data)
        for value in values:
            data.extend(struct.pack("<" + fmt, *value))
        view = len(gltf["bufferViews"])
        gltf["bufferViews"].append({"buffer": buffer_index, "byteOffset": offset,
                                  "byteLength": len(data) - offset, "target": target})
        index = len(gltf["accessors"])
        gltf["accessors"].append({"bufferView": view, "componentType": component,
                                  "count": len(values), "type": shape, **extra})
        return index

    def material(name, color):
        index = len(gltf["materials"])
        gltf["materials"].append({"name": name, "pbrMetallicRoughness": {
            "baseColorFactor": [*color, 1], "metallicFactor": 0.65,
            "roughnessFactor": 0.65}})
        return index

    steel = material("Trash can - painted steel", (0.28, 0.34, 0.36))
    rim = material("Trash can - metal rim and handles", (0.55, 0.60, 0.62))
    interior = material("Trash can - dark interior", (0.035, 0.045, 0.05))
    accent = material("Trash can - orange marker", (0.9, 0.32, 0.055))
    parts = {key: ([], [], []) for key in (steel, rim, interior, accent)}

    def quad(mat, points):
        positions, normals, indices = parts[mat]
        a, b, c, _ = points
        u = [b[i] - a[i] for i in range(3)]
        v = [c[i] - a[i] for i in range(3)]
        normal = [u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2],
                  u[0]*v[1] - u[1]*v[0]]
        length = math.sqrt(sum(x*x for x in normal))
        normal = tuple(x / length for x in normal)
        start = len(positions)
        positions.extend(points)
        normals.extend([normal] * 4)
        indices.extend((start + i,) for i in (0, 1, 2, 0, 2, 3))

    def ring(r1, y1, r2, y2, mat):
        for i in range(24):
            a = 2 * math.pi * i / 24
            b = 2 * math.pi * (i + 1) / 24
            quad(mat, [(r1*math.cos(a), y1, r1*math.sin(a)),
                       (r2*math.cos(a), y2, r2*math.sin(a)),
                       (r2*math.cos(b), y2, r2*math.sin(b)),
                       (r1*math.cos(b), y1, r1*math.sin(b))])

    def box(center, size, mat):
        faces = [((0, 0, 1), [(-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)]),
                 ((0, 0, -1), [(1,-1,-1), (-1,-1,-1), (-1,1,-1), (1,1,-1)]),
                 ((1, 0, 0), [(1,-1,1), (1,-1,-1), (1,1,-1), (1,1,1)]),
                 ((-1, 0, 0), [(-1,-1,-1), (-1,-1,1), (-1,1,1), (-1,1,-1)]),
                 ((0, 1, 0), [(-1,1,1), (1,1,1), (1,1,-1), (-1,1,-1)]),
                 ((0, -1, 0), [(-1,-1,-1), (1,-1,-1), (1,-1,1), (-1,-1,1)])]
        for _, corners in faces:
            quad(mat, [tuple(center[j] + p[j]*size[j]/2 for j in range(3))
                       for p in corners])

    # Tapered body, reinforcing bands, open rolled rim, and inward-facing liner.
    ring(0.255, 0.0, 0.255, 0.065, rim)
    ring(0.255, 0.065, 0.32, 0.80, steel)
    ring(0.32, 0.80, 0.34, 0.80, rim)
    ring(0.34, 0.80, 0.34, 0.855, rim)
    ring(0.34, 0.855, 0.29, 0.855, rim)
    ring(0.29, 0.855, 0.29, 0.79, interior)
    ring(0.29, 0.79, 0.225, 0.055, interior)
    # A shallow box seals the bottom and provides an upward-facing interior floor.
    box((0, 0.035, 0), (0.45, 0.035, 0.45), interior)
    for y in (0.18, 0.65):
        r = 0.255 + (y - 0.065) * (0.065 / 0.735) + 0.004
        ring(r, y, r + 0.004, y + 0.025, rim)
    # Two rectangular handles with open centers.
    for x in (-0.345, 0.345):
        for z in (-0.075, 0.075):
            box((x, 0.73, z), (0.095, 0.045, 0.035), rim)
        box((x + math.copysign(0.035, x), 0.73, 0), (0.025, 0.045, 0.18), rim)
    box((0, 0.50, 0.30), (0.14, 0.18, 0.02), accent)

    primitives = []
    for mat, (positions, normals, indices) in parts.items():
        bounds = {"min": [min(p[i] for p in positions) for i in range(3)],
                  "max": [max(p[i] for p in positions) for i in range(3)]}
        primitives.append({"attributes": {
            "POSITION": accessor(positions, "3f", "VEC3", **bounds),
            "NORMAL": accessor(normals, "3f", "VEC3")},
            "indices": accessor(indices, "H", "SCALAR", 5123, 34963),
            "material": mat, "mode": 4})
    gltf["buffers"].append({"byteLength": len(data), "uri":
        "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii")})
    mesh_index = len(gltf["meshes"])
    gltf["meshes"].append({"name": "Trash can", "primitives": primitives})
    node_index = len(gltf["nodes"])
    gltf["nodes"].append({"name": "Trash can", "mesh": mesh_index,
                          "translation": [-5.25, 0.015608316287398338, 4.4]})
    gltf["scenes"][gltf.get("scene", 0)]["nodes"].append(node_index)
    SCENE.write_text(json.dumps(gltf, indent="\t") + "\n", encoding="utf-8")
    print(f"Added trash can: {sum(len(p[0]) for p in parts.values())} vertices.")


if __name__ == "__main__":
    main()
