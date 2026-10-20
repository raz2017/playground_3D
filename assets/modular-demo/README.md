# Upstream lesson 03 demo

The four upstream scene files in this directory originate from
[nickenchev/modern-vulkan, lesson_03](https://github.com/nickenchev/modern-vulkan/tree/lesson_03/assets/modular-demo)
at commit `061d59d15c43e5f5a78ac2159f7fb7d91c9ddaca`:

- `modular-demo.gltf`
- `modular-demo.bin`
- `output_albedo.png`
- `OS2-Warp45-RTM-Desktop.png`

Keep the files together: the glTF references the buffer and textures by relative
path. The separate `../demo.gltf` is the original, reproducible Rust project demo.

The glTF now also includes a locally generated metal trash can beside the terminal
at `(-5.25, 0.0156083163, 4.4)`. Its geometry uses an embedded buffer; the upstream
binary buffer and textures are unchanged. `python tools/add_trash_can.py` adds the
can to a fresh upstream copy and leaves an existing `Trash can` node unchanged.
