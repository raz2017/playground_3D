# Modern Vulkan lesson 03 in Rust

A Rust port of [constref's modern-vulkan lesson_03](https://github.com/nickenchev/modern-vulkan/tree/lesson_03), from [Learn Modern Vulkan in 2 Hours](https://www.youtube.com/watch?v=DC9FBRQKNck). It loads glTF scenes and cameras, uploads indexed geometry and textures, traverses a node hierarchy, and renders using buffer device addresses and indexed multi-draw indirect commands. Lesson 03 adds first-person mouse look, walking/flying, and fullscreen switching. The shaders preserve the lesson's directional and hemisphere ambient lighting.

The lesson 01 checkpoint is tagged `vulkan-triangle-baseline`.
The lesson 02 renderer with learning comments is commit `53a63cd`.

## Run

```powershell
cargo run
```

A resizable 1920 x 1080 window displays the upstream modular demo, using its imported camera. Its glTF, binary buffer, and textures are bundled in `assets/modular-demo` with source attribution. Escape or the window's close button exits. Press **F12** to toggle FPS diagnostics in the top-right corner (hidden initially). Debug builds enable the Khronos validation layer and synchronization validation.

Use the **mouse** to look around, **W / S** to move forward/backward, and **A / D** to strafe. Movement is 3 scene units per second; mouse sensitivity is 0.3 degrees per relative pixel. Press **Grave (`)** to toggle walking/flying: walking maintains height, while flying follows the look direction. There is no collision or gravity. **F11** toggles fullscreen; **Tab** releases/captures the mouse. Input pauses when the window loses focus, and mouse capture resumes according to your preference when focus returns.

The first camera in the active scene supplies position, orientation, field of view, and clipping planes. Parent transforms are accumulated, and camera-attached meshes follow movement. Without a camera, the viewer starts at `(0, 1, 0)` looking along `-Z`, with a 65-degree FOV and near/far planes 0.01/1000. Use `--orbit` to orbit the origin instead (distance 3, W/S zoom, Up/Down pitch, mouse look).

The original two-cube demo remains available and now includes a camera under its transformed parent:

```powershell
cargo run -- --model assets/demo.gltf
cargo run -- --model assets/demo.gltf --orbit
```

Load an external `.gltf` (with its referenced buffers/images) or `.glb`:

```powershell
cargo run -- --model "D:/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf"
```

```powershell
cargo run -- --smoke-test
cargo run -- --frames 300
cargo run -- --no-validation
cargo run --release
```

The smoke test renders 120 submitted frames, exercises mouse-look math and camera movement, toggles walk/fly and mouse capture, toggles diagnostics, changes the window size twice, minimizes/restores it, enters/exits fullscreen, and checks for validation errors through shutdown. It requires at least six swapchain creations and completion of the lifecycle sequence. A validation error or incomplete smoke test makes the program exit unsuccessfully. Combine it with `--model` or `--orbit` to test another scene or mode. `--frames N` provides a finite render run. Release builds disable validation by default, except when running the smoke test. `--no-validation` overrides either default.

## Prerequisites on Windows

- Rust with the `x86_64-pc-windows-msvc` toolchain.
- Visual Studio C++ build tools and Windows SDK.
- CMake on PATH; Cargo builds and statically links SDL3 automatically.
- A GPU driver supporting Vulkan 1.3, dynamic rendering, synchronization2, timeline semaphores, buffer device addresses, scalar block layout, runtime texture descriptor arrays with nonuniform indexing and update-after-bind/partially-bound bindings, shaderInt64, multiDrawIndirect, drawIndirectFirstInstance, and a graphics queue capable of presentation. Support is checked before device creation.
- The LunarG Vulkan SDK, with `VULKAN_SDK` pointing to its installation directory. The shaderc crate uses the SDK's static shader compiler library. Validation layers come from the SDK as well.

Run from an x64 Visual Studio Developer PowerShell if native tools cannot be found. If the shaderc library is unavailable, its crate falls back to building from source, which also requires Git, Python, and Ninja on PATH.

## Code map

| File | Responsibility |
| --- | --- |
| `src/main.rs` | CLI, SDL keyboard/mouse events, capture/fullscreen, resize/minimize handling, smoke test |
| `src/camera.rs` | First-person/orbit camera math, walking/flying, projection import and validation |
| `src/scene.rs` | glTF import, meshes/materials/cameras, node transforms, camera attachment, draw list |
| `src/renderer.rs` | Vulkan instance/device, swapchain, pipeline, frame synchronization, cleanup |
| `src/renderer/scene_gpu.rs` | VMA scene uploads, texture descriptors, per-frame render items and indirect commands |
| `src/diagnostics.rs` | FPS sampling and bitmap overlay |
| `src/shaders/shader.vert` | Vertex pulling, world/view/projection and normal transforms, material lookup |
| `src/shaders/shader.frag` | Base-color texture sampling and directional/hemisphere lighting |
| `assets/modular-demo/` | Upstream lesson 03 scene, buffer, textures, and source attribution |
| `assets/demo.gltf` | Self-contained original demo with shared geometry, two materials, and a camera |
| `tools/generate_demo.py` | Reproduce the demo using Python's standard library |

Dependencies are `ash` (Vulkan bindings/loading), `sdl3` (windows/events/surfaces), `vk-mem` (VMA), `shaderc` (GLSL to SPIR-V), `gltf` (scene/buffer/image import), `glam` (matrix math), and `bytemuck` (initialized shader data as bytes). Shaders are embedded and compiled at startup. The default demo is resolved relative to the source checkout at compile time; use `--model` when distributing the executable with other assets.

## Port decisions

- Compared upstream `lesson_02` to `lesson_03` at commit `061d59d15c43e5f5a78ac2159f7fb7d91c9ddaca`. Ports the camera controls/import, 1920 x 1080 default window, modular demo, clear color, and changed directional light; existing lesson 02 GPU resource management remains in place.
- Keeps camera math in its own Rust module, with comments explaining quaternion rotation and inverse view transforms. Initializes from the first camera in the active hierarchy, including parent transforms; static camera parents remain fixed during interactive movement. Like the lesson, yaw/pitch controls keep the horizon level and discard imported camera roll.
- Converts the default 65-degree FOV to radians; imported glTF `yfov` is already radians. Perspective projection follows the drawable aspect ratio on resize. Missing glTF far planes use an infinite perspective; orthographic cameras retain their authored magnification and clip planes. Imported perspective `aspectRatio` is overridden by the drawable aspect, matching the lesson's resize behavior.
- Adds a CLI orbit mode and Tab mouse-release control, while preserving the upstream Grave and F11 controls. Focus changes prevent background look/movement.
- Requests Vulkan **1.3**, which contains every feature the lesson actually uses. The C++ source requests 1.4, but does not use a 1.4-specific feature. This also matches the published Ash 0.38 bindings.
- Uses two frame slots and a timeline semaphore to wait before reusing each slot's command pool, acquisition semaphore, indirect buffer, and render-item buffer. Counters advance only after successful submission, so an out-of-date acquire cannot leave a skipped timeline signal. Host writes are flushed even on non-coherent memory.
- Uses one presentation semaphore per swapchain image and one VMA depth image per frame slot.
- Checks required GPU features, texture descriptor limits, surface capabilities, supported image counts, transforms, alpha modes, and color formats; rebuilds the pipeline if the surface format changes. Multi-draw batches respect `maxDrawIndirectCount`.
- Handles both acquire and present out-of-date/suboptimal results and pauses rendering while minimized.
- Uses `Drop` to destroy resources in dependency order, including on initialization failures. The renderer is destroyed before its SDL window.
- Builds SDL's native library with an optimized development dependency profile to match Rust's release C runtime on MSVC; the application's Rust code keeps debug information.
- Uses scalar-compatible Rust structs (vertex 44 bytes, material 20, render item 132) and pushes three 64-bit buffer addresses. Indexed indirect commands select their render item through `firstInstance` / `gl_InstanceIndex`.
- Uploads only the imported geometry instead of allocating the lesson's fixed 64/32 MB vertex/index budgets. Textures use sRGB color decoding, and texture indexing explicitly uses `nonuniformEXT`.
- Preserves parent-to-child transforms and mesh instances. Supports embedded/external buffers and images, sparse/interleaved accessors through the glTF reader, 8/16/32-bit indices, normalized colors/UVs, and unindexed triangles. Generates normals if omitted and supplies a white texture/default material when absent.
- Uses one mip level per texture, matching the lesson's uploads; glTF sampler wrapping and nearest/linear filters are retained. The active/default scene is selected on import.

This is the lesson's static, opaque base-color renderer: it does not implement full metallic/roughness PBR, normal maps, alpha blending/masking, double-sided materials, animation, skinning, or morph targets. Non-triangle primitives, skinning, morph targets, and base-color textures using UV sets other than TEXCOORD_0 are rejected with an error. Negative-scale transforms can reverse the fixed back-face winding.

## Verification

```powershell
cargo test
cargo clippy --all-targets -- -D warnings
cargo run -- --smoke-test
cargo run -- --model assets/demo.gltf --smoke-test
cargo run -- --model assets/demo.gltf --orbit --smoke-test
cargo run -- --model "path/to/model.glb" --smoke-test
```

Tests check the shader ABI, shared-mesh imports, material/texture selection, accumulated world/WVP transforms, cycle rejection, pixel conversion, imported camera parent transforms and selection, attached mesh movement, mouse-look direction/pitch limits, walk/fly motion, frame-rate independence, view inversion, and projection depth/aspect. The smoke test additionally validates GPU uploads, texture descriptors, indirect draws, and the fullscreen/resize/minimize window lifecycle.
