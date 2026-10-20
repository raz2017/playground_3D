# Modern Vulkan lesson 03 in Rust

A Rust port of [constref's modern-vulkan lesson_03](https://github.com/nickenchev/modern-vulkan/tree/lesson_03), from [Learn Modern Vulkan in 2 Hours](https://www.youtube.com/watch?v=DC9FBRQKNck). It loads glTF scenes and cameras, uploads indexed geometry and textures, traverses a node hierarchy, and renders using buffer device addresses and indexed multi-draw indirect commands. Lesson 03 adds first-person mouse look, walking/flying, and fullscreen switching. The shaders preserve the lesson's directional and hemisphere ambient lighting.

The lesson 01 checkpoint is tagged `vulkan-triangle-baseline`.
The lesson 02 renderer with learning comments is commit `53a63cd`.

## Quick start (Windows)

Run these commands from the repository root in PowerShell. Set up the prerequisites below before the first build.

```powershell
cargo run -- --help
cargo run
```

The viewer opens a resizable 1920 x 1080 window with the bundled modular demo and its imported camera. The mouse is captured immediately: press **Tab** to release it or **Escape** to quit. **F12** shows FPS diagnostics, which are hidden initially.

Cargo builds SDL3 from source on the first run, so the initial build can take longer. Shaders are embedded in the executable and compiled to SPIR-V at startup.

### Prerequisites

- Rust with the `x86_64-pc-windows-msvc` toolchain. The crate uses Rust edition 2024.
- Visual Studio C++ build tools and Windows SDK.
- CMake on `PATH`; Cargo builds and statically links SDL3 automatically.
- The LunarG Vulkan SDK, with `VULKAN_SDK` pointing to its installation directory. The `shaderc` crate uses the SDK's static shader compiler library. Validation layers come from the SDK as well.
- A GPU and driver supporting Vulkan 1.3 and the features listed below.

Run from an x64 Visual Studio Developer PowerShell if native tools cannot be found. Check the tools and SDK environment with:

```powershell
rustc --version
cargo --version
rustup show active-toolchain
cmake --version
$env:VULKAN_SDK
```

If the shaderc library is unavailable, its crate falls back to building from source, which also requires Git, Python, and Ninja on `PATH`.

The renderer requires dynamic rendering, synchronization2, timeline semaphores, buffer device addresses, scalar block layout, runtime texture descriptor arrays with nonuniform indexing and update-after-bind/partially-bound bindings, `shaderInt64`, `multiDrawIndirect`, `drawIndirectFirstInstance`, and a graphics queue capable of presentation. Support is checked before device creation; Vulkan 1.3 support alone is not sufficient.

## Controls

| Input | Action |
| --- | --- |
| Mouse | Look around while captured |
| W / S | Move forward / backward; zoom in / out in orbit mode |
| A / D | Strafe in first-person mode |
| Up / Down | Adjust pitch in orbit mode |
| Grave (backtick) | Toggle walking / flying in first-person mode |
| Tab | Release / capture the mouse |
| F11 | Toggle fullscreen |
| F12 | Toggle FPS diagnostics |
| Escape / close window | Quit |

Walking maintains height; flying follows the look direction. There is no collision or gravity. Movement is 3 scene units per second; mouse sensitivity is 0.3 degrees per relative pixel. Input pauses when the window loses focus, and mouse capture resumes according to your preference when focus returns.

## Load a scene

Load an external `.gltf` or `.glb` with `--model`. Keep a `.gltf` file with all of its referenced buffers and images. Relative model paths are resolved from the current working directory.

```powershell
cargo run -- --model "D:/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf"
```

The original two-cube demo is useful for a small, self-contained scene:

```powershell
cargo run -- --model assets/demo.gltf
cargo run -- --model assets/demo.gltf --orbit
```

The first camera in the active scene supplies position, orientation, field of view, and clipping planes. Parent transforms are accumulated, and camera-attached meshes follow movement. Without a camera, the viewer starts at `(0, 1, 0)` looking along `-Z`, with a 65-degree FOV and near/far planes 0.01/1000. `--orbit` orbits the origin at an initial distance of 3; it does not automatically frame the model.

The default model is `assets/modular-demo/modular-demo.gltf`, resolved relative to the source checkout at compile time. When distributing the executable, supply assets separately and use `--model`. See the [bundled scene notes](assets/modular-demo/README.md) for attribution and the locally added trash can.

### Command-line options

Pass application options after Cargo's `--` separator, for example `cargo run --release -- --model assets/demo.gltf`.

| Option | Behavior |
| --- | --- |
| `--help` | Print controls and options, then exit without opening a window |
| `--model PATH` | Load a `.gltf` or `.glb`; defaults to the bundled modular demo |
| `--orbit` | Orbit the origin instead of using first-person movement |
| `--frames N` | Exit after N submitted frames; N must be a positive integer |
| `--smoke-test` | Exercise camera/window behavior; defaults to 120 submitted frames |
| `--no-validation` | Disable Vulkan validation, including during a smoke test |

Debug builds enable the Khronos validation layer and synchronization validation. Release builds disable validation by default, except when running the smoke test. `--no-validation` overrides either default.

```powershell
cargo run --release
cargo run -- --frames 300
cargo run -- --no-validation
```

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Native build cannot find a compiler, linker, or CMake | Install the C++ build tools and Windows SDK, put CMake on `PATH`, and use an x64 Visual Studio Developer PowerShell. |
| shaderc fails to build or link | Check `VULKAN_SDK` and the SDK's static shader compiler library. A source build also needs Git, Python, and Ninja on `PATH`. |
| `Validation layer missing` | Install the Vulkan SDK's validation layers, or use `cargo run -- --no-validation` to run without validation. |
| `Vulkan 1.3 or newer is required` / `No GPU supports lesson 02` | Check the installed GPU driver and the required features above. The latter message retains the previous lesson's name. |
| Model, buffer, or image cannot be loaded | Check the model path and keep external `.gltf` resources at their referenced relative paths. |
| External model is out of view | The viewer does not automatically frame assets. Check the scene camera and scale; orbit mode targets the origin. |
| Mouse is locked to the window | Press Tab to release it. |

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
| `tools/generate_demo.py` | Reproduce the two-cube demo using Python's standard library |
| `tools/add_trash_can.py` | Add the local trash can to a fresh upstream modular demo |

Dependencies are `ash` (Vulkan bindings/loading), `sdl3` (windows/events/surfaces), `vk-mem` (VMA), `shaderc` (GLSL to SPIR-V), `gltf` (scene/buffer/image import), `glam` (matrix math), and `bytemuck` (initialized shader data as bytes).

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

Run unit tests and lint checks from the repository root:

```powershell
cargo test
cargo clippy --all-targets -- -D warnings
```

Smoke tests open a real window and require the Vulkan setup above; they are not headless tests. Let each run finish without closing its window:

```powershell
cargo run -- --smoke-test
cargo run -- --model assets/demo.gltf --smoke-test
cargo run -- --model assets/demo.gltf --orbit --smoke-test
cargo run -- --model "path/to/model.glb" --smoke-test
```

The smoke test renders at least 120 submitted frames, exercises mouse-look math and camera movement, toggles walk/fly, mouse capture and diagnostics, changes the window size twice, minimizes/restores it, and enters/exits fullscreen. It requires at least six swapchain creations and completion of the lifecycle sequence within 30 seconds. Validation errors, including those reported during shutdown, or an incomplete sequence cause a nonzero exit code. A successful run reports `Clean shutdown; validation errors: 0`; that message only confirms validation ran when validation was enabled. Combine `--smoke-test` with `--model` or `--orbit` as shown above. If you also pass `--frames`, use at least 120; a smaller limit cannot satisfy the smoke-test checks.

Unit tests check the shader ABI, shared-mesh imports, material/texture selection, accumulated world/WVP transforms, cycle rejection, pixel conversion, imported camera parent transforms and selection, attached mesh movement, mouse-look direction/pitch limits, walk/fly motion, frame-rate independence, view inversion, and projection depth/aspect. The smoke test additionally validates GPU uploads, texture descriptors, indirect draws, and the fullscreen/resize/minimize window lifecycle.
