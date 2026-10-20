//! # Program workflow
//!
//! This executable is an interactive Vulkan viewer for glTF scenes. This file
//! coordinates the application: `scene` loads the model, `camera` controls the
//! viewpoint, `renderer` draws it, and `diagnostics` tracks frame timing.
//!
//! ## 1. Start and choose the configuration
//!
//! `main()` calls `run()` and reports any error with a nonzero exit code. `run()`
//! first reads the command line. `--help` prints the controls and exits early;
//! otherwise, the options select the model, camera mode, Vulkan validation, and
//! an optional frame limit. Without a model argument, the bundled demo is used.
//!
//! ## 2. Load the scene and create the application resources
//!
//! The scene loader reads the selected `.gltf` or `.glb` file. The application
//! derives an initial camera from that scene and optionally enables orbit mode.
//! It then initializes SDL, creates a resizable Vulkan window, and captures the
//! mouse for relative movement. Finally, it transfers the scene to the renderer
//! and prepares the event pump, diagnostics, and timing state. SDL and the window
//! stay alive while the renderer uses the window's Vulkan surface.
//!
//! ## 3. Repeat the frame loop
//!
//! Each loop iteration performs these steps in order:
//!
//! - Process pending SDL events. Closing the window or pressing Escape exits.
//!   Other events update mouse look, camera movement mode, mouse capture,
//!   fullscreen state, diagnostics visibility, and window state. Size changes
//!   tell the renderer that its swapchain needs updating.
//! - If smoke testing is enabled, perform the scheduled window and input actions
//!   and check the timeout.
//! - Calculate elapsed time since the previous iteration, capped at 0.1 seconds
//!   to avoid large camera jumps after a pause. Update camera movement from held
//!   keys when the window has focus and is not minimized. Smoke testing also
//!   applies scripted camera motion.
//! - Read the drawable size in physical pixels. If minimized or zero-sized,
//!   reset timing diagnostics, sleep briefly, and start the next iteration.
//! - Ask the renderer to draw the scene using the current camera and diagnostics
//!   label. A render call may skip submission while handling a resize, so record
//!   frame timing only when the submitted-frame counter actually increases.
//! - Exit if the requested number of submitted frames has been reached.
//!
//! ## 4. Exercise lifecycle behavior with `--smoke-test`
//!
//! Smoke mode defaults to a 120-frame limit and enables validation unless
//! `--no-validation` is supplied. Across loop iterations it toggles diagnostics,
//! resizes, minimizes and restores the window, switches fullscreen on and off,
//! changes mouse capture, and exercises camera movement. Loop ticks keep these
//! actions progressing even while rendering is paused by minimization. The run
//! fails if it exceeds 30 seconds or exits without the required lifecycle
//! actions, submitted frames, and swapchain creations.
//!
//! ## 5. Finish and check cleanup
//!
//! After the loop, verify smoke-test completion when requested and print the
//! rendering totals. Explicitly drop the renderer before checking Vulkan
//! validation errors so messages produced during renderer teardown are counted.
//! Any validation errors become a failed result; otherwise, report a clean
//! shutdown and return success. SDL and the window are then cleaned up as their
//! owners leave scope. Errors propagated with `?` also unwind local ownership
//! back to `main()`, which prints the failure and exits.

mod camera;
mod diagnostics;
mod renderer;
mod scene;

use renderer::{Renderer, Result, validation_errors};
use sdl3::{
    event::{Event, WindowEvent},
    keyboard::{Keycode, Scancode},
    video::FullscreenType,
};
use std::time::{Duration, Instant};

/// The executable's entry point. Keep error reporting here so `run` can use `?`
/// to return failures, while Rust still cleans up its local resources normally.
/// `if let Err(error)` runs this branch only when `run` returns an error.
fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

/// Own the window and renderer, process input, and keep drawing until we exit.
/// `Result<()>` means success carries no value (`()`); failure carries an error.
/// The `?` operator unwraps a successful result or immediately returns its error.
/// Keeping SDL/window creation before the renderer ensures normal reverse-order
/// cleanup destroys Vulkan's surface before the window it refers to.
fn run() -> Result<()> {
    // Collect command-line arguments into an owned, growable list of strings.
    // skip(1) removes the executable name; |arg| is a closure (a small function).
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help") {
        println!(
            "Vulkan glTF viewer (lesson 03)\n\nEscape / close window: quit\nF12: toggle FPS diagnostics\nMouse: look around\nWASD: move / strafe\nGrave (`): toggle walk / fly\nF11: toggle fullscreen\nTab: release / capture mouse\n--orbit: orbit the origin (W / S zoom, Up / Down pitch)\n--model PATH: load a .gltf or .glb scene (default: bundled demo)\n--frames N: exit after N submitted frames\n--smoke-test: render 120 frames, resize, minimize, restore, toggle fullscreen, then exit\n--no-validation: disable Vulkan validation"
        );
        return Ok(());
    }
    let smoke = args.iter().any(|arg| arg == "--smoke-test");
    // An Option represents a value that may be absent: Some(count) or None.
    // `mut` lets us change these variables while parsing the remaining options.
    let mut limit = smoke.then_some(120_u64);
    let mut validation = cfg!(debug_assertions) || smoke;
    let mut model = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/modular-demo/modular-demo.gltf");
    let mut orbit = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = args
                    .get(i)
                    .ok_or("--model requires a .gltf or .glb path")?
                    .into();
            }
            "--frames" => {
                i += 1;
                let count = args
                    .get(i)
                    .ok_or("--frames requires a positive integer")?
                    .parse::<u64>()?;
                if count == 0 {
                    return Err("--frames must be greater than zero".into());
                }
                limit = Some(count);
            }
            "--no-validation" => validation = false,
            "--orbit" => orbit = true,
            "--smoke-test" => (),
            arg => return Err(format!("Unknown argument: {arg}; use --help").into()),
        }
        i += 1;
    }
    // SDL handles the operating system's window and input events. The builder
    // configures a Vulkan-capable window; build() creates the actual window.
    let scene = scene::Scene::load(&model)?;
    // Initialize the camera before moving scene ownership into the renderer.
    let mut camera = scene.initial_camera()?;
    if orbit {
        camera.orbit();
    }
    let sdl = sdl3::init()?;
    let video = sdl.video()?;
    let mut window = video
        .window("Modern Vulkan — Rust", 1920, 1080)
        .vulkan()
        .resizable()
        .high_pixel_density()
        .position_centered()
        .build()?;
    let mouse = sdl.mouse();
    let mut capture_mouse = true;
    mouse.set_relative_mouse_mode(&window, capture_mouse);
    if !mouse.relative_mouse_mode(&window) {
        return Err(format!("Cannot enable relative mouse mode: {}", sdl3::get_error()).into());
    }
    // The window and SDL must outlive the renderer's Vulkan surface.
    let mut renderer = Renderer::new(&window, validation, scene)?;
    let mut events = sdl.event_pump()?;
    let started = Instant::now();
    let mut ticks = 0;
    let mut minimized = false;
    let mut diagnostics = diagnostics::Diagnostics::new();
    let mut previous_time = Instant::now();
    // The label lets `break 'running` exit this outer loop from inside the event
    // loop. `match` chooses a branch by the event's enum variant; `_` ignores others.
    'running: loop {
        for event in events.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::KeyDown {
                    keycode: Some(Keycode::F12),
                    repeat: false,
                    ..
                } => diagnostics.toggle(),
                Event::MouseMotion { xrel, yrel, .. }
                    if capture_mouse && window.has_input_focus() && !minimized =>
                {
                    camera.mouse_look(xrel, yrel);
                }
                // Key-up toggles once per press, independently of keyboard repeat.
                Event::KeyUp {
                    scancode: Some(Scancode::Grave),
                    ..
                } => {
                    camera.fly = !camera.fly;
                    println!(
                        "Camera movement: {}",
                        if camera.fly { "fly" } else { "walk" }
                    );
                }
                Event::KeyUp {
                    scancode: Some(Scancode::F11),
                    ..
                } => {
                    window.set_fullscreen(window.fullscreen_state() == FullscreenType::Off)?;
                    renderer.request_resize();
                }
                Event::KeyUp {
                    scancode: Some(Scancode::Tab),
                    ..
                } => {
                    capture_mouse = !capture_mouse;
                    mouse.set_relative_mouse_mode(&window, capture_mouse);
                }
                Event::Window { win_event, .. } => match win_event {
                    // Stop reading look/movement while another application has
                    // focus. Restore the user's capture preference on return.
                    WindowEvent::FocusLost => mouse.set_relative_mouse_mode(&window, false),
                    WindowEvent::FocusGained => {
                        mouse.set_relative_mouse_mode(&window, capture_mouse)
                    }
                    WindowEvent::Minimized => minimized = true,
                    WindowEvent::Restored | WindowEvent::Maximized => {
                        minimized = false;
                        renderer.request_resize();
                    }
                    WindowEvent::Resized(..) | WindowEvent::PixelSizeChanged(..) => {
                        renderer.request_resize()
                    }
                    _ => (),
                },
                _ => (),
            }
        }
        // Exercise lifecycle paths as well as the glTF scene in a finite run.
        if smoke {
            match ticks {
                5 | 45 | 75 => diagnostics.toggle(),
                20 => {
                    window.set_size(900, 600)?;
                    renderer.request_resize();
                }
                40 => {
                    window.set_size(1920, 1080)?;
                    renderer.request_resize();
                }
                60 => {
                    window.minimize();
                    minimized = true;
                    println!("Smoke test: minimize");
                }
                70 => {
                    window.restore();
                    minimized = false;
                    renderer.request_resize();
                    println!("Smoke test: restore");
                }
                80 => {
                    window.set_fullscreen(true)?;
                    renderer.request_resize();
                    println!("Smoke test: fullscreen");
                }
                90 => {
                    window.set_fullscreen(false)?;
                    renderer.request_resize();
                    println!("Smoke test: windowed");
                }
                95 => {
                    mouse.set_relative_mouse_mode(&window, false);
                    mouse.set_relative_mouse_mode(&window, true);
                    camera.fly = true;
                }
                105 => camera.fly = false,
                _ => (),
            }
            if started.elapsed() > Duration::from_secs(30) {
                return Err("Smoke test timed out".into());
            }
        }
        ticks += 1;
        let now = Instant::now();
        // Clamp long pauses so restoring the window cannot cause a camera jump.
        let delta = now.duration_since(previous_time).as_secs_f32().min(0.1);
        previous_time = now;
        if window.has_input_focus() && !minimized {
            camera.update(&events.keyboard_state(), delta);
        }
        if smoke {
            camera.mouse_look(-0.5, if ticks < 100 { -0.2 } else { 0.2 });
            camera.move_axes(0.25, 0.5, 0.0, delta);
        }
        // Render using actual pixels, which can differ from logical window units
        // on a high-DPI display. A minimized window may have no drawable area.
        let (width, height) = window.size_in_pixels();
        if minimized || width == 0 || height == 0 {
            diagnostics.reset();
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }
        // render can return successfully without submitting during resize. Count
        // only actual submissions for FPS and the finite-run frame limit.
        let previous_frames = renderer.submitted_frames();
        renderer.render(width, height, diagnostics.label(), &camera)?;
        if renderer.submitted_frames() > previous_frames {
            diagnostics.record_frame(Instant::now());
        }
        if limit.is_some_and(|count| renderer.submitted_frames() >= count) {
            break;
        }
    }
    if smoke
        && (ticks <= 105 || renderer.submitted_frames() < 120 || renderer.swapchain_creations() < 6)
    {
        return Err(
            "Smoke test exited before completing rendering and window lifecycle checks".into(),
        );
    }
    println!(
        "Rendered {} frames; {} swapchain creations",
        renderer.submitted_frames(),
        renderer.swapchain_creations()
    );
    // Count validation messages emitted during teardown too.
    drop(renderer);
    let errors = validation_errors();
    if errors != 0 {
        return Err(format!("Vulkan validation reported {errors} error(s)").into());
    }
    println!("Clean shutdown; validation errors: {errors}");
    Ok(())
}
