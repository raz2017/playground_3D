//! FPS sampling and a small bitmap overlay without a font dependency.
use ash::{Device, vk};
use std::time::{Duration, Instant};

pub struct Diagnostics {
    visible: bool,
    started: Instant,
    frames: u64,
    label: String,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self {
            visible: false,
            started: Instant::now(),
            frames: 0,
            label: "FPS: --".into(),
        }
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
        self.reset();
    }

    /// Exclude time spent paused/minimized from the next FPS sample.
    pub fn reset(&mut self) {
        self.started = Instant::now();
        self.frames = 0;
        self.label = "FPS: --".into();
    }

    /// Average submitted frames over half a second to keep the label readable.
    /// This measures application submission rate, not individual GPU execution time.
    pub fn record_frame(&mut self, now: Instant) {
        if !self.visible {
            return;
        }
        self.frames += 1;
        let elapsed = now.duration_since(self.started);
        if elapsed >= Duration::from_millis(500) {
            self.label = format!("FPS: {:.0}", self.frames as f64 / elapsed.as_secs_f64());
            self.started = now;
            self.frames = 0;
        }
    }

    /// Borrow the stored string as &str; None tells the renderer to skip the overlay.
    /// The returned text cannot outlive the Diagnostics value that owns it.
    pub fn label(&self) -> Option<&str> {
        self.visible.then_some(self.label.as_str())
    }
}

// Seven rows of five bits each: bit 4 is the leftmost pixel, bit 0 the rightmost.
fn glyph(character: char) -> [u8; 7] {
    match character {
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        ':' => [0, 4, 4, 0, 4, 4, 0],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        _ => [0; 7],
    }
}

/// Clip rectangles so resizing to a tiny window still produces valid commands.
fn rect(extent: vk::Extent2D, x: u32, y: u32, width: u32, height: u32) -> Option<vk::ClearRect> {
    // Saturating subtraction stops at zero instead of underflowing unsigned ints.
    let width = width.min(extent.width.saturating_sub(x));
    let height = height.min(extent.height.saturating_sub(y));
    (width > 0 && height > 0).then_some(vk::ClearRect {
        rect: vk::Rect2D {
            offset: vk::Offset2D {
                x: x as i32,
                y: y as i32,
            },
            extent: vk::Extent2D { width, height },
        },
        base_array_layer: 0,
        layer_count: 1,
    })
}

/// Draw after the scene so diagnostics stay visible over scene geometry.
/// SAFETY: command must be recording inside rendering with color attachment 0,
/// a single layer, and a render area covering extent.
pub unsafe fn draw(device: &Device, command: vk::CommandBuffer, extent: vk::Extent2D, label: &str) {
    const SCALE: u32 = 3;
    const MARGIN: u32 = 12;
    const PADDING: u32 = 8;
    let width = label.len() as u32 * 6 * SCALE - SCALE + 2 * PADDING;
    let height = 7 * SCALE + 2 * PADDING;
    let x = extent.width.saturating_sub(width + MARGIN);
    let y = MARGIN.min(extent.height.saturating_sub(height));
    // Clear small rectangles directly in the active color attachment. This draws
    // the bitmap without a text shader, font texture, or additional pipeline.
    let attachment = |color| {
        vk::ClearAttachment::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .color_attachment(0)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            })
    };
    if let Some(background) = rect(extent, x, y, width, height) {
        unsafe {
            device.cmd_clear_attachments(
                command,
                &[attachment([0.025, 0.025, 0.025, 1.0])],
                &[background],
            );
        }
    }
    let mut pixels = Vec::new();
    for (index, character) in label.chars().enumerate() {
        for (row, bits) in glyph(character).iter().enumerate() {
            for column in 0..5 {
                if bits & (1 << (4 - column)) == 0 {
                    continue;
                }
                if let Some(pixel) = rect(
                    extent,
                    x + PADDING + index as u32 * 6 * SCALE + column * SCALE,
                    y + PADDING + row as u32 * SCALE,
                    SCALE,
                    SCALE,
                ) {
                    pixels.push(pixel);
                }
            }
        }
    }
    if !pixels.is_empty() {
        unsafe {
            device.cmd_clear_attachments(command, &[attachment([0.8, 1.0, 0.8, 1.0])], &pixels);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_frames_and_resets_on_toggle() {
        let mut diagnostics = Diagnostics::new();
        assert_eq!(diagnostics.label(), None);
        diagnostics.toggle();
        let started = diagnostics.started;
        for frame in 1..=30 {
            diagnostics.record_frame(started + Duration::from_micros(frame * 500_000 / 30));
        }
        assert_eq!(diagnostics.label(), Some("FPS: 60"));
        diagnostics.toggle();
        assert_eq!(diagnostics.label(), None);
        diagnostics.toggle();
        assert_eq!(diagnostics.label(), Some("FPS: --"));
        assert_eq!(diagnostics.frames, 0);
    }

    #[test]
    fn clips_rectangles_to_tiny_windows() {
        let extent = vk::Extent2D {
            width: 2,
            height: 1,
        };
        let clipped = rect(extent, 1, 0, 3, 3).unwrap();
        assert_eq!(clipped.rect.extent.width, 1);
        assert_eq!(clipped.rect.extent.height, 1);
        assert!(rect(extent, 2, 0, 3, 3).is_none());
        assert!(rect(extent, 0, 1, 3, 3).is_none());
    }
}
