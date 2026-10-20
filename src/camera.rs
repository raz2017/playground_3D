//! Lesson 03 camera math and controls, kept separate from SDL and Vulkan calls.
//! A camera pose places it in world space; its inverse is the view matrix that
//! moves the world into camera space. Angles and glTF field of view use radians.
use crate::renderer::Result;
use glam::{Mat4, Quat, Vec3};

#[derive(Clone, Copy, Debug)]
pub enum Projection {
    Perspective {
        fov_y: f32,
        near: f32,
        far: Option<f32>,
    },
    Orthographic {
        xmag: f32,
        ymag: f32,
        near: f32,
        far: f32,
    },
}

impl Projection {
    pub fn from_gltf(projection: gltf::camera::Projection<'_>) -> Result<Self> {
        let result = match projection {
            gltf::camera::Projection::Perspective(p) => Self::Perspective {
                fov_y: p.yfov(),
                near: p.znear(),
                far: p.zfar(),
            },
            gltf::camera::Projection::Orthographic(p) => Self::Orthographic {
                xmag: p.xmag(),
                ymag: p.ymag(),
                near: p.znear(),
                far: p.zfar(),
            },
        };
        let valid = match result {
            Self::Perspective { fov_y, near, far } => {
                fov_y.is_finite()
                    && fov_y > 0.0
                    && fov_y < std::f32::consts::PI
                    && near.is_finite()
                    && near > 0.0
                    && far.is_none_or(|far| far.is_finite() && far > near)
            }
            Self::Orthographic {
                xmag,
                ymag,
                near,
                far,
            } => {
                xmag.is_finite()
                    && xmag > 0.0
                    && ymag.is_finite()
                    && ymag > 0.0
                    && near.is_finite()
                    && near >= 0.0
                    && far.is_finite()
                    && far > near
            }
        };
        if !valid {
            return Err("Invalid glTF camera projection".into());
        }
        Ok(result)
    }

    fn matrix(self, aspect: f32) -> Mat4 {
        // Use the drawable aspect for perspective cameras, as in the lesson, so
        // resizing updates the projection. An omitted glTF zfar means infinity.
        // glam's RH variants here use Vulkan's 0..1 clip-space depth range.
        match self {
            Self::Perspective {
                fov_y,
                near,
                far: Some(far),
            } => Mat4::perspective_rh(fov_y, aspect, near, far),
            Self::Perspective {
                fov_y,
                near,
                far: None,
            } => Mat4::perspective_infinite_rh(fov_y, aspect, near),
            Self::Orthographic {
                xmag,
                ymag,
                near,
                far,
            } => Mat4::orthographic_rh(-xmag, xmag, -ymag, ymag, near, far),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraMode {
    FirstPerson,
    Orbit,
}

/// Keep the imported camera connected to its node so attached meshes move too.
/// The scene is static apart from this camera, so its parent's world matrix is fixed.
pub struct CameraAttachment {
    pub node: usize,
    pub parent_world: Mat4,
    pub world_scale: Vec3,
}

pub struct Camera {
    pub mode: CameraMode,
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub fly: bool,
    pub projection: Projection,
    pub attachment: Option<CameraAttachment>,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            mode: CameraMode::FirstPerson,
            position: Vec3::new(0.0, 1.0, 0.0),
            yaw: 0.0,
            pitch: 0.0,
            distance: 3.0,
            fly: false,
            // Convert the fallback 65-degree FOV once; glTF yfov is already radians.
            projection: Projection::Perspective {
                fov_y: 65.0_f32.to_radians(),
                near: 0.01,
                far: Some(1000.0),
            },
            attachment: None,
        }
    }
}

impl Camera {
    /// Initialize from an accumulated node transform, including its parents.
    /// Like the lesson, interactive look uses yaw/pitch and keeps the horizon level.
    pub fn from_world(world: Mat4, projection: Projection) -> Result<Self> {
        if !world.is_finite() || world.determinant().abs() < 1e-12 {
            return Err("Camera transform is non-finite or singular".into());
        }
        // glTF cameras look along local -Z. Transform a direction with w=0 so
        // translation is excluded; normalization removes its scale component.
        let forward = world.transform_vector3(Vec3::NEG_Z).normalize();
        Ok(Self {
            position: world.transform_point3(Vec3::ZERO),
            yaw: (-forward.x).atan2(-forward.z),
            pitch: forward.y.clamp(-1.0, 1.0).asin(),
            projection,
            ..Self::default()
        })
    }

    pub fn orbit(&mut self) {
        self.mode = CameraMode::Orbit;
        self.yaw = std::f32::consts::FRAC_PI_2;
        self.pitch = 0.0;
        self.distance = 3.0;
    }

    pub fn mouse_look(&mut self, xrel: f32, yrel: f32) {
        // SDL gives relative pixels, not a speed. Do not multiply mouse deltas
        // by dt: that would make sensitivity vary with the frame rate.
        let sensitivity = 0.3_f32.to_radians();
        self.yaw -= xrel * sensitivity;
        self.pitch = Self::clamp_pitch(self.pitch - yrel * sensitivity);
    }

    fn clamp_pitch(pitch: f32) -> f32 {
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        pitch.clamp(-limit, limit)
    }

    fn orientation(&self) -> Quat {
        // First pitch around local X, then yaw around world Y; no roll.
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }

    pub fn update(&mut self, keys: &sdl3::keyboard::KeyboardState<'_>, dt: f32) {
        use sdl3::keyboard::Scancode;
        let axis = |positive, negative| {
            (i32::from(keys.is_scancode_pressed(positive))
                - i32::from(keys.is_scancode_pressed(negative))) as f32
        };
        self.move_axes(
            axis(Scancode::D, Scancode::A),
            axis(Scancode::W, Scancode::S),
            axis(Scancode::Up, Scancode::Down),
            dt,
        );
    }

    /// Movement axes are separate from SDL so camera behavior can be unit-tested.
    pub fn move_axes(&mut self, strafe: f32, forward: f32, pitch: f32, dt: f32) {
        let step = 3.0 * dt;
        match self.mode {
            CameraMode::Orbit => {
                self.distance = (self.distance - forward * step).max(0.01);
                self.pitch = Self::clamp_pitch(self.pitch + pitch * step);
            }
            CameraMode::FirstPerson => {
                let rotation = self.orientation();
                let right = rotation * Vec3::X;
                let direction = if self.fly {
                    rotation * Vec3::NEG_Z
                } else {
                    // Walking follows yaw only, keeping height fixed even when
                    // looking up/down. This is free movement without collision.
                    Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos())
                };
                self.position += (right * strafe + direction * forward) * step;
            }
        }
    }

    fn view(&self) -> Mat4 {
        match self.mode {
            CameraMode::FirstPerson => {
                // The inverse of T*R is R^-1*T^-1: first translate world points
                // by -position, then undo the camera rotation. Unit quaternion
                // conjugation gives the inverse rotation without a full inverse.
                Mat4::from_quat(self.orientation().conjugate())
                    * Mat4::from_translation(-self.position)
            }
            CameraMode::Orbit => {
                let position = Vec3::new(
                    self.yaw.cos() * self.pitch.cos(),
                    self.pitch.sin(),
                    self.yaw.sin() * self.pitch.cos(),
                ) * self.distance;
                Mat4::look_at_rh(position, Vec3::ZERO, Vec3::Y)
            }
        }
    }

    pub fn world_pose(&self) -> Mat4 {
        match self.mode {
            CameraMode::FirstPerson => {
                Mat4::from_rotation_translation(self.orientation(), self.position)
            }
            CameraMode::Orbit => self.view().inverse(),
        }
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection.matrix(aspect) * self.view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mouse_look_has_correct_direction_and_clamps_pitch() {
        let mut camera = Camera::default();
        camera.mouse_look(10.0, -10.0);
        let forward = camera.orientation() * Vec3::NEG_Z;
        assert!(forward.x > 0.0 && forward.y > 0.0);
        camera.mouse_look(0.0, -100_000.0);
        assert!(camera.pitch < std::f32::consts::FRAC_PI_2);
        assert!(camera.view().is_finite());
    }

    #[test]
    fn walking_keeps_height_and_flying_follows_pitch() {
        let mut camera = Camera {
            pitch: 0.5,
            ..Camera::default()
        };
        let start = camera.position;
        camera.move_axes(0.0, 1.0, 0.0, 0.5);
        assert!((camera.position.y - start.y).abs() < 1e-6);
        assert!((camera.position.z - start.z + 1.5).abs() < 1e-6);
        camera.fly = true;
        let start = camera.position;
        camera.move_axes(0.0, 1.0, 0.0, 0.5);
        assert!(camera.position.y > start.y);
        assert!(((camera.position - start).length() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn rigid_view_is_inverse_pose_and_movement_uses_elapsed_time() {
        let mut camera = Camera {
            yaw: 0.4,
            pitch: -0.2,
            ..Camera::default()
        };
        let mut split = Camera {
            yaw: 0.4,
            pitch: -0.2,
            ..Camera::default()
        };
        camera.move_axes(1.0, 1.0, 0.0, 1.0);
        for _ in 0..10 {
            split.move_axes(1.0, 1.0, 0.0, 0.1);
        }
        assert!(camera.position.abs_diff_eq(split.position, 1e-5));
        let pose = Mat4::from_rotation_translation(camera.orientation(), camera.position);
        assert!((camera.view() * pose).abs_diff_eq(Mat4::IDENTITY, 1e-5));
    }

    #[test]
    fn perspective_uses_radians_resize_aspect_and_vulkan_depth() {
        let projection = Projection::Perspective {
            fov_y: 65.0_f32.to_radians(),
            near: 0.1,
            far: Some(100.0),
        };
        let wide = projection.matrix(2.0);
        let square = projection.matrix(1.0);
        assert!((square.x_axis.x - 2.0 * wide.x_axis.x).abs() < 1e-6);
        assert!(wide.project_point3(Vec3::new(0.0, 0.0, -0.1)).z.abs() < 1e-5);
        assert!((wide.project_point3(Vec3::new(0.0, 0.0, -100.0)).z - 1.0).abs() < 1e-5);
        assert!(
            Projection::Perspective {
                fov_y: 1.0,
                near: 0.1,
                far: None
            }
            .matrix(1.0)
            .is_finite()
        );
    }

    #[test]
    fn orbit_preserves_distance_limit_and_finite_view() {
        let mut camera = Camera::default();
        camera.orbit();
        camera.move_axes(0.0, 1.0, 1.0, 100.0);
        assert_eq!(camera.distance, 0.01);
        assert!(camera.view_projection(1.0).is_finite());
    }

    #[test]
    fn gltf_infinite_and_orthographic_projections_are_imported() {
        let gltf = gltf::Gltf::from_slice(br#"{
            "asset":{"version":"2.0"},
            "cameras":[
                {"type":"perspective","perspective":{"yfov":1.0,"znear":0.1}},
                {"type":"orthographic","orthographic":{"xmag":2.0,"ymag":1.0,"znear":0.1,"zfar":10.0}}
            ]
        }"#).unwrap();
        let mut cameras = gltf.cameras();
        let infinite = Projection::from_gltf(cameras.next().unwrap().projection()).unwrap();
        assert!(matches!(
            infinite,
            Projection::Perspective { far: None, .. }
        ));
        let ortho = Projection::from_gltf(cameras.next().unwrap().projection())
            .unwrap()
            .matrix(1.0);
        let corner = ortho.project_point3(Vec3::new(2.0, 1.0, -10.0));
        assert!(corner.abs_diff_eq(Vec3::ONE, 1e-5));
    }

    #[test]
    fn invalid_camera_transform_is_rejected() {
        let camera = Camera::default();
        assert!(Camera::from_world(Mat4::from_scale(Vec3::ZERO), camera.projection).is_err());
    }
}
