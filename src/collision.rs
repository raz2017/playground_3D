//! Static triangle collision with a swept, upright player box.
use crate::{renderer::Result, scene::Scene};
use glam::{Mat4, Vec3};

// Clearance for the held handgun in every horizontal viewing direction.
// The box stays world-aligned, so both horizontal axes need the extra reach.
const HALF_SIZE: Vec3 = Vec3::new(1.1, 0.95, 1.1);
const EYE_TO_CENTER: Vec3 = Vec3::new(0.0, -0.75, 0.0);
const SKIN: f32 = 0.001;

struct Triangle {
    points: [Vec3; 3],
    min: Vec3,
    max: Vec3,
}

pub struct CollisionWorld {
    triangles: Vec<Triangle>,
}

pub struct Motion {
    pub position: Vec3,
    pub grounded: bool,
    pub ceiling: bool,
}

impl CollisionWorld {
    pub fn from_scene(scene: &Scene, camera_node: Option<usize>) -> Result<Self> {
        let mut world = Self {
            triangles: Vec::new(),
        };
        let mut stack: Vec<_> = scene
            .roots
            .iter()
            .map(|&id| (id, Mat4::IDENTITY, 0))
            .collect();
        while let Some((id, parent, depth)) = stack.pop() {
            if depth >= scene.nodes.len() {
                return Err("Cycle in collision node hierarchy".into());
            }
            // Camera children (e.g. a held model) are not static obstacles.
            if Some(id) == camera_node {
                continue;
            }
            let node = &scene.nodes[id];
            let transform = parent * node.local;
            if let Some(mesh) = node.mesh {
                for primitive in &scene.meshes[mesh] {
                    let start = primitive.first_index as usize;
                    let end = start + primitive.index_count as usize;
                    for indices in scene.indices[start..end].as_chunks::<3>().0 {
                        let points = std::array::from_fn(|i| {
                            let index = (i64::from(indices[i]) + i64::from(primitive.vertex_offset))
                                as usize;
                            transform
                                .transform_point3(Vec3::from_array(scene.vertices[index].position))
                        });
                        world.add_triangle(points);
                    }
                }
            }
            stack.extend(
                node.children
                    .iter()
                    .map(|&child| (child, transform, depth + 1)),
            );
        }
        Ok(world)
    }

    fn add_triangle(&mut self, points: [Vec3; 3]) {
        if (points[1] - points[0])
            .cross(points[2] - points[0])
            .length_squared()
            < 1e-12
        {
            return;
        }
        self.triangles.push(Triangle {
            min: points[0].min(points[1]).min(points[2]),
            max: points[0].max(points[1]).max(points[2]),
            points,
        });
    }

    pub fn move_player(&self, eye: Vec3, delta: Vec3) -> Motion {
        let mut center = eye + EYE_TO_CENTER;
        let mut remaining = delta;
        let mut grounded = false;
        let mut ceiling = false;
        for _ in 0..5 {
            if remaining.length_squared() < 1e-12 {
                break;
            }
            let mut hit = None;
            let mut earliest = 1.0;
            let min = center.min(center + remaining) - HALF_SIZE;
            let max = center.max(center + remaining) + HALF_SIZE;
            for triangle in &self.triangles {
                if triangle.max.cmplt(min).any() || triangle.min.cmpgt(max).any() {
                    continue;
                }
                if let Some((time, normal)) = sweep_triangle(center, remaining, triangle)
                    && time <= earliest
                {
                    earliest = time;
                    hit = Some(normal);
                }
            }
            let Some(normal) = hit else {
                center += remaining;
                break;
            };
            center += remaining * earliest + normal * SKIN;
            grounded |= normal.y > 0.7;
            ceiling |= normal.y < -0.7;
            remaining *= 1.0 - earliest;
            remaining -= normal * remaining.dot(normal).min(0.0);
        }
        Motion {
            position: center - EYE_TO_CENTER,
            grounded,
            ceiling,
        }
    }
}

/// Continuous separating-axis test: box axes, triangle normal, and edge cross axes.
/// Time intervals prevent passing through thin triangles even on long movements.
fn sweep_triangle(center: Vec3, delta: Vec3, triangle: &Triangle) -> Option<(f32, Vec3)> {
    let [a, b, c] = triangle.points;
    let edges = [b - a, c - b, a - c];
    let mut axes = [Vec3::ZERO; 13];
    axes[..3].copy_from_slice(&[Vec3::X, Vec3::Y, Vec3::Z]);
    axes[3] = edges[0].cross(edges[1]);
    for (i, edge) in edges.iter().enumerate() {
        for (j, axis) in [Vec3::X, Vec3::Y, Vec3::Z].iter().enumerate() {
            axes[4 + i * 3 + j] = edge.cross(*axis);
        }
    }
    let mut enter = f32::NEG_INFINITY;
    let mut exit = 1.0_f32;
    let mut normal = Vec3::ZERO;
    for axis in axes {
        if axis.length_squared() < 1e-12 {
            continue;
        }
        let axis = axis.normalize();
        let p = triangle.points.map(|point| (point - center).dot(axis));
        let radius = HALF_SIZE.dot(axis.abs());
        let lo = p[0].min(p[1]).min(p[2]) - radius;
        let hi = p[0].max(p[1]).max(p[2]) + radius;
        let speed = delta.dot(axis);
        if speed.abs() < 1e-8 {
            if lo > 0.0 || hi < 0.0 {
                return None;
            }
            continue;
        }
        let t1 = lo / speed;
        let t2 = hi / speed;
        let near = t1.min(t2);
        if near > enter {
            enter = near;
            normal = -speed.signum() * axis;
        }
        exit = exit.min(t1.max(t2));
        if enter > exit {
            return None;
        }
    }
    // Initial deep overlaps need an authored clear spawn (fly can escape them).
    if !(-1e-5..=1.0).contains(&enter) || exit < 0.0 {
        return None;
    }
    Some((enter.max(0.0), normal))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floor() -> CollisionWorld {
        let mut world = CollisionWorld {
            triangles: Vec::new(),
        };
        world.add_triangle([
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, -10.0),
            Vec3::new(0.0, 0.0, 10.0),
        ]);
        world
    }

    #[test]
    fn lands_on_triangle_without_tunneling() {
        let hit = floor().move_player(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -20.0, 0.0));
        assert!(hit.grounded);
        assert!((hit.position.y - 1.701).abs() < 1e-4);
    }

    #[test]
    fn wall_blocks_and_allows_sliding() {
        let mut world = floor();
        world.add_triangle([
            Vec3::new(2.0, -10.0, -10.0),
            Vec3::new(2.0, 10.0, -10.0),
            Vec3::new(2.0, 0.0, 10.0),
        ]);
        let hit = world.move_player(Vec3::new(0.0, 2.0, 0.0), Vec3::new(5.0, 0.0, 2.0));
        assert!((hit.position.x - 0.899).abs() < 1e-4);
        assert!((hit.position.z - 2.0).abs() < 1e-4);
    }

    #[test]
    fn upward_movement_hits_ceiling_but_leaves_floor() {
        let mut world = floor();
        world.add_triangle([
            Vec3::new(-10.0, 3.0, -10.0),
            Vec3::new(10.0, 3.0, -10.0),
            Vec3::new(0.0, 3.0, 10.0),
        ]);
        let hit = world.move_player(Vec3::new(0.0, 1.701, 0.0), Vec3::Y * 5.0);
        assert!(hit.ceiling);
        assert!(!hit.grounded);
        assert!((hit.position.y - 2.799).abs() < 1e-4);
    }

    #[test]
    fn player_can_jump_after_resting_and_falls_off_edges() {
        let mut camera = crate::camera::Camera::default();
        camera.position.y = 1.7;
        camera.set_collision(floor());
        for _ in 0..60 {
            camera.move_axes(0.0, 0.0, 0.0, 1.0 / 60.0);
        }
        let resting = camera.position.y;
        camera.jump();
        camera.move_axes(0.0, 0.0, 0.0, 0.1);
        assert!(camera.position.y > resting + 0.3);
        for _ in 0..120 {
            camera.move_axes(0.0, 0.0, 0.0, 1.0 / 60.0);
        }
        assert!((camera.position.y - resting).abs() < 0.002);
        camera.position.x = 20.0;
        camera.move_axes(0.0, 0.0, 0.0, 0.5);
        assert!(camera.position.y < resting - 1.0);
        let falling_height = camera.position.y;
        camera.jump();
        camera.move_axes(0.0, 0.0, 0.0, 0.1);
        assert!(camera.position.y < falling_height);
    }

    #[test]
    fn scene_collision_uses_parent_transforms_and_excludes_camera_subtree() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/demo.gltf");
        let mut scene = Scene::load(&path).unwrap();
        let original = CollisionWorld::from_scene(&scene, None).unwrap();
        scene.nodes[scene.roots[0]].local =
            Mat4::from_translation(Vec3::X * 10.0) * scene.nodes[scene.roots[0]].local;
        let moved = CollisionWorld::from_scene(&scene, None).unwrap();
        assert_eq!(original.triangles.len(), moved.triangles.len());
        for (a, b) in original.triangles.iter().zip(&moved.triangles) {
            for i in 0..3 {
                assert!((a.points[i] + Vec3::X * 10.0).abs_diff_eq(b.points[i], 1e-5));
            }
        }
        let excluded = CollisionWorld::from_scene(&scene, Some(scene.roots[0])).unwrap();
        assert!(excluded.triangles.is_empty());
    }
}
