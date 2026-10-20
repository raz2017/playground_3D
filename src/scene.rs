//! CPU scene data and glTF import. Meshes are shared by nodes; transforms are
//! accumulated from parent to child when preparing each frame's draw list.
use crate::camera::{Camera, CameraAttachment, Projection};
use crate::renderer::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::path::Path;

// These three structs are copied directly into GPU buffers. repr(C) fixes field
// order/layout; GLSL's `scalar` layout must agree with their sizes and offsets.
// Pod means plain data with no padding or references; Zeroable permits all-zero
// initialization. Copy makes assignment copy the value rather than move ownership.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Material {
    pub base_color: [f32; 4],
    pub texture: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RenderItem {
    pub wvp: [f32; 16],
    pub world: [f32; 16],
    pub material: u32,
}

/// One indexed triangle batch. Offsets refer to the scene's shared geometry arrays.
pub struct Primitive {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
    pub material: u32,
}

/// A node places a mesh in the scene; several nodes can reference the same mesh.
/// Store child indices rather than Rust references so the Vec can grow freely.
pub struct Node {
    pub local: Mat4,
    pub mesh: Option<usize>,
    pub camera: Option<Projection>,
    pub children: Vec<usize>,
}

/// An image plus sampling rules. Multiple textures can reuse one image.
pub struct Texture {
    pub image: usize,
    pub mag: ash::vk::Filter,
    pub min: ash::vk::Filter,
    pub wrap_u: ash::vk::SamplerAddressMode,
    pub wrap_v: ash::vk::SamplerAddressMode,
}

pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// CPU-owned data, independent of Vulkan allocations. GPU upload borrows these
/// arrays as bytes; meshes and nodes retain indices into the shared collections.
pub struct Scene {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub materials: Vec<Material>,
    pub images: Vec<Image>,
    pub textures: Vec<Texture>,
    pub meshes: Vec<Vec<Primitive>>,
    pub nodes: Vec<Node>,
    pub roots: Vec<usize>,
}

impl Scene {
    /// Import glTF into the smaller, static scene format this renderer supports.
    pub fn load(path: &Path) -> Result<Self> {
        use ash::vk::{Filter, SamplerAddressMode};
        let (document, buffers, images) = gltf::import(path)
            .map_err(|error| format!("Cannot load {}: {error}", path.display()))?;
        // Reserve slot 0 for a white texture and default material. Imported image,
        // texture, and material indices are shifted by one to keep this fallback.
        let mut scene = Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            materials: vec![Material {
                base_color: [1.0; 4],
                texture: 0,
            }],
            images: vec![Image {
                width: 1,
                height: 1,
                rgba: vec![255; 4],
            }],
            textures: vec![Texture {
                image: 0,
                mag: Filter::NEAREST,
                min: Filter::NEAREST,
                wrap_u: SamplerAddressMode::REPEAT,
                wrap_v: SamplerAddressMode::REPEAT,
            }],
            meshes: Vec::new(),
            nodes: Vec::new(),
            roots: Vec::new(),
        };
        for image in images {
            scene.images.push(Image {
                width: image.width,
                height: image.height,
                rgba: rgba_pixels(&image)?,
            });
        }
        let wrap = |mode| match mode {
            gltf::texture::WrappingMode::ClampToEdge => SamplerAddressMode::CLAMP_TO_EDGE,
            gltf::texture::WrappingMode::MirroredRepeat => SamplerAddressMode::MIRRORED_REPEAT,
            gltf::texture::WrappingMode::Repeat => SamplerAddressMode::REPEAT,
        };
        for texture in document.textures() {
            let sampler = texture.sampler();
            scene.textures.push(Texture {
                image: texture.source().index() + 1,
                mag: if sampler.mag_filter() == Some(gltf::texture::MagFilter::Nearest) {
                    Filter::NEAREST
                } else {
                    Filter::LINEAR
                },
                min: match sampler.min_filter() {
                    Some(
                        gltf::texture::MinFilter::Nearest
                        | gltf::texture::MinFilter::NearestMipmapNearest
                        | gltf::texture::MinFilter::NearestMipmapLinear,
                    ) => Filter::NEAREST,
                    _ => Filter::LINEAR,
                },
                wrap_u: wrap(sampler.wrap_s()),
                wrap_v: wrap(sampler.wrap_t()),
            });
        }
        for material in document.materials() {
            let pbr = material.pbr_metallic_roughness();
            if pbr
                .base_color_texture()
                .is_some_and(|info| info.tex_coord() != 0)
            {
                return Err("Only TEXCOORD_0 base color textures are supported".into());
            }
            scene.materials.push(Material {
                base_color: pbr.base_color_factor(),
                texture: pbr
                    .base_color_texture()
                    .map_or(0, |info| info.texture().index() as u32 + 1),
            });
        }
        for mesh in document.meshes() {
            let mut primitives = Vec::new();
            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    return Err("Only glTF triangle primitives are supported".into());
                }
                if primitive.morph_targets().next().is_some() {
                    return Err("Morph targets are outside lesson 02".into());
                }
                // The closure lends the reader the bytes of each requested buffer.
                // The glTF reader handles accessor strides and component formats.
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                let positions: Vec<_> = reader
                    .read_positions()
                    .ok_or("Mesh has no POSITION attribute")?
                    .collect();
                let count = positions.len();
                // map_or_else lazily chooses a fallback for None or converts Some.
                // Unindexed triangles get sequential indices; all indices become u32.
                let local_indices: Vec<u32> = reader.read_indices().map_or_else(
                    || (0..count as u32).collect(),
                    |indices| indices.into_u32().collect(),
                );
                if count == 0
                    || local_indices.is_empty()
                    || !local_indices.len().is_multiple_of(3)
                    || local_indices.iter().any(|&index| index as usize >= count)
                {
                    return Err("Invalid triangle indices or empty primitive".into());
                }
                // If normals are absent, accumulate each triangle's cross product
                // at its vertices, then normalize. Larger faces contribute more.
                let normals: Vec<_> = reader.read_normals().map_or_else(
                    || {
                        let mut normals = vec![Vec3::ZERO; count];
                        for tri in local_indices.as_chunks::<3>().0 {
                            let a = Vec3::from_array(positions[tri[0] as usize]);
                            let b = Vec3::from_array(positions[tri[1] as usize]);
                            let c = Vec3::from_array(positions[tri[2] as usize]);
                            let normal = (b - a).cross(c - a);
                            for &index in tri {
                                normals[index as usize] += normal;
                            }
                        }
                        normals
                            .into_iter()
                            .map(|normal| normal.normalize_or_zero().to_array())
                            .collect()
                    },
                    |normals| normals.collect(),
                );
                let colors: Vec<_> = reader.read_colors(0).map_or_else(
                    || vec![[1.0; 3]; count],
                    |colors| colors.into_rgb_f32().collect(),
                );
                let uvs: Vec<_> = reader
                    .read_tex_coords(0)
                    .map_or_else(|| vec![[0.0; 2]; count], |uvs| uvs.into_f32().collect());
                if normals.len() != count || colors.len() != count || uvs.len() != count {
                    return Err("Vertex attribute counts differ".into());
                }
                // Keep indices local to this primitive. Vulkan adds vertex_offset
                // when fetching a vertex from the combined scene vertex buffer.
                // try_from reports overflow instead of silently truncating offsets.
                primitives.push(Primitive {
                    first_index: u32::try_from(scene.indices.len())?,
                    index_count: u32::try_from(local_indices.len())?,
                    vertex_offset: i32::try_from(scene.vertices.len())?,
                    material: primitive
                        .material()
                        .index()
                        .map_or(0, |index| index as u32 + 1),
                });
                scene.vertices.extend((0..count).map(|i| Vertex {
                    position: positions[i],
                    normal: normals[i],
                    color: colors[i],
                    uv: uvs[i],
                }));
                scene.indices.extend(local_indices);
            }
            scene.meshes.push(primitives);
        }
        for node in document.nodes() {
            if node.skin().is_some() {
                return Err("Skinned meshes are outside lesson 02".into());
            }
            scene.nodes.push(Node {
                local: Mat4::from_cols_array_2d(&node.transform().matrix()),
                mesh: node.mesh().map(|mesh| mesh.index()),
                // Option::map produces Option<Result<...>>; transpose turns it
                // into Result<Option<...>>, letting ? propagate projection errors.
                camera: node
                    .camera()
                    .map(|camera| Projection::from_gltf(camera.projection()))
                    .transpose()?,
                children: node.children().map(|child| child.index()).collect(),
            });
        }
        let selected = document
            .default_scene()
            .or_else(|| document.scenes().next())
            .ok_or("glTF has no scene")?;
        scene.roots = selected.nodes().map(|node| node.index()).collect();
        let draws = scene.draws(Mat4::IDENTITY)?;
        if draws.0.is_empty() {
            return Err("Selected glTF scene has no drawable meshes".into());
        }
        println!(
            "Scene: {} vertices, {} indices, {} draws, {} textures",
            scene.vertices.len(),
            scene.indices.len(),
            draws.0.len(),
            scene.textures.len()
        );
        Ok(scene)
    }

    /// Select the first camera reachable in the active scene, in document order.
    /// Cameras in inactive scenes are ignored. The movable camera owns a world
    /// pose initialized from the node; static mesh geometry remains shared.
    pub fn initial_camera(&self) -> Result<Camera> {
        let mut stack: Vec<_> = self
            .roots
            .iter()
            .rev()
            .map(|&id| (id, Mat4::IDENTITY, 0))
            .collect();
        while let Some((id, parent, depth)) = stack.pop() {
            if depth >= self.nodes.len() {
                return Err("Cycle in glTF node hierarchy".into());
            }
            let node = &self.nodes[id];
            let world = parent * node.local;
            if let Some(projection) = node.camera {
                let mut camera = Camera::from_world(world, projection)?;
                camera.attachment = Some(CameraAttachment {
                    node: id,
                    parent_world: parent,
                    world_scale: world.to_scale_rotation_translation().0,
                });
                return Ok(camera);
            }
            stack.extend(node.children.iter().rev().map(|&id| (id, world, depth + 1)));
        }
        // Match the lesson's camera at (0, 1, 0) when a model has no camera node.
        Ok(Camera::default())
    }

    /// Convert the interactive world pose back to node-local space. Updating the
    /// node lets normal scene traversal move camera children along with the camera.
    pub fn update_camera(&mut self, camera: &Camera) {
        if let Some(attachment) = &camera.attachment {
            self.nodes[attachment.node].local = attachment.parent_world.inverse()
                * camera.world_pose()
                * Mat4::from_scale(attachment.world_scale);
        }
    }

    /// Build matching indirect commands and shader records for this camera.
    /// This borrows the scene; the returned Vecs own the generated frame data.
    pub fn draws(
        &self,
        view_projection: Mat4,
    ) -> Result<(Vec<ash::vk::DrawIndexedIndirectCommand>, Vec<RenderItem>)> {
        let mut commands = Vec::new();
        let mut items = Vec::new();
        // Iterative depth-first traversal: each entry carries its parent's world
        // matrix and depth. A valid path cannot contain more nodes than the scene.
        let mut stack: Vec<_> = self
            .roots
            .iter()
            .map(|&id| (id, Mat4::IDENTITY, 0))
            .collect();
        while let Some((id, parent, depth)) = stack.pop() {
            if depth >= self.nodes.len() {
                return Err("Cycle in glTF node hierarchy".into());
            }
            let node = &self.nodes[id];
            // Column-vector math applies the local transform first, then parent.
            let world = parent * node.local;
            // The vertex shader inverts world for normals; singular matrices fail.
            if !world.is_finite() || world.determinant().abs() < 1e-12 {
                return Err("Node transform is non-finite or singular".into());
            }
            if let Some(mesh) = node.mesh {
                for primitive in &self.meshes[mesh] {
                    // first_instance becomes gl_InstanceIndex in the vertex shader,
                    // linking this draw to items[items.len()] before it is appended.
                    commands.push(ash::vk::DrawIndexedIndirectCommand {
                        index_count: primitive.index_count,
                        instance_count: 1,
                        first_index: primitive.first_index,
                        vertex_offset: primitive.vertex_offset,
                        first_instance: u32::try_from(items.len())?,
                    });
                    items.push(RenderItem {
                        wvp: (view_projection * world).to_cols_array(),
                        world: world.to_cols_array(),
                        material: primitive.material,
                    });
                }
            }
            stack.extend(node.children.iter().map(|&id| (id, world, depth + 1)));
        }
        Ok((commands, items))
    }
}

/// Convert imported pixels to the RGBA8 byte format used by texture uploads.
fn rgba_pixels(image: &gltf::image::Data) -> Result<Vec<u8>> {
    use gltf::image::Format;
    let (channels, bytes) = match image.format {
        Format::R8 => (1, 1),
        Format::R8G8 => (2, 1),
        Format::R8G8B8 => (3, 1),
        Format::R8G8B8A8 => (4, 1),
        Format::R16 => (1, 2),
        Format::R16G16 => (2, 2),
        Format::R16G16B16 => (3, 2),
        Format::R16G16B16A16 => (4, 2),
        Format::R32G32B32FLOAT => (3, 4),
        Format::R32G32B32A32FLOAT => (4, 4),
    };
    let mut rgba = Vec::new();
    for pixel in image.pixels.chunks_exact(channels * bytes) {
        let mut components = [0, 0, 0, 255];
        for c in 0..channels {
            let data = &pixel[c * bytes..(c + 1) * bytes];
            components[c] = match bytes {
                1 => data[0],
                2 => (u16::from_ne_bytes(data.try_into()?) / 257) as u8,
                _ => (f32::from_ne_bytes(data.try_into()?).clamp(0.0, 1.0) * 255.0).round() as u8,
            };
        }
        // glTF import expands grayscale PNGs, but handle one/two channels too.
        if channels <= 2 {
            let alpha = if channels == 2 { components[1] } else { 255 };
            components = [components[0], components[0], components[0], alpha];
        }
        rgba.extend(components);
    }
    if rgba.len() != image.width as usize * image.height as usize * 4 {
        return Err("Invalid image byte count".into());
    }
    Ok(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_scalar_layout_matches_rust() {
        // These strides are used by the upstream GLSL buffer-reference arrays.
        assert_eq!(size_of::<Vertex>(), 44);
        assert_eq!(size_of::<Material>(), 20);
        assert_eq!(size_of::<RenderItem>(), 132);
        assert_eq!(std::mem::offset_of!(Vertex, uv), 36);
        assert_eq!(std::mem::offset_of!(Material, texture), 16);
        assert_eq!(std::mem::offset_of!(RenderItem, material), 128);
    }

    #[test]
    fn demo_import_preserves_shared_mesh_materials_and_parent_transforms() {
        let scene =
            Scene::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/demo.gltf")).unwrap();
        assert_eq!(scene.meshes.len(), 1);
        assert_eq!(scene.meshes[0].len(), 2);
        assert_eq!(scene.images.len(), 2);
        assert_eq!(scene.images[1].rgba.len(), 8 * 8 * 4);
        assert_eq!(scene.materials[1].texture, 1);
        assert_eq!(scene.materials[2].texture, 0);
        let vp = Camera::default().view_projection(16.0 / 9.0);
        let (commands, items) = scene.draws(vp).unwrap();
        assert_eq!(commands.len(), 4);
        for (index, (command, item)) in commands.iter().zip(&items).enumerate() {
            assert_eq!(command.first_instance as usize, index);
            assert_eq!(command.index_count, 18);
            assert_eq!(item.material as usize, index % 2 + 1);
            let child = if index < 2 { 2 } else { 1 };
            let expected_world = scene.nodes[0].local * scene.nodes[child].local;
            assert!(Mat4::from_cols_array(&item.world).abs_diff_eq(expected_world, 1e-6));
            assert!(Mat4::from_cols_array(&item.wvp).abs_diff_eq(vp * expected_world, 1e-6));
        }
        // Both nodes share geometry rather than duplicating the mesh's buffers.
        assert_eq!(commands[0].first_index, commands[2].first_index);
        assert_eq!(commands[1].vertex_offset, commands[3].vertex_offset);
    }

    #[test]
    fn cyclic_hierarchy_is_rejected() {
        let mut scene =
            Scene::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/demo.gltf")).unwrap();
        scene.nodes[1].children.push(0);
        assert!(scene.draws(Mat4::IDENTITY).is_err());
    }

    #[test]
    fn imports_camera_projection_and_parent_transform() {
        let scene =
            Scene::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/demo.gltf")).unwrap();
        let camera = scene.initial_camera().unwrap();
        let world = scene.nodes[0].local * scene.nodes[3].local;
        assert!(
            camera
                .position
                .abs_diff_eq(world.transform_point3(Vec3::ZERO), 1e-5)
        );
        let forward = camera.world_pose().transform_vector3(Vec3::NEG_Z);
        assert!(forward.abs_diff_eq(world.transform_vector3(Vec3::NEG_Z).normalize(), 1e-5));
        match camera.projection {
            Projection::Perspective { fov_y, near, far } => {
                assert!((fov_y - 65.0_f32.to_radians()).abs() < 1e-6);
                assert_eq!(near, 0.1);
                assert_eq!(far, Some(100.0));
            }
            _ => panic!("Expected demo perspective camera"),
        }
        assert_eq!(camera.attachment.unwrap().node, 3);
    }

    #[test]
    fn camera_selection_ignores_inactive_nodes_and_uses_fallback() {
        let mut scene =
            Scene::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/demo.gltf")).unwrap();
        // The camera still exists in nodes but is no longer in the active hierarchy.
        scene.nodes[0].children.retain(|&id| id != 3);
        let fallback = scene.initial_camera().unwrap();
        assert!(fallback.attachment.is_none());
        assert_eq!(fallback.position, Vec3::new(0.0, 1.0, 0.0));
        scene.nodes[1].camera = scene.nodes[3].camera;
        scene.nodes[2].camera = scene.nodes[3].camera;
        assert_eq!(scene.initial_camera().unwrap().attachment.unwrap().node, 1);
    }

    #[test]
    fn camera_motion_moves_attached_meshes_in_world_space() {
        let mut scene =
            Scene::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/demo.gltf")).unwrap();
        scene.nodes[0].children.retain(|&id| id != 1);
        scene.nodes[3].children.push(1);
        let mut camera = scene.initial_camera().unwrap();
        scene.update_camera(&camera);
        let before = scene.draws(Mat4::IDENTITY).unwrap().1;
        let delta = Vec3::new(0.7, 0.0, -0.4);
        camera.position += delta;
        scene.update_camera(&camera);
        let after = scene.draws(Mat4::IDENTITY).unwrap().1;
        // Stack traversal visits the camera's child before the other cube.
        for i in 0..2 {
            let old = Mat4::from_cols_array(&before[i].world).transform_point3(Vec3::ZERO);
            let new = Mat4::from_cols_array(&after[i].world).transform_point3(Vec3::ZERO);
            assert!((new - old).abs_diff_eq(delta, 1e-5));
        }
        assert_eq!(before[2].world, after[2].world);
    }

    #[test]
    fn converts_gray_alpha_and_sixteen_bit_images() {
        let gray = gltf::image::Data {
            width: 1,
            height: 1,
            format: gltf::image::Format::R8G8,
            pixels: vec![80, 128],
        };
        assert_eq!(rgba_pixels(&gray).unwrap(), [80, 80, 80, 128]);
        let rgb = gltf::image::Data {
            width: 1,
            height: 1,
            format: gltf::image::Format::R16G16B16,
            pixels: [65535_u16, 32768, 0]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect(),
        };
        assert_eq!(rgba_pixels(&rgb).unwrap(), [255, 127, 0, 255]);
    }
}
