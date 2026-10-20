//! Own the lesson's scene buffers, texture array, and per-frame draw data.
//! Uploads finish before their staging resources are released. Host writes are
//! flushed for non-coherent memory; frame data is rewritten only after its wait.
use super::{Context, FRAMES_IN_FLIGHT, Result, image_range};
use crate::{camera::Camera, scene::Scene};
use ash::{Device, vk};
use std::sync::Arc;
use vk_mem::Alloc;

/// Own a Vulkan buffer and its VMA allocation together. pub(super) exposes this
/// type to the parent renderer module. Arc shares the allocator's ownership;
/// it does not by itself ensure that the GPU has finished using this buffer.
pub(super) struct Buffer {
    allocator: Arc<vk_mem::Allocator>,
    pub handle: vk::Buffer,
    allocation: vk_mem::Allocation,
    pub address: u64,
    size: usize,
}

impl Buffer {
    /// Allocate CPU-writable memory for frame/staging data, or prefer GPU memory
    /// for static resources. Usage flags declare which operations are permitted.
    fn new(
        context: &Context,
        size: usize,
        usage: vk::BufferUsageFlags,
        host: bool,
    ) -> Result<Self> {
        let allocator = context.allocator().clone();
        let info = vk::BufferCreateInfo::default()
            .size(size.max(4) as u64)
            .usage(usage);
        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: if host {
                vk_mem::MemoryUsage::AutoPreferHost
            } else {
                vk_mem::MemoryUsage::AutoPreferDevice
            },
            flags: if host {
                vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
            } else {
                vk_mem::AllocationCreateFlags::empty()
            },
            ..Default::default()
        };
        let (handle, allocation) = unsafe { allocator.create_buffer(&info, &allocation_info)? };
        // This is a GPU address for shader buffer references, not a CPU pointer.
        let address = if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            unsafe {
                context.device().get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(handle),
                )
            }
        } else {
            0
        };
        Ok(Self {
            allocator,
            handle,
            allocation,
            address,
            size,
        })
    }

    /// Copy a borrowed byte slice into host-accessible memory. The caller must
    /// ensure no pending GPU work reads the bytes being overwritten.
    fn write(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err("Scene buffer capacity exceeded".into());
        }
        unsafe {
            // Mapping gives a CPU pointer. Flush makes writes visible even when
            // memory is not host-coherent; unmap releases this temporary mapping.
            let mapped = self.allocator.map_memory(&mut self.allocation)?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped, data.len());
            let result = self
                .allocator
                .flush_allocation(&self.allocation, 0, data.len() as u64);
            self.allocator.unmap_memory(&mut self.allocation);
            result?;
        }
        Ok(())
    }

    /// Stage static data in CPU-visible memory and record a copy to GPU memory.
    /// Recording is deferred execution, so Upload retains the staging buffer.
    fn upload(
        context: &Context,
        upload: &mut Upload,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let mut staging = Self::new(
            context,
            data.len(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            true,
        )?;
        staging.write(data)?;
        let buffer = Self::new(
            context,
            data.len(),
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            false,
        )?;
        unsafe {
            context.device().cmd_copy_buffer(
                upload.command,
                staging.handle,
                buffer.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: data.len() as u64,
                }],
            );
        }
        upload.staging.push(staging);
        Ok(buffer)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_buffer(self.handle, &mut self.allocation);
        }
    }
}

// An image owns pixels/memory; its view describes how shaders access those pixels.
// Cloning ash::Device copies the API wrapper, not the underlying Vulkan device.
struct TextureImage {
    device: Device,
    allocator: Arc<vk_mem::Allocator>,
    image: vk::Image,
    allocation: vk_mem::Allocation,
    view: vk::ImageView,
}

impl TextureImage {
    fn upload(
        context: &Context,
        upload: &mut Upload,
        source: &crate::scene::Image,
    ) -> Result<Self> {
        let allocator = context.allocator().clone();
        let device = context.device().clone();
        let extent = vk::Extent3D {
            width: source.width,
            height: source.height,
            depth: 1,
        };
        // Base-color textures use sRGB: sampling decodes RGB to linear values
        // for lighting. Alpha is not sRGB-decoded. This lesson uploads one mip.
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST);
        let (image, allocation) = unsafe {
            allocator.create_image(
                &info,
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    ..Default::default()
                },
            )?
        };
        let mut result = Self {
            device,
            allocator,
            image,
            allocation,
            view: vk::ImageView::null(),
        };
        unsafe {
            result.view = result.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_SRGB)
                    .subresource_range(image_range(vk::ImageAspectFlags::COLOR)),
                None,
            )?;
        }
        let mut staging = Buffer::new(
            context,
            source.rgba.len(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            true,
        )?;
        staging.write(&source.rgba)?;
        unsafe {
            // Discard initial contents and transition to the layout needed for
            // transfer writes. Layout transitions are recorded GPU operations.
            let before = [vk::ImageMemoryBarrier2::default()
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(image_range(vk::ImageAspectFlags::COLOR))];
            result.device.cmd_pipeline_barrier2(
                upload.command,
                &vk::DependencyInfo::default().image_memory_barriers(&before),
            );
            let region = vk::BufferImageCopy::default()
                .image_extent(extent)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                );
            result.device.cmd_copy_buffer_to_image(
                upload.command,
                staging.handle,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
            // Make copy writes visible to fragment sampling and switch layouts.
            // Stage masks identify operations; access masks identify memory uses.
            let after = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(image_range(vk::ImageAspectFlags::COLOR))];
            result.device.cmd_pipeline_barrier2(
                upload.command,
                &vk::DependencyInfo::default().image_memory_barriers(&after),
            );
        }
        upload.staging.push(staging);
        Ok(result)
    }
}

impl Drop for TextureImage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.allocator
                .destroy_image(self.image, &mut self.allocation);
        }
    }
}

/// One startup command buffer batches copies. Waiting once at finish is simple
/// for static scene loading; normal rendering instead waits per frame slot.
struct Upload {
    device: Device,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    staging: Vec<Buffer>,
}

impl Upload {
    fn new(context: &Context) -> Result<Self> {
        let device = context.device().clone();
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.queue_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )?
        };
        let mut upload = Self {
            device,
            pool,
            command: vk::CommandBuffer::null(),
            staging: Vec::new(),
        };
        unsafe {
            upload.command = upload.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            upload.device.begin_command_buffer(
                upload.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        Ok(upload)
    }

    fn finish(&self, context: &Context) -> Result<()> {
        unsafe {
            // Buffer copies must become visible to vertex pulling and index fetch.
            // Texture sampling dependencies were recorded per image above.
            let barriers = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::INDEX_INPUT,
                )
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::INDEX_READ,
                )];
            self.device.cmd_pipeline_barrier2(
                self.command,
                &vk::DependencyInfo::default().memory_barriers(&barriers),
            );
            self.device.end_command_buffer(self.command)?;
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(self.command)];
            self.device.queue_submit2(
                context.queue,
                &[vk::SubmitInfo2::default().command_buffer_infos(&commands)],
                vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(context.queue)?;
        }
        Ok(())
    }
}

impl Drop for Upload {
    fn drop(&mut self) {
        unsafe {
            // Also covers an error after submission: staging memory stays alive until idle.
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}

// Each frame slot has its own writable data so the CPU does not overwrite the
// previous frame's commands/matrices while the GPU is still reading them.
struct FrameData {
    indirect: Buffer,
    items: Buffer,
}

pub(super) struct SceneGpu {
    device: Device,
    pub layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    images: Vec<TextureImage>,
    samplers: Vec<vk::Sampler>,
    vertex: Option<Buffer>,
    index: Option<Buffer>,
    material: Option<Buffer>,
    frames: Vec<FrameData>,
    draw_count: u32,
    max_draw_count: u32,
    pub scene: Scene,
}

impl SceneGpu {
    pub fn new(context: &Context, scene: Scene) -> Result<Self> {
        let count = u32::try_from(scene.draws(glam::Mat4::IDENTITY)?.0.len())?;
        let props = unsafe {
            context
                .instance
                .get_physical_device_properties(context.physical)
        };
        let mut indexing = vk::PhysicalDeviceDescriptorIndexingProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut indexing);
        unsafe {
            context
                .instance
                .get_physical_device_properties2(context.physical, &mut properties);
        }
        let texture_count = u32::try_from(scene.textures.len())?;
        if texture_count > indexing.max_descriptor_set_update_after_bind_sampled_images
            || texture_count > indexing.max_descriptor_set_update_after_bind_samplers
            || texture_count > indexing.max_per_stage_descriptor_update_after_bind_sampled_images
            || texture_count > indexing.max_per_stage_descriptor_update_after_bind_samplers
            || texture_count > indexing.max_per_stage_update_after_bind_resources
            || texture_count > indexing.max_update_after_bind_descriptors_in_all_pools
        {
            return Err("Scene exceeds GPU texture descriptor limits".into());
        }
        let mut gpu = Self {
            device: context.device().clone(),
            layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            images: Vec::new(),
            samplers: Vec::new(),
            vertex: None,
            index: None,
            material: None,
            frames: Vec::new(),
            draw_count: count,
            max_draw_count: props.limits.max_draw_indirect_count,
            scene,
        };
        // Declare upload after gpu: on failure it waits before gpu resources are dropped.
        let mut upload = Upload::new(context)?;
        // bytemuck::cast_slice borrows Pod arrays as bytes without allocating or
        // serializing. The Rust/GLSL layout agreement is checked in scene's tests.
        let storage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        gpu.vertex = Some(Buffer::upload(
            context,
            &mut upload,
            bytemuck::cast_slice(&gpu.scene.vertices),
            storage,
        )?);
        gpu.index = Some(Buffer::upload(
            context,
            &mut upload,
            bytemuck::cast_slice(&gpu.scene.indices),
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?);
        gpu.material = Some(Buffer::upload(
            context,
            &mut upload,
            bytemuck::cast_slice(&gpu.scene.materials),
            storage,
        )?);
        for source in &gpu.scene.images {
            gpu.images
                .push(TextureImage::upload(context, &mut upload, source)?);
        }
        upload.finish(context)?;
        for texture in &gpu.scene.textures {
            let sampler = unsafe {
                gpu.device.create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(texture.mag)
                        .min_filter(texture.min)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(texture.wrap_u)
                        .address_mode_v(texture.wrap_v)
                        .address_mode_w(vk::SamplerAddressMode::REPEAT)
                        .max_lod(0.0),
                    None,
                )?
            };
            gpu.samplers.push(sampler);
        }
        unsafe {
            // One descriptor set exposes all textures at binding 0. A descriptor
            // pairs each image view with a sampler; material indices select entries.
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(texture_count)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            // These match the descriptor-indexing features enabled on the device.
            // This scene fills every entry and writes them only during startup.
            let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];
            let mut binding_flags =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
            gpu.layout = gpu.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .bindings(&bindings)
                    .push_next(&mut binding_flags),
                None,
            )?;
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: texture_count,
            }];
            gpu.pool = gpu.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?;
            gpu.set = gpu.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(gpu.pool)
                    .set_layouts(&[gpu.layout]),
            )?[0];
            let infos: Vec<_> = gpu
                .scene
                .textures
                .iter()
                .enumerate()
                .map(|(i, texture)| {
                    vk::DescriptorImageInfo::default()
                        .sampler(gpu.samplers[i])
                        .image_view(gpu.images[texture.image].view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                })
                .collect();
            gpu.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(gpu.set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&infos)],
                &[],
            );
        }
        for _ in 0..FRAMES_IN_FLIGHT {
            gpu.frames.push(FrameData {
                indirect: Buffer::new(
                    context,
                    count as usize * size_of::<vk::DrawIndexedIndirectCommand>(),
                    vk::BufferUsageFlags::INDIRECT_BUFFER,
                    true,
                )?,
                items: Buffer::new(
                    context,
                    count as usize * size_of::<crate::scene::RenderItem>(),
                    storage,
                    true,
                )?,
            });
        }
        Ok(gpu)
    }

    /// Refresh a frame slot after Renderer has waited for its timeline value.
    pub fn prepare(&mut self, frame: usize, camera: &Camera, extent: vk::Extent2D) -> Result<()> {
        self.scene.update_camera(camera);
        let (commands, items) = self
            .scene
            .draws(camera.view_projection(extent.width as f32 / extent.height as f32))?;
        // Ash's command is repr(C), five initialized 32-bit fields without padding.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                commands.as_ptr().cast::<u8>(),
                commands.len() * size_of::<vk::DrawIndexedIndirectCommand>(),
            )
        };
        self.frames[frame].indirect.write(bytes)?;
        self.frames[frame]
            .items
            .write(bytemuck::cast_slice(&items))?;
        Ok(())
    }

    /// Record indexed draws; the GPU later reads draw arguments from a buffer
    /// (indirect drawing), instead of receiving each argument in a separate call.
    pub fn draw(&self, command: vk::CommandBuffer, layout: vk::PipelineLayout, frame: usize) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[self.set],
                &[],
            );
            // Push constants carry only three u64 addresses (24 bytes). Actual
            // vertices, materials, and matrices remain in the referenced buffers.
            let addresses = [
                self.vertex.as_ref().unwrap().address,
                self.material.as_ref().unwrap().address,
                self.frames[frame].items.address,
            ];
            self.device.cmd_push_constants(
                command,
                layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::cast_slice(&addresses),
            );
            self.device.cmd_bind_index_buffer(
                command,
                self.index.as_ref().unwrap().handle,
                0,
                vk::IndexType::UINT32,
            );
            // Split unusually large scenes at the hardware multi-draw limit.
            let mut first = 0;
            while first < self.draw_count {
                let count = (self.draw_count - first).min(self.max_draw_count);
                self.device.cmd_draw_indexed_indirect(
                    command,
                    self.frames[frame].indirect.handle,
                    first as u64 * size_of::<vk::DrawIndexedIndirectCommand>() as u64,
                    count,
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );
                first += count;
            }
        }
    }
}

impl Drop for SceneGpu {
    fn drop(&mut self) {
        unsafe {
            // Renderer normally waits first; this also covers initialization failures.
            let _ = self.device.device_wait_idle();
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.destroy_descriptor_set_layout(self.layout, None);
            for sampler in &self.samplers {
                self.device.destroy_sampler(*sampler, None);
            }
        }
    }
}
