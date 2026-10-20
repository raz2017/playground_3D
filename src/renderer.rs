//! Rust port of constref's modern-vulkan lesson_03.
//! Vulkan calls live in explicit unsafe blocks; resource ownership stays in structs.
//!
//! Vulkan setup follows this order: loader -> instance -> surface -> physical GPU
//! -> logical device/queue -> swapchain/pipeline -> recorded and submitted frames.
//! Rust's ownership tells us who must clean up each resource. Ash's handle values
//! do not automatically free Vulkan objects, so our Drop implementations do that.
//! `unsafe` marks calls whose pointer/lifetime/synchronization rules we must uphold;
//! it does not switch off Rust's usual type checking or error handling.
use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use sdl3::video::Window;
use std::{
    error::Error,
    ffi::{CStr, CString, c_void},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use vk_mem::Alloc;
mod scene_gpu;
use crate::{camera::Camera, scene::Scene};
use scene_gpu::SceneGpu;

/// A shared result type: success holds T, failure holds a heap-allocated error.
/// `dyn Error` lets SDL, Vulkan, shader compilation, and parsing errors use one type.
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
const API_VERSION: u32 = vk::API_VERSION_1_3;
const FRAMES_IN_FLIGHT: usize = 2;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
static VALIDATION_ERRORS: AtomicUsize = AtomicUsize::new(0);

/// Read the number of validation errors so the application can fail its smoke test.
/// Vulkan can invoke its callback from different threads, so the counter is atomic.
/// Relaxed ordering is enough: this counter does not synchronize other memory.
pub fn validation_errors() -> usize {
    VALIDATION_ERRORS.load(Ordering::Relaxed)
}

/// Receive diagnostic messages from Vulkan's loader and validation layers.
/// `extern "system"` uses the calling convention expected by Vulkan, which is
/// necessary when native code calls a Rust function through a function pointer.
/// The raw pointers are borrowed for this callback only. CStr reads the native
/// null-terminated message; to_string_lossy handles any invalid UTF-8 bytes.
/// Returning VK_FALSE means we do not ask validation to abort the Vulkan call.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
    }
    if !data.is_null() {
        // Vulkan supplies callback data and a null-terminated string for this call.
        let message = unsafe { (*data).p_message };
        if !message.is_null() {
            eprintln!(
                "Vulkan {severity:?}: {}",
                unsafe { CStr::from_ptr(message) }.to_string_lossy()
            );
        }
    }
    vk::FALSE
}

/// Describe which messages we want and which callback should receive them.
/// Ash's `default().method(...)` pattern fills a Vulkan configuration struct.
/// This value borrows no temporary data, so its lifetime can be `'static`.
/// Building this description does not create a Vulkan object yet.
fn debug_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
}

// Option fields let Drop clean up a partially initialized context as well.
struct Context {
    _entry: Entry,
    instance: Instance,
    debug: Option<debug_utils::Instance>,
    messenger: vk::DebugUtilsMessengerEXT,
    surface_api: surface::Instance,
    surface: vk::SurfaceKHR,
    physical: vk::PhysicalDevice,
    device: Option<Device>,
    queue_family: u32,
    queue: vk::Queue,
    allocator: Option<Arc<vk_mem::Allocator>>,
}

impl Context {
    /// Build the connection to Vulkan and choose a usable GPU.
    /// An instance connects the application to Vulkan; a physical device is a
    /// GPU we discover; a logical device is the API connection we create to it.
    /// The surface connects Vulkan to SDL's window. We select a queue family
    /// (a group of queues with the same capabilities) that can draw and present.
    /// `&Window` borrows the window; `Self` means Context inside this impl block.
    /// On any `?` failure, the partially built context is dropped and cleaned up.
    fn new(window: &Window, validation: bool) -> Result<Self> {
        VALIDATION_ERRORS.store(0, Ordering::Relaxed);
        // SAFETY: Input arrays live through their calls. Child handles are retained
        // by Context and destroyed before their parents in Drop.
        unsafe {
            let entry = Entry::load()?;
            if entry
                .try_enumerate_instance_version()?
                .unwrap_or(vk::API_VERSION_1_0)
                < API_VERSION
            {
                return Err("Vulkan 1.3 or newer is required".into());
            }
            // SDL supplies the platform-specific extensions needed for its surface.
            // CString owns C-compatible strings. Keep these alive while passing
            // their raw pointers to Vulkan; a pointer does not own its string.
            let mut extensions: Vec<CString> = window
                .vulkan_instance_extensions()?
                .into_iter()
                .map(CString::new)
                .collect::<std::result::Result<_, _>>()?;
            let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
            if validation {
                let available = entry.enumerate_instance_layer_properties()?;
                if !available.iter().any(|layer| {
                    CStr::from_ptr(layer.layer_name.as_ptr()) == c"VK_LAYER_KHRONOS_validation"
                }) {
                    return Err(
                        "Validation layer missing. Install the Vulkan SDK or use --no-validation."
                            .into(),
                    );
                }
                extensions.push(debug_utils::NAME.to_owned());
                extensions.push(ash::ext::validation_features::NAME.to_owned());
            }
            let extension_names: Vec<_> = extensions.iter().map(|name| name.as_ptr()).collect();
            let app_info = vk::ApplicationInfo::default()
                .application_name(c"Modern Vulkan - Rust")
                .api_version(API_VERSION);
            // Validation checks API usage; synchronization validation additionally
            // looks for conflicting GPU accesses without the required dependencies.
            let enabled = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
            let mut validation_info =
                vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled);
            let mut debug_create = debug_info();
            let mut create = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&extension_names);
            if validation {
                create = create
                    .enabled_layer_names(&layers)
                    .push_next(&mut debug_create)
                    .push_next(&mut validation_info);
            }
            let instance = entry.create_instance(&create, None)?;
            let surface_api = surface::Instance::new(&entry, &instance);
            let mut context = Self {
                _entry: entry,
                instance,
                surface_api,
                debug: None,
                messenger: vk::DebugUtilsMessengerEXT::null(),
                surface: vk::SurfaceKHR::null(),
                physical: vk::PhysicalDevice::null(),
                device: None,
                queue_family: 0,
                queue: vk::Queue::null(),
                allocator: None,
            };
            if validation {
                let debug = debug_utils::Instance::new(&context._entry, &context.instance);
                context.messenger = debug.create_debug_utils_messenger(&debug_info(), None)?;
                context.debug = Some(debug);
            }
            context.surface = window.vulkan_create_surface(context.instance.handle())?;
            // Hardware support must be queried before requesting features. Prefer
            // a discrete GPU, but only among GPUs that pass every required check.
            let mut candidates = Vec::new();
            for physical in context.instance.enumerate_physical_devices()? {
                let props = context.instance.get_physical_device_properties(physical);
                if props.api_version < API_VERSION {
                    continue;
                }
                let device_exts = context
                    .instance
                    .enumerate_device_extension_properties(physical)?;
                if !device_exts
                    .iter()
                    .any(|ext| CStr::from_ptr(ext.extension_name.as_ptr()) == swapchain::NAME)
                {
                    continue;
                }
                let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
                let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
                // Vulkan extends structs through a linked `pNext` chain. Ash's
                // push_next attaches these feature structs; the query fills them.
                let mut features = vk::PhysicalDeviceFeatures2::default()
                    .push_next(&mut features12)
                    .push_next(&mut features13);
                context
                    .instance
                    .get_physical_device_features2(physical, &mut features);
                let core_features = features.features;
                if features12.timeline_semaphore == vk::FALSE
                    || features12.buffer_device_address == vk::FALSE
                    || features12.scalar_block_layout == vk::FALSE
                    || features12.runtime_descriptor_array == vk::FALSE
                    || features12.shader_sampled_image_array_non_uniform_indexing == vk::FALSE
                    || features12.descriptor_binding_sampled_image_update_after_bind == vk::FALSE
                    || features12.descriptor_binding_partially_bound == vk::FALSE
                    || core_features.shader_int64 == vk::FALSE
                    || core_features.multi_draw_indirect == vk::FALSE
                    || core_features.draw_indirect_first_instance == vk::FALSE
                    || features13.dynamic_rendering == vk::FALSE
                    || features13.synchronization2 == vk::FALSE
                {
                    continue;
                }
                let depth_props = context
                    .instance
                    .get_physical_device_format_properties(physical, DEPTH_FORMAT);
                if !depth_props
                    .optimal_tiling_features
                    .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
                {
                    continue;
                }
                if context
                    .surface_api
                    .get_physical_device_surface_formats(physical, context.surface)?
                    .is_empty()
                    || context
                        .surface_api
                        .get_physical_device_surface_present_modes(physical, context.surface)?
                        .is_empty()
                {
                    continue;
                }
                for (index, family) in context
                    .instance
                    .get_physical_device_queue_family_properties(physical)
                    .iter()
                    .enumerate()
                {
                    if family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && context.surface_api.get_physical_device_surface_support(
                            physical,
                            index as u32,
                            context.surface,
                        )?
                    {
                        candidates.push((
                            props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
                            physical,
                            index as u32,
                        ));
                        break;
                    }
                }
            }
            candidates.sort_by_key(|candidate| candidate.0);
            let (_, physical, queue_family) = candidates.pop().ok_or("No GPU supports lesson 02: Vulkan 1.3, dynamic rendering, synchronization2, timeline semaphores, buffer device addresses, scalar layout, texture descriptor indexing, shaderInt64, multiDrawIndirect, drawIndirectFirstInstance, and presentation")?;
            context.physical = physical;
            context.queue_family = queue_family;
            let props = context.instance.get_physical_device_properties(physical);
            println!(
                "GPU: {}",
                CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy()
            );
            println!("Vulkan 1.3; validation: {validation}");
            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let device_extensions = [swapchain::NAME.as_ptr()];
            // Querying support above does not enable features. Request them again
            // here when creating the logical device we will submit commands to.
            let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
                .timeline_semaphore(true)
                .buffer_device_address(true)
                .scalar_block_layout(true)
                .runtime_descriptor_array(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                .descriptor_binding_sampled_image_update_after_bind(true)
                .descriptor_binding_partially_bound(true);
            let features = vk::PhysicalDeviceFeatures::default()
                .shader_int64(true)
                .multi_draw_indirect(true)
                .draw_indirect_first_instance(true);
            let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);
            let create = vk::DeviceCreateInfo::default()
                .enabled_features(&features)
                .queue_create_infos(&queues)
                .enabled_extension_names(&device_extensions)
                .push_next(&mut features12)
                .push_next(&mut features13);
            context.device = Some(context.instance.create_device(physical, &create, None)?);
            context.queue = context.device().get_device_queue(queue_family, 0);
            let mut allocator_info =
                vk_mem::AllocatorCreateInfo::new(&context.instance, context.device(), physical);
            allocator_info.vulkan_api_version = API_VERSION;
            allocator_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
            context.allocator = Some(Arc::new(vk_mem::Allocator::new(allocator_info)?));
            Ok(context)
        }
    }
    /// Borrow the initialized logical device without moving it out of Context.
    /// as_ref turns Option<Device> into Option<&Device>. expect documents the
    /// internal invariant that callers only use this helper after device creation.
    fn device(&self) -> &Device {
        self.device.as_ref().expect("device initialized")
    }
    /// Borrow the shared VMA allocator. VMA chooses and manages GPU memory blocks.
    /// Arc keeps one allocator alive while Context and swapchain resources share
    /// ownership; cloning an Arc increases a reference count, not the GPU allocation.
    fn allocator(&self) -> &Arc<vk_mem::Allocator> {
        self.allocator.as_ref().expect("allocator initialized")
    }
}

impl Drop for Context {
    /// Finish cleanup after the renderer has released its device resources.
    /// Drop is Rust's destructor trait: it runs automatically when an owner leaves
    /// scope, including early error returns. take() moves a value out of an Option
    /// and leaves None, allowing us to destroy parents after their children.
    fn drop(&mut self) {
        // Renderer has already destroyed all device children.
        unsafe {
            self.allocator.take();
            if let Some(device) = self.device.take() {
                device.destroy_device(None);
            }
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_api.destroy_surface(self.surface, None);
            }
            if let Some(debug) = &self.debug {
                debug.destroy_debug_utils_messenger(self.messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Default)]
// A frame slot is reusable CPU/GPU work storage, not a swapchain image index.
// completed_value tracks its last submission on the shared timeline semaphore.
struct Frame {
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    acquired: vk::Semaphore,
    completed_value: u64,
}

struct DepthImage {
    image: vk::Image,
    allocation: vk_mem::Allocation,
    view: vk::ImageView,
}

struct SwapchainResources {
    device: Device,
    allocator: Arc<vk_mem::Allocator>,
    api: swapchain::Device,
    handle: vk::SwapchainKHR,
    extent: vk::Extent2D,
    format: vk::Format,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    presented: Vec<vk::Semaphore>,
    depths: Vec<DepthImage>,
}

/// Select the part of an image that an image view or layout barrier refers to.
/// `aspect` chooses color or depth. This example uses mip level 0 and array layer
/// 0 only, with a count of one each; no mipmaps or layered textures are needed.
fn image_range(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .level_count(1)
        .layer_count(1)
}

impl SwapchainResources {
    /// Create the window's presentable images and the resources sized to match them.
    /// A swapchain cycles images between the renderer and the presentation system.
    /// Surface capabilities tell us the permitted size, format, and image count.
    /// An image view describes how Vulkan should interpret an image for rendering.
    /// VMA separately creates our depth images, which store distances for depth tests.
    /// `Result<Option<Self>>` distinguishes an error from a successful creation,
    /// or Ok(None) when the surface has no usable area (for example, minimized).
    fn new(context: &Context, width: u32, height: u32) -> Result<Option<Self>> {
        // The caller waits for device idle before replacing old swapchain resources.
        unsafe {
            let caps = context
                .surface_api
                .get_physical_device_surface_capabilities(context.physical, context.surface)?;
            // Some platforms dictate an exact pixel size. u32::MAX is Vulkan's
            // sentinel for "application chooses"; clamp that choice to valid limits.
            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                    height: height
                        .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
                }
            };
            if extent.width == 0 || extent.height == 0 {
                return Ok(None);
            }
            if !caps
                .supported_usage_flags
                .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            {
                return Err("Surface does not support color attachments".into());
            }
            let formats = context
                .surface_api
                .get_physical_device_surface_formats(context.physical, context.surface)?;
            let preferred = vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            };
            let format = if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
                preferred
            } else {
                *formats
                    .iter()
                    .find(|format| **format == preferred)
                    .unwrap_or(&formats[0])
            };
            let mut count = caps.min_image_count.saturating_add(1);
            if caps.max_image_count != 0 {
                count = count.min(caps.max_image_count);
            }
            let alpha = [
                vk::CompositeAlphaFlagsKHR::OPAQUE,
                vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
                vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
                vk::CompositeAlphaFlagsKHR::INHERIT,
            ]
            .into_iter()
            .find(|alpha| caps.supported_composite_alpha.contains(*alpha))
            .ok_or("No supported composite alpha mode")?;
            let api = swapchain::Device::new(&context.instance, context.device());
            // FIFO presents queued images in order at display refresh time.
            // EXCLUSIVE is sufficient because one queue family draws and presents.
            let create = vk::SwapchainCreateInfoKHR::default()
                .surface(context.surface)
                .min_image_count(count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(alpha)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true);
            let handle = api.create_swapchain(&create, None)?;
            let mut resources = Self {
                device: context.device().clone(),
                allocator: Arc::clone(context.allocator()),
                api,
                handle,
                extent,
                format: format.format,
                images: Vec::new(),
                views: Vec::new(),
                presented: Vec::new(),
                depths: Vec::new(),
            };
            resources.images = resources.api.get_swapchain_images(handle)?;
            for image in &resources.images {
                let create = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .subresource_range(image_range(vk::ImageAspectFlags::COLOR));
                resources
                    .views
                    .push(resources.device.create_image_view(&create, None)?);
                // Presentation semaphores belong to images, not frames: reacquiring
                // that image proves the previous presentation consumed its wait.
                resources.presented.push(
                    resources
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
                );
            }
            // Separate depth images avoid overlapping writes by in-flight frames.
            for _ in 0..FRAMES_IN_FLIGHT {
                let image_info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(DEPTH_FORMAT)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED);
                let alloc_info = vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::Auto,
                    flags: vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
                    ..Default::default()
                };
                let (image, allocation) =
                    resources.allocator.create_image(&image_info, &alloc_info)?;
                resources.depths.push(DepthImage {
                    image,
                    allocation,
                    view: vk::ImageView::null(),
                });
                let view = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(DEPTH_FORMAT)
                    .subresource_range(image_range(vk::ImageAspectFlags::DEPTH));
                resources.depths.last_mut().unwrap().view =
                    resources.device.create_image_view(&view, None)?;
            }
            Ok(Some(resources))
        }
    }
}

impl Drop for SwapchainResources {
    /// Release all resources owned by this swapchain, also after partial creation.
    /// Destroy views before their images, then free each depth image's VMA memory.
    /// Swapchain color images belong to the swapchain and must not be freed by VMA.
    /// The renderer waits for GPU work to finish before dropping a used swapchain.
    fn drop(&mut self) {
        unsafe {
            for depth in &mut self.depths {
                if depth.view != vk::ImageView::null() {
                    self.device.destroy_image_view(depth.view, None);
                }
                self.allocator
                    .destroy_image(depth.image, &mut depth.allocation);
            }
            for semaphore in &self.presented {
                self.device.destroy_semaphore(*semaphore, None);
            }
            for view in &self.views {
                self.device.destroy_image_view(*view, None);
            }
            self.api.destroy_swapchain(self.handle, None);
        }
    }
}

pub struct Renderer {
    context: Context,
    scene: Option<SceneGpu>,
    swapchain: Option<SwapchainResources>,
    frames: Vec<Frame>,
    timeline: vk::Semaphore,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pipeline_format: vk::Format,
    submitted: u64,
    resize: bool,
    recreations: u64,
}

impl Renderer {
    /// Prepare everything needed to draw: Context, synchronization, command buffers,
    /// the scene, pipeline layout, and first swapchain and graphics pipeline.
    /// A command pool owns memory for command buffers, which record GPU instructions.
    /// Two frame slots let the CPU prepare a frame while the GPU finishes another.
    /// The timeline semaphore tracks completed submissions using increasing numbers.
    /// Store each handle as soon as it is created so Drop can clean up on failure.
    pub fn new(window: &Window, validation: bool, scene: Scene) -> Result<Self> {
        let context = Context::new(window, validation)?;
        let mut renderer = Self {
            context,
            scene: None,
            swapchain: None,
            frames: Vec::new(),
            timeline: vk::Semaphore::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            pipeline_format: vk::Format::UNDEFINED,
            submitted: 0,
            resize: true,
            recreations: 0,
        };
        renderer.scene = Some(SceneGpu::new(&renderer.context, scene)?);
        unsafe {
            let device = renderer.context.device();
            let mut kind = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            renderer.timeline = device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )?;
            // A pipeline layout describes shader resource bindings and push constants.
            // Three buffer addresses reach the vertex shader through push constants;
            // the fragment shader samples the scene's global texture descriptor array.
            let layouts = [renderer.scene.as_ref().unwrap().layout];
            let constants = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .size(24)];
            renderer.pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&constants),
                None,
            )?;
            for _ in 0..FRAMES_IN_FLIGHT {
                renderer.frames.push(Frame::default());
                let frame = renderer.frames.last_mut().unwrap();
                frame.acquired =
                    device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                frame.pool = device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(renderer.context.queue_family),
                    None,
                )?;
                frame.command = device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(frame.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?[0];
            }
        }
        let (width, height) = window.size_in_pixels();
        if width != 0 && height != 0 {
            renderer.recreate_swapchain(width, height)?;
        }
        Ok(renderer)
    }
    /// Mark the swapchain for replacement on the next render call.
    /// `&mut self` is an exclusive mutable borrow: this method changes the renderer.
    /// Deferring the work lets the event loop report size changes cheaply.
    pub fn request_resize(&mut self) {
        self.resize = true;
    }
    /// Read how many frames were successfully submitted to the GPU queue.
    /// `&self` borrows the renderer without changing it; submission does not mean
    /// the GPU has finished yet. The timeline semaphore tells us about completion.
    pub fn submitted_frames(&self) -> u64 {
        self.submitted
    }
    /// Read the number of successful swapchain creations, including the initial one.
    /// The smoke test uses this to check that resize/restore triggered replacements.
    pub fn swapchain_creations(&self) -> u64 {
        self.recreations
    }

    /// Replace images that no longer match the window's size or surface state.
    /// Waiting for device idle makes it safe to destroy the old images and views.
    /// The pipeline can survive a size change because viewport/scissor are dynamic,
    /// but a color format change needs a pipeline built for that new format.
    /// If creation returns None, keep the resize flag set and try on a later frame.
    fn recreate_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
            self.context.device().device_wait_idle()?;
        }
        // Keep timeline values intact across resizes; release surface resources first.
        self.swapchain.take();
        self.swapchain = SwapchainResources::new(&self.context, width, height)?;
        if let Some(swapchain) = &self.swapchain {
            let format = swapchain.format;
            let extent = swapchain.extent;
            if format != self.pipeline_format {
                let pipeline = self.create_pipeline(format)?;
                unsafe {
                    if self.pipeline != vk::Pipeline::null() {
                        self.context.device().destroy_pipeline(self.pipeline, None);
                    }
                }
                self.pipeline = pipeline;
                self.pipeline_format = format;
            }
            self.resize = false;
            self.recreations += 1;
            println!("Swapchain: {} x {}", extent.width, extent.height);
        }
        Ok(())
    }

    /// Compile borrowed GLSL text (`&str`) into SPIR-V words with shaderc, then
    /// wrap those words in a Vulkan shader module for graphics pipeline creation.
    /// `kind` selects vertex or fragment compilation; `name` labels compiler errors.
    /// Vulkan copies the code during creation, so the temporary binary can go away.
    /// The returned raw handle still needs explicit destruction by the caller.
    fn compile_shader(
        &self,
        source: &str,
        name: &str,
        kind: shaderc::ShaderKind,
    ) -> Result<vk::ShaderModule> {
        let compiler = shaderc::Compiler::new()?;
        let mut options = shaderc::CompileOptions::new()?;
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_3 as u32,
        );
        options.set_target_spirv(shaderc::SpirvVersion::V1_6);
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        let binary = compiler.compile_into_spirv(source, kind, name, "main", Some(&options))?;
        unsafe {
            Ok(self.context.device().create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(binary.as_binary()),
                None,
            )?)
        }
    }

    /// Assemble the shaders and fixed rendering rules into a graphics pipeline.
    /// The vertex shader supplies positions; the fragment shader supplies color.
    /// Other settings define triangle assembly, culling, depth testing, and writes.
    /// Dynamic rendering declares attachment formats without creating a render pass.
    /// Shader modules are temporary: after pipeline creation they can be destroyed.
    /// include_str! embeds the shader files at Rust compile time as string slices.
    fn create_pipeline(&self, format: vk::Format) -> Result<vk::Pipeline> {
        let vertex = self.compile_shader(
            include_str!("shaders/shader.vert"),
            "shader.vert",
            shaderc::ShaderKind::Vertex,
        )?;
        let fragment = match self.compile_shader(
            include_str!("shaders/shader.frag"),
            "shader.frag",
            shaderc::ShaderKind::Fragment,
        ) {
            Ok(shader) => shader,
            Err(error) => {
                unsafe {
                    self.context.device().destroy_shader_module(vertex, None);
                }
                return Err(error);
            }
        };
        unsafe {
            let device = self.context.device();
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex)
                    .name(c"main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment)
                    .name(c"main"),
            ];
            // Vertex pulling: gl_VertexIndex addresses vertices in a storage buffer.
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let samples = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            // Smaller depth values win; clear to the far plane at 1.0.
            let depth = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS);
            let attachments = [vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
            // Supply viewport/scissor during command recording rather than baking
            // pixel dimensions into this pipeline; resizing can then reuse it.
            let states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
            let formats = [format];
            let mut rendering = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&formats)
                .depth_attachment_format(DEPTH_FORMAT);
            let create = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&samples)
                .depth_stencil_state(&depth)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(self.pipeline_layout)
                .push_next(&mut rendering);
            let result =
                device.create_graphics_pipelines(vk::PipelineCache::null(), &[create], None);
            device.destroy_shader_module(fragment, None);
            device.destroy_shader_module(vertex, None);
            match result {
                Ok(pipelines) => Ok(pipelines[0]),
                Err((partial, error)) => {
                    for pipeline in partial {
                        device.destroy_pipeline(pipeline, None);
                    }
                    Err(error.into())
                }
            }
        }
    }

    /// Draw and present one frame: wait for a reusable frame slot, acquire an image,
    /// record transitions and drawing commands, submit them, then request presentation.
    /// Recording commands does not execute them; queue_submit2 schedules GPU work.
    /// The acquire semaphore protects the image, the presentation semaphore orders
    /// display after rendering, and the timeline protects reuse of frame resources.
    /// Ok(()) may also mean no frame was submitted, such as during a surface resize.
    pub fn render(
        &mut self,
        width: u32,
        height: u32,
        diagnostics: Option<&str>,
        camera: &Camera,
    ) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        if self.resize {
            self.recreate_swapchain(width, height)?;
        }
        // `let ... else` extracts Some's value or returns early for None. Borrow
        // the swapchain instead of moving ownership of it out of the renderer.
        let Some(swap) = &self.swapchain else {
            return Ok(());
        };
        let frame_index = self.submitted as usize % FRAMES_IN_FLIGHT;
        let frame = &mut self.frames[frame_index];
        let device = self.context.device();
        // SAFETY: A frame slot is reused only after its timeline signal. Acquired
        // images belong to this submission until presentation. Arrays live through
        // calls; replacement and destruction wait for device work to complete.
        unsafe {
            // Wait on the CPU until this frame slot's previous GPU submission is
            // complete. Only then can we reset and overwrite its command buffer.
            if frame.completed_value != 0 {
                let semaphores = [self.timeline];
                let values = [frame.completed_value];
                device.wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&semaphores)
                        .values(&values),
                    u64::MAX,
                )?;
            }
            self.scene
                .as_mut()
                .unwrap()
                .prepare(frame_index, camera, swap.extent)?;
            // Ask for an available swapchain image. The returned index identifies
            // it; the binary semaphore tells GPU work when the image is ready to use.
            let (image_index, suboptimal) = match swap.api.acquire_next_image(
                swap.handle,
                u64::MAX,
                frame.acquired,
                vk::Fence::null(),
            ) {
                Ok(acquired) => acquired,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.resize = true;
                    return Ok(());
                }
                Err(error) => return Err(error.into()),
            };
            // Do not advance counters on a failed acquire: that would leave an
            // unsignaled timeline value that later frames could wait for forever.
            self.resize |= suboptimal;
            device.reset_command_pool(frame.pool, vk::CommandPoolResetFlags::empty())?;
            let command = frame.command;
            device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let image = swap.images[image_index as usize];
            let depth = &swap.depths[frame_index];
            let depth_stages = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
            // A layout describes how Vulkan may use an image; a barrier transitions
            // that layout and orders memory accesses. UNDEFINED discards old contents,
            // which is fine because we clear both attachments on every frame.
            // IGNORED means these barriers do not transfer queue-family ownership.
            let barriers = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(image_range(vk::ImageAspectFlags::COLOR)),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(depth_stages)
                    .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_stage_mask(depth_stages)
                    .dst_access_mask(
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(depth.image)
                    .subresource_range(image_range(vk::ImageAspectFlags::DEPTH)),
            ];
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
            // CLEAR initializes the background; STORE preserves color for presentation.
            // Depth is cleared for comparisons but is not needed after this frame.
            let color = [vk::RenderingAttachmentInfo::default()
                .image_view(swap.views[image_index as usize])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.15, 0.15, 1.0, 1.0],
                    },
                })];
            let depth_attach = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });
            let area = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: swap.extent,
            };
            device.cmd_begin_rendering(
                command,
                &vk::RenderingInfo::default()
                    .render_area(area)
                    .layer_count(1)
                    .color_attachments(&color)
                    .depth_attachment(&depth_attach),
            );
            // The viewport maps shader coordinates to pixels. The scissor clips
            // drawing to a rectangle. Both cover the entire swapchain image here.
            // Negative height flips Y while preserving the camera's upward Y axis.
            let viewport = vk::Viewport {
                x: 0.0,
                y: swap.extent.height as f32,
                width: swap.extent.width as f32,
                height: -(swap.extent.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(command, 0, &[viewport]);
            device.cmd_set_scissor(command, 0, &[area]);
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.scene
                .as_ref()
                .unwrap()
                .draw(command, self.pipeline_layout, frame_index);
            if let Some(label) = diagnostics {
                crate::diagnostics::draw(device, command, swap.extent, label);
            }
            device.cmd_end_rendering(command);
            // Finish color writes and transition into the layout required by the
            // presentation system. No subsequent drawing stage reads this image here.
            let present_barrier = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(image_range(vk::ImageAspectFlags::COLOR))];
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&present_barrier),
            );
            device.end_command_buffer(command)?;
            // Submit waits for acquisition before color output and signals both
            // presentation readiness and the new timeline value after all commands.
            // `&[value]` / arrays passed by reference are borrowed slices, not copies
            // of Vulkan resources. These configuration arrays live through the call.
            let signal_value = self.submitted + 1;
            let wait = [vk::SemaphoreSubmitInfo::default()
                .semaphore(frame.acquired)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
            let signals = [
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(swap.presented[image_index as usize])
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.timeline)
                    .value(signal_value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            ];
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
            let submit = [vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait)
                .command_buffer_infos(&commands)
                .signal_semaphore_infos(&signals)];
            device.queue_submit2(self.context.queue, &submit, vk::Fence::null())?;
            frame.completed_value = signal_value;
            self.submitted += 1;
            let semaphores = [swap.presented[image_index as usize]];
            let handles = [swap.handle];
            let indices = [image_index];
            // Presentation waits on the binary semaphore; it does not make the CPU
            // wait for rendering. Handle surface changes here as well as at acquisition.
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&semaphores)
                .swapchains(&handles)
                .image_indices(&indices);
            match swap.api.queue_present(self.context.queue, &present) {
                Ok(suboptimal) => self.resize |= suboptimal,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.resize = true,
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }
}

impl Drop for Renderer {
    /// Stop GPU use of our resources, then destroy pipelines, swapchain, command
    /// pools, and semaphores. Destroying a command pool also frees its buffers.
    /// Context's Drop runs afterward to release the allocator, device, and instance.
    /// Drop cannot return an error; `let _ =` deliberately ignores the idle result
    /// so we still attempt cleanup if the device has already failed.
    fn drop(&mut self) {
        unsafe {
            let device = self.context.device();
            let _ = device.device_wait_idle();
            if self.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
            self.swapchain.take();
            self.scene.take();
            for frame in &self.frames {
                if frame.pool != vk::CommandPool::null() {
                    device.destroy_command_pool(frame.pool, None);
                }
                if frame.acquired != vk::Semaphore::null() {
                    device.destroy_semaphore(frame.acquired, None);
                }
            }
            if self.timeline != vk::Semaphore::null() {
                device.destroy_semaphore(self.timeline, None);
            }
        }
    }
}
