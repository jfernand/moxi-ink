//! The simulation on the GPU, with wgpu compute shaders (`shaders/moxi.wgsl`).
//!
//! [`GpuInk`] runs the same passes as [`Ink::step`], with the same arithmetic, on any wgpu device:
//! pass it the game's or tool's own device, or make one with [`default_device`]. Results match the
//! CPU closely but not bit for bit, since GPUs may fuse multiply–adds and divide less exactly; the
//! wet front's pinning is a threshold, so tiny differences can move it by a site here and there.
//!
//! Everything is `f32`, so none of the half-float problems the paper's authors met (lost pigment,
//! paling, quantised transfers) apply.
//!
//! ```no_run
//! use moxi_ink::{Ink, Load, Paper, Params, footprint, gpu};
//!
//! let (device, queue) = gpu::default_device().expect("a GPU");
//! let mut ink = gpu::GpuInk::new(&device, &queue, &Ink::new(Paper::generate(256, 256, 1), Params::default()));
//! ink.deposit(&footprint::disc(256, 256, (128.0, 128.0), 20.0, 2.0), Load::ink(5.0, 0.3));
//! ink.step(2000);
//! let done: Ink = ink.read();
//! ```

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::ink::{Ink, Load};
use crate::params::Params;

const CELL: usize = 12;
const SHADER: &str = include_str!("../shaders/moxi.wgsl");
const ENTRY_POINTS: [&str; 7] =
    ["block_and_pin", "collide", "stream", "gather_water", "fix_ink", "move_ink", "deposit"];

/// A device and queue on the most capable adapter (a discrete GPU where there is one), or `None`
/// without any.
pub fn default_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        };
        let adapter = instance.request_adapter(&options).await.ok()?;
        adapter.request_device(&wgpu::DeviceDescriptor::default()).await.ok()
    })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    omega: f32,
    advection: f32,
    evaporation: f32,
    edge_evaporation: f32,
    block_base: f32,
    block_grain: f32,
    block_alum: f32,
    block_glue: f32,
    block_fixed: f32,
    pin_base: f32,
    pin_fixed: f32,
    pin_disorder: f32,
    toe_glue: f32,
    pin_diagonal: f32,
    capacity: f32,
    surface_capacity: f32,
    receptivity: f32,
    base_mask: f32,
    hindrance: f32,
    hindrance_speed: f32,
    hindrance_still: f32,
    fix_base: f32,
    fix_dryness: f32,
    fix_glue: f32,
    load_water: f32,
    load_glue: f32,
    load_pigment: [f32; 4],
}

impl Uniforms {
    fn new(width: usize, height: usize, p: &Params, load: Load) -> Self {
        Self {
            width: width as u32,
            height: height as u32,
            omega: p.omega,
            advection: p.advection,
            evaporation: p.evaporation,
            edge_evaporation: p.edge_evaporation,
            block_base: p.block_base,
            block_grain: p.block_grain,
            block_alum: p.block_alum,
            block_glue: p.block_glue,
            block_fixed: p.block_fixed,
            pin_base: p.pin_base,
            pin_fixed: p.pin_fixed,
            pin_disorder: p.pin_disorder,
            toe_glue: p.toe_glue,
            pin_diagonal: p.pin_diagonal,
            capacity: p.capacity,
            surface_capacity: p.surface_capacity,
            receptivity: p.receptivity,
            base_mask: p.base_mask,
            hindrance: p.hindrance,
            hindrance_speed: p.hindrance_speed,
            hindrance_still: p.hindrance_still,
            fix_base: p.fix_base,
            fix_dryness: p.fix_dryness,
            fix_glue: p.fix_glue,
            load_water: load.water,
            load_glue: load.glue,
            load_pigment: [load.pigment[0], load.pigment[1], load.pigment[2], 0.0],
        }
    }
}

/// Ink on paper, simulated on a GPU.
pub struct GpuInk {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The paper and constants, for [`GpuInk::read`]; change constants with [`GpuInk::set_params`].
    template: Ink,
    uniforms: wgpu::Buffer,
    f: wgpu::Buffer,
    f_out: wgpu::Buffer,
    cell: wgpu::Buffer,
    ink: wgpu::Buffer,
    ink_out: wgpu::Buffer,
    layers: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipelines: Vec<wgpu::ComputePipeline>,
}

impl GpuInk {
    /// Uploads `ink` (its paper, constants and current state) to `device`.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, ink: &Ink) -> Self {
        let n = ink.width * ink.height;
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let buffer = |label: &str, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: storage,
            })
        };
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("moxi uniforms"),
            contents: bytemuck::bytes_of(&Uniforms::new(ink.width, ink.height, &ink.params, Load::ink(0.0, 0.0))),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let f = buffer("moxi f", bytemuck::cast_slice(&ink.f));
        let f_out = buffer("moxi f_out", bytemuck::cast_slice(&ink.f));
        let cells: Vec<f32> = (0..n)
            .flat_map(|c| {
                let mut v = [0.0f32; CELL];
                v[0] = ink.water[c];
                v[1] = ink.last_water[c];
                v[2] = ink.velocity[c][0];
                v[3] = ink.velocity[c][1];
                v[4] = ink.surface_water[c];
                v[5] = ink.seep[c];
                v[6] = if ink.block[c].is_infinite() { 1e30 } else { ink.block[c] };
                v[7] = ink.stain[c];
                v
            })
            .collect();
        let cell = buffer("moxi cell", bytemuck::cast_slice(&cells));
        let flow = buffer("moxi ink", bytemuck::cast_slice(&ink.flow_ink));
        let ink_out = buffer("moxi ink_out", bytemuck::cast_slice(&ink.flow_ink));
        let layer_data: Vec<[f32; 4]> = (0..n).flat_map(|c| [ink.surface_ink[c], ink.fixed_ink[c]]).collect();
        let layers = buffer("moxi layers", bytemuck::cast_slice(&layer_data));
        let paper_data: Vec<[f32; 4]> = (0..n)
            .map(|c| [ink.paper.grain[c], ink.paper.alum[c], ink.paper.pinning[c], ink.paper.rougher[c]])
            .collect();
        let paper = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("moxi paper"),
            contents: bytemuck::cast_slice(&paper_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("moxi"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, rw),
                entry(2, rw),
                entry(3, rw),
                entry(4, rw),
                entry(5, rw),
                entry(6, rw),
                entry(7, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });
        let buffers = [&uniforms, &f, &f_out, &cell, &flow, &ink_out, &layers, &paper];
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(k, b)| wgpu::BindGroupEntry { binding: k as u32, resource: b.as_entire_binding() })
            .collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("moxi"),
            layout: &layout,
            entries: &entries,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("moxi"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("moxi"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ENTRY_POINTS
            .iter()
            .map(|name| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        let mut template = Ink::new(ink.paper.clone(), ink.params);
        template.params = ink.params;
        Self {
            device: device.clone(),
            queue: queue.clone(),
            template,
            uniforms,
            f,
            f_out,
            cell,
            ink: flow,
            ink_out,
            layers,
            bind_group,
            pipelines,
        }
    }

    /// Sites per row and rows.
    pub fn size(&self) -> (usize, usize) {
        (self.template.width, self.template.height)
    }

    /// Changes the constants for the following steps.
    pub fn set_params(&mut self, params: Params) {
        self.template.params = params;
        self.write_uniforms(Load::ink(0.0, 0.0));
    }

    fn write_uniforms(&self, load: Load) {
        let (w, h) = self.size();
        self.queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&Uniforms::new(w, h, &self.template.params, load)),
        );
    }

    fn dispatch(&self, pass: &mut wgpu::ComputePass, pipeline: usize) {
        let (w, h) = self.size();
        pass.set_pipeline(&self.pipelines[pipeline]);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(w.div_ceil(8) as u32, h.div_ceil(8) as u32, 1);
    }

    /// [`Ink::deposit`] on the GPU.
    ///
    /// # Panics
    ///
    /// If `footprint` does not hold one weight per site.
    pub fn deposit(&mut self, footprint: &[f32], load: Load) {
        let (w, h) = self.size();
        assert_eq!(footprint.len(), w * h, "one footprint weight per site");
        let weights: Vec<[f32; 4]> = footprint.iter().map(|&x| [x, 0.0, 0.0, 0.0]).collect();
        self.queue.write_buffer(&self.ink_out, 0, bytemuck::cast_slice(&weights));
        self.write_uniforms(load);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.dispatch(&mut pass, 6);
        }
        self.queue.submit([encoder.finish()]);
        self.write_uniforms(Load::ink(0.0, 0.0));
    }

    /// Runs `steps` of [`Ink::step`], submitted in batches so no single submission runs long
    /// enough for a driver to give up on it.
    pub fn step(&mut self, steps: usize) {
        const BATCH: usize = 100;
        let mut left = steps;
        while left > 0 {
            let batch = left.min(BATCH);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            for _ in 0..batch {
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    for pipeline in 0..6 {
                        self.dispatch(&mut pass, pipeline);
                    }
                }
                encoder.copy_buffer_to_buffer(&self.f_out, 0, &self.f, 0, None);
                encoder.copy_buffer_to_buffer(&self.ink_out, 0, &self.ink, 0, None);
            }
            self.queue.submit([encoder.finish()]);
            left -= batch;
        }
    }

    /// Waits for the GPU and returns the current state as an [`Ink`], to render, inspect or carry
    /// on with on the CPU.
    pub fn read(&self) -> Ink {
        let f: Vec<[f32; 9]> = self.download(&self.f);
        let cells: Vec<[f32; CELL]> = self.download(&self.cell);
        let flow: Vec<[f32; 4]> = self.download(&self.ink);
        let layers: Vec<[f32; 4]> = self.download(&self.layers);
        let mut ink = self.template.clone();
        for (c, v) in cells.iter().enumerate() {
            ink.water[c] = v[0];
            ink.last_water[c] = v[1];
            ink.velocity[c] = [v[2], v[3]];
            ink.surface_water[c] = v[4];
            ink.seep[c] = v[5];
            ink.block[c] = if v[6] > 1e29 { f32::INFINITY } else { v[6] };
            ink.stain[c] = v[7];
            ink.surface_ink[c] = layers[2 * c];
            ink.fixed_ink[c] = layers[2 * c + 1];
        }
        ink.f = f;
        ink.flow_ink = flow;
        ink
    }

    fn download<T: Pod>(&self, buffer: &wgpu::Buffer) -> Vec<T> {
        let size = buffer.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("moxi read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |r| r.expect("mapping a read-back buffer"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("waiting for the GPU");
        let data = bytemuck::cast_slice(&staging.get_mapped_range(..)).to_vec();
        staging.unmap();
        data
    }
}
