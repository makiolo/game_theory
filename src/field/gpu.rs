//! Backend de GPU: la rejilla se difunde en un compute shader.
//!
//! Usa el dispositivo que Bevy ya tiene abierto ([`RenderDevice`]), no uno
//! propio. Anadir `wgpu` como dependencia directa habria creado un segundo
//! `Device` — y ademas una version distinta de la que usa Bevy por dentro, con
//! tipos incompatibles.
//!
//! El paso es sincrono: sube la rejilla, encadena todos los sub-pasos en la
//! tarjeta y baja el resultado bloqueando hasta que esta listo. No es la forma
//! mas eficiente de usar una GPU, pero es la que permite compararlo de igual a
//! igual con los backends de CPU bajo el mismo trait, y es lo que necesita el
//! acoplamiento, que vive en CPU junto a Rapier y quiere el campo del frame.
//!
//! Los sub-pasos intermedios no salen de la tarjeta: solo hay una subida y una
//! bajada por frame, que es donde se va casi todo el tiempo.

use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType,
    Buffer, BufferBinding, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    PipelineLayoutDescriptor, PollType, RawComputePipelineDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bytemuck::{Pod, Zeroable};
use ndarray::Array2;
use std::borrow::Cow;
use std::num::NonZeroU64;

use super::{DiffusionParams, FieldBackend};

/// Lado del grupo de trabajo; tiene que coincidir con el `@workgroup_size` del
/// shader.
const WORKGROUP: u32 = 8;

/// Espejo de `Params` en el shader. El relleno es explicito porque un uniform
/// se alinea a 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    kx: f32,
    ky: f32,
    decay: f32,
    ambient: f32,
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
}

/// Buffers ligados a un tamano de rejilla concreto. Se rehacen si cambia.
struct Buffers {
    dims: (usize, usize),
    /// Los dos lados del ping-pong.
    a: Buffer,
    b: Buffer,
    params: Buffer,
    /// Buffer mapeable a CPU al que se copia el resultado final.
    staging: Buffer,
    /// Lee de `a` y escribe en `b`, y al reves.
    bind_ab: BindGroup,
    bind_ba: BindGroup,
}

pub struct GpuBackend {
    device: RenderDevice,
    queue: RenderQueue,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
    buffers: Option<Buffers>,
}

impl GpuBackend {
    pub fn new(device: RenderDevice, queue: RenderQueue) -> Self {
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("brownian_diffuse"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("diffuse.wgsl"))),
        });

        let uniform_entry = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(size_of::<GpuParams>() as u64),
            },
            count: None,
        };
        let storage_entry = |binding: u32, read_only: bool| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let layout = device.create_bind_group_layout(
            "brownian_diffuse_layout",
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        );

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("brownian_diffuse_pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("brownian_diffuse_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("diffuse"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            layout,
            pipeline,
            buffers: None,
        }
    }

    /// Crea los buffers si no existen o si la rejilla ha cambiado de tamano.
    fn ensure_buffers(&mut self, height: usize, width: usize) {
        if let Some(b) = &self.buffers
            && b.dims == (height, width)
        {
            return;
        }

        let cells = (width * height) as u64;
        let bytes = cells * size_of::<f32>() as u64;

        let storage = |label: &str| {
            self.device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let a = storage("brownian_field_a");
        let b = storage("brownian_field_b");

        let params = self.device.create_buffer(&BufferDescriptor {
            label: Some("brownian_diffuse_params"),
            size: size_of::<GpuParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("brownian_field_staging"),
            size: bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = |label: &str, src: &Buffer, dst: &Buffer| {
            self.device.create_bind_group(
                label,
                &self.layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &params,
                            offset: 0,
                            size: None,
                        }),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: src,
                            offset: 0,
                            size: None,
                        }),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: dst,
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            )
        };

        let bind_ab = bind("brownian_bind_ab", &a, &b);
        let bind_ba = bind("brownian_bind_ba", &b, &a);

        self.buffers = Some(Buffers {
            dims: (height, width),
            a,
            b,
            params,
            staging,
            bind_ab,
            bind_ba,
        });
    }
}

impl FieldBackend for GpuBackend {
    fn name(&self) -> &'static str {
        "gpu"
    }

    fn diffuse(&mut self, src: &Array2<f32>, dst: &mut Array2<f32>, p: &DiffusionParams) {
        // Un solo sub-paso: se apoya en `run` para no duplicar el montaje.
        dst.assign(src);
        let mut scratch = Array2::zeros(src.dim());
        self.run(dst, &mut scratch, p, 1);
    }

    fn run(
        &mut self,
        front: &mut Array2<f32>,
        back: &mut Array2<f32>,
        p: &DiffusionParams,
        substeps: u32,
    ) {
        if substeps == 0 {
            return;
        }
        let _ = &back; // el ping-pong ocurre en la tarjeta, no aqui

        let (height, width) = front.dim();
        self.ensure_buffers(height, width);
        let buffers = self.buffers.as_ref().expect("buffers recien creados");

        let params = GpuParams {
            kx: p.kx,
            ky: p.ky,
            decay: p.decay,
            ambient: p.ambient,
            width: width as u32,
            height: height as u32,
            pad0: 0,
            pad1: 0,
        };
        self.queue
            .write_buffer(&buffers.params, 0, bytemuck::bytes_of(&params));

        let input = front.as_slice().expect("rejilla contigua");
        self.queue
            .write_buffer(&buffers.a, 0, bytemuck::cast_slice(input));

        let groups_x = width.div_ceil(WORKGROUP as usize) as u32;
        let groups_y = height.div_ceil(WORKGROUP as usize) as u32;

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("brownian_diffuse_encoder"),
            });

        for step in 0..substeps {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("brownian_diffuse_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            // Los sub-pasos alternan origen y destino sin volver a la CPU.
            let bind = if step % 2 == 0 {
                &buffers.bind_ab
            } else {
                &buffers.bind_ba
            };
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }

        // Tras un numero impar de sub-pasos el resultado se quedo en `b`.
        let result = if substeps % 2 == 1 {
            &buffers.b
        } else {
            &buffers.a
        };
        let bytes = (width * height * size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(result, 0, &buffers.staging, 0, bytes);

        self.queue.submit([encoder.finish()]);

        // Bajada del resultado. `map_async` avisa por callback, y el poll
        // bloqueante es lo que garantiza que el dato esta antes de seguir.
        let slice = buffers.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(bevy::render::render_resource::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        if self.device.poll(PollType::Wait).is_err() {
            error!("la GPU no respondio al esperar el campo termico");
            return;
        }

        match rx.recv() {
            Ok(Ok(())) => {
                let view = slice.get_mapped_range();
                front
                    .as_slice_mut()
                    .expect("rejilla contigua")
                    .copy_from_slice(bytemuck::cast_slice(&view));
                drop(view);
                buffers.staging.unmap();
            }
            other => {
                error!("no se pudo leer el campo desde la GPU: {other:?}");
            }
        }
    }
}

use bevy::prelude::error;
