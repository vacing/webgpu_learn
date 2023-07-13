// (async () => { console.log("hello") })()
(async () => {
    if (! ('gpu' in navigator)) {
        console.error("webgpu not support")
        return
    }

    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) {
        console.error("failed to get gpu adapter")
        return
    }

    console.log(adapter.limits)
    const device = await adapter.requestDevice()

    // First matrix
    const firstMatrix = new Float32Array([
        2, 4, // row, col
        1, 2, 3, 4,
        5, 6, 7, 8
    ])
    const gpuBufFirst = device.createBuffer({
        mappedAtCreation: true,
        size: firstMatrix.byteLength,
        usage: GPUBufferUsage.STORAGE
    })
    const arrayBufFirst = gpuBufFirst.getMappedRange()
    new Float32Array(arrayBufFirst).set(firstMatrix)
    gpuBufFirst.unmap()

    // second matrix
    const secondMatrix = new Float32Array([
        4, 3, // row, col
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    ])
    const gpuBufSecond = device.createBuffer({
        mappedAtCreation: true,
        size: secondMatrix.byteLength,
        usage: GPUBufferUsage.STORAGE
    })
    const arrayBufSecond = gpuBufSecond.getMappedRange()
    new Float32Array(arrayBufSecond).set(secondMatrix)
    gpuBufSecond.unmap()

    // result matrix
    const resultBufSize = Float32Array.BYTES_PER_ELEMENT * (
        2 + firstMatrix[0]  * secondMatrix[1]
    )
    const resultMatrix = device.createBuffer({
        size: resultBufSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    })


    // shader code
    const shaderModule = device.createShaderModule({
      code: /* wgsl */ `
            struct Matrix {
                size: vec2f,
                numbers: array<f32>
            }

            @group(0) @binding(0) var<storage, read> firstM : Matrix;
            @group(0) @binding(1) var<storage, read> secondM : Matrix;
            @group(0) @binding(2) var<storage, read_write> resultM : Matrix;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) global_id: vec3u) {
                if (global_id.x >= u32(firstM.size.x) || global_id.y >= u32(secondM.size.y)) {
                    return;
                }

                resultM.size = vec2(firstM.size.x, secondM.size.y);
                let resultCell = vec2(global_id.x, global_id.y);
                let index = resultCell.y + resultCell.x * u32(resultM.size.y);
                resultM.numbers[index] = f32(index);
            }
        `,
    });

    // pipeline setup
    const computerLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'read-only-storage'
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'read-only-storage'
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'storage'
                }
            },
        ]
    })

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [ computerLayout ]
        }),
        compute: {
            module: shaderModule,
            entryPoint: "main"
        }
    })

    const bindGroup = device.createBindGroup({
      layout: computerLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: gpuBufFirst,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: gpuBufSecond,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: resultMatrix,
          },
        },
      ],
    });

const commandEncoder = device.createCommandEncoder()
const passEncoder = commandEncoder.beginComputePass()
passEncoder.setPipeline(computePipeline)
passEncoder.setBindGroup(0, bindGroup)
const wgx = Math.ceil(firstMatrix[0] / 8)
const wgy = Math.ceil(secondMatrix[1] / 8)
passEncoder.dispatchWorkgroups(wgx, wgy)
passEncoder.end()

const gpuReadBuffer = device.createBuffer({
    size: resultBufSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
})

commandEncoder.copyBufferToBuffer(
    resultMatrix, 0,
    gpuReadBuffer, 0,
    resultBufSize
)

const gpucommands = commandEncoder.finish()
device.queue.submit([gpucommands])

await gpuReadBuffer.mapAsync(GPUMapMode.READ)
const arrayBuffer = gpuReadBuffer.getMappedRange()
console.log(new Float32Array(arrayBuffer))

})();






