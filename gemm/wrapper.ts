import gemm_str from "./conv.js"

export const compileProgram =
    (device: GPUDevice, program: WebGPUProgram, inputsData: InputInfo[],
     output: TensorInfo, parallelCompilation: boolean): GPUComputePipeline|
    Promise<GPUComputePipeline> => {
      const outputData = {dtype: output.dtype, shape: output.shape};
      const source = gemm_str
      const module = device.createShaderModule({
        code: source,
        label: program.constructor.name
    });

    return device.createComputePipeline({
        compute: {
            module,
            entryPoint: '_start'
        },
        label: program.constructor.name,
        layout: 'auto'
    });
};

recordAndSubmit(
    program: webgpu_program.WebGPUProgram, output: TensorInfo,
    inputs: TensorInfo[], programDefinedUniform?: ProgramUniform) {
    if (program.pipeline instanceof Promise<GPUComputePipeline>) {
        throw new Error(
            'Please call checkCompileCompletionAsync to ensure parallel compilation is done!');
    }
    // There are six kinds of uniforms: NAN, INFINITY, shapes, shape strides,
    // program size, program defined uniforms.
    let programUniform: ProgramUniform = [];
    let bufferShapes: number[][] = [];
    const uniformsType = 'int32';
    if (program.pixelsOpType == null) {
        programUniform.push(
            {type: 'float32', data: [NaN]}, {type: 'float32', data: [Infinity]});
        bufferShapes = inputs.concat(output).map(d => d.shape);
        const uniformsType = 'int32';
        bufferShapes.map(d => {
        programUniform.push({type: uniformsType, data: d});
        const strides = util.computeStrides(d);
        programUniform.push({type: uniformsType, data: strides});
        });
    } else {
        const strides = util.computeStrides(output.shape);
        programUniform.push({type: uniformsType, data: strides});
    }
    if (program.size) {
        const size = util.sizeFromShape(program.outputShape);
        programUniform.push({
        type: uniformsType,
        data: [program.outputComponent ? size / program.outputComponent : size]
        });
    }

    if (programDefinedUniform) {
        programUniform = [...programUniform, ...programDefinedUniform];
    }
    const bindings = [
        this.tensorToBinding(output), ...inputs.map(t => this.tensorToBinding(t)),
        this.makeUniforms(programUniform)
    ];

    inputs.forEach(input => {
        this.commandQueueOwnedIds.add(input.dataId);
    });
    this.commandQueueOwnedIds.add(output.dataId);

    const bindGroup = this.device.createBindGroup({
        layout: program.pipeline.getBindGroupLayout(0),
        entries: bindings.map((b, i) => ({binding: i, resource: b})),
    });

    const shouldTimeProgram = this.activeTimers != null;
    this.ensureCommandEncoderReady();

    const computePassDescriptor: GPUComputePassDescriptor = {};
    if (shouldTimeProgram && this.supportTimestampQuery) {
        this.endComputePassEncoder();
        if (this.querySet == null) {
        this.querySet = this.device.createQuerySet({
            type: 'timestamp',
            count: this.querySetCount,
        });
        }
        computePassDescriptor.timestampWrites = [
        {
            querySet: this.querySet,
            queryIndex: 0,
            location: 'beginning',
        },
        {
            querySet: this.querySet,
            queryIndex: 1,
            location: 'end',
        }
        ];
        this.computePassEncoder =
            this.commandEncoder.beginComputePass(computePassDescriptor);
    } else if (!this.computePassEncoder) {
        this.computePassEncoder =
            this.commandEncoder.beginComputePass(computePassDescriptor);
    }

    this.computePassEncoder.setPipeline(program.pipeline);
    this.computePassEncoder.setBindGroup(0, bindGroup);
    this.computePassEncoder.dispatchWorkgroups(
        program.dispatch[0], program.dispatch[1], program.dispatch[2]);
    this.dispatchCountInPass++;

    if (shouldTimeProgram ||
        env().get('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE') as
            number <= this.dispatchCountInPass ||
        program.pixelsOpType === webgpu_program.PixelsOpType.DRAW) {
        this.endComputePassEncoder();
        if (shouldTimeProgram) {
        this.activeTimers.push(
            {name: program.constructor.name, query: this.getQueryTime()});
        } else {
        this.submitQueue();
        }
    }
}