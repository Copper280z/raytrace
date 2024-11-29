const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;

pub const Compute = struct {
    gctx: *zgpu.GraphicsContext,
    bindGroupLayout: zgpu.BindGroupLayoutHandle,
    computePipeline: zgpu.ComputePipelineHandle,
    bindGroup: zgpu.BindGroupHandle,
    shader_source: [*:0]const u8,
    bufferSize: usize, // bytes
    inputBuffer: zgpu.BufferHandle,
    outputBuffer: zgpu.BufferHandle, // result of the computation, but can't be directly copied to host
    mapBuffer: zgpu.BufferHandle, // use this buffer to copy to host

    compute_done: bool = false,
    mapped: bool = false,

    pub fn create(gctx: *zgpu.GraphicsContext, allocator: std.mem.Allocator) !*Compute {
        const shader = try allocator.create(Compute);
        shader.*.shader_source = @embedFile("demo_compute_shader.wgsl");

        shader.*.gctx = gctx;
        shader.*.initBindGroupLayout();
        shader.*.initComputePipeline();
        shader.*.initBuffers();
        shader.*.initBindGroup();
        return shader;
    }

    fn initBindGroupLayout(self: *@This()) void {
        // Create bind group layout
        // Bind group layout holds the description of the IO buffers for the shader
        const bindings: [2]wgpu.BindGroupLayoutEntry = .{
            .{
                .binding = 0,
                .buffer = .{ .binding_type = .read_only_storage },
                .visibility = .{ .compute = true },
            },
            .{
                .binding = 1,
                .buffer = .{ .binding_type = .storage },
                .visibility = .{ .compute = true },
            },
        };

        self.bindGroupLayout = self.gctx.createBindGroupLayout(&bindings);

        // const bindGroupLayoutDesc = wgpu.BindGroupLayoutDescriptor{
        //     .entry_count = bindings.len,
        //     .entries = bindings,
        // };

        // self.bindGroupLayout = self.gctx.device.createBindGroupLayout(bindGroupLayoutDesc);
    }

    fn initComputePipeline(self: *@This()) void {
        // Load compute shader
        const computeShaderModule = zgpu.createWgslShaderModule(self.gctx.device, self.shader_source, "compute!");
        defer computeShaderModule.release();

        // const pipelineLayoutDesc = wgpu.PipelineLayoutDescriptor{ .bind_group_layout_count = 1, .bind_group_layouts = self.bindGroupLayout };

        // const pipelineLayout = self.gctx.device.createPipelineLayout(pipelineLayoutDesc);
        // // // Create compute pipeline

        // self.computePipeline = self.gctx.device.createComputePipeline(computePipelineDesc);

        const bindGroupLayouts: [1]zgpu.BindGroupLayoutHandle = .{self.bindGroupLayout};
        const pipelineLayout = self.gctx.createPipelineLayout(&bindGroupLayouts);
        const computePipelineDesc = wgpu.ComputePipelineDescriptor{ .compute = .{ .entry_point = "computeStuff", .module = computeShaderModule } };

        self.computePipeline = self.gctx.createComputePipeline(pipelineLayout, computePipelineDesc);
    }

    fn initBuffers(self: *@This()) void {
        // We save this size in an attribute, it will be useful later on
        self.bufferSize = 64 * @sizeOf(f32);

        // // Create input buffers
        var bufferDesc = wgpu.BufferDescriptor{
            .mapped_at_creation = false,
            .size = self.bufferSize,
            .usage = .{
                .storage = true,
                .copy_dst = true,
            },
        };

        self.inputBuffer = self.gctx.createBuffer(bufferDesc);

        // Create output buffer: the only difference is the usage
        bufferDesc.usage = .{ .storage = true, .copy_src = true };
        self.outputBuffer = self.gctx.createBuffer(bufferDesc);

        bufferDesc.usage = .{
            .copy_dst = true,
            .map_read = true,
        };
        bufferDesc.mapped_at_creation = true;
        self.mapBuffer = self.gctx.createBuffer(bufferDesc);
        const mapBuffer: wgpu.Buffer = self.gctx.lookupResource(self.mapBuffer).?;
        // std.debug.print("Buffer map state: {}\n", .{mapBuffer.getMapState()});

        const result = mapBuffer.getConstMappedRange(f32, 0, self.bufferSize);
        if (result) |res| {
            std.debug.print("input: {} became {}\n", .{ 0.1, res[0] });
        } else {
            std.debug.print("GetMappedRange failed and returned null at the init\n", .{});
        }

        mapBuffer.unmap();
    }

    fn initBindGroup(self: *@This()) void {
        const binds: [2]zgpu.BindGroupEntryInfo = .{
            .{
                .binding = 0,
                .buffer_handle = self.inputBuffer,
                .offset = 0,
                .size = self.bufferSize,
            },
            .{
                .binding = 1,
                .buffer_handle = self.outputBuffer,
                .offset = 0,
                .size = self.bufferSize,
            },
        };

        // self.bindGroup = wgpu.BindGroup{
        //     .layout = self.bindGroupLayout,
        // };
        self.bindGroup = self.gctx.createBindGroup(self.bindGroupLayout, &binds);
    }

    pub fn onCompute(self: *@This()) void {
        self.compute_done = false;
        // Initialize a command encoder
        const queue = self.gctx.device.getQueue();
        defer queue.release();

        var input: [64]f32 = undefined;

        for (0..input.len) |i| {
            input[i] = 0.1 * @as(f32, @floatFromInt(i));
        }

        queue.writeBuffer(
            self.gctx.lookupResource(self.inputBuffer).?,
            0,
            f32,
            &input,
        );

        const encoderDesc = wgpu.CommandEncoderDescriptor{};
        const encoder = self.gctx.device.createCommandEncoder(encoderDesc);
        defer encoder.release();

        // Create and use compute pass here!
        const computePassDesc = wgpu.ComputePassDescriptor{
            .timestamp_write_count = 0,
            .timestamp_writes = null,
        };

        const computePass = encoder.beginComputePass(computePassDesc);
        defer {
            // computePass.end();
            computePass.release();
        }

        const computePipeline = self.gctx.lookupResource(self.computePipeline).?;
        const bind_group = self.gctx.lookupResource(self.bindGroup).?;
        // now use the compute pass?
        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, bind_group, null);
        const invocationCount = @as(u32, @intCast(self.bufferSize)) / @sizeOf(f32);
        const workgroupSize = 32;
        // This ceils invocationCount / workgroupSize
        const workgroupCount = (invocationCount + workgroupSize - 1) / workgroupSize;
        computePass.dispatchWorkgroups(workgroupCount, 1, 1); // this calls the compute shader

        computePass.end();

        encoder.copyBufferToBuffer(
            self.gctx.lookupResource(self.outputBuffer).?,
            0,
            self.gctx.lookupResource(self.mapBuffer).?,
            0,
            self.bufferSize,
        );

        // Encode and submit the GPU commands
        const command = encoder.finish(null);
        defer command.release();
        const commands: [1]wgpu.CommandBuffer = .{command};
        queue.submit(&commands);
        queue.onSubmittedWorkDone(0, compute_done, @ptrCast(self));

        while (!self.compute_done) {
            self.gctx.device.tick();
        }

        const mapBuffer: wgpu.Buffer = self.gctx.lookupResource(self.mapBuffer).?;
        mapBuffer.mapAsync(.{ .read = true }, 0, 0, print_callback, @ptrCast(self));
        while (!self.mapped) {
            self.gctx.device.tick();
        }

        std.debug.print("Buffer map state: {s}\n", .{@tagName(mapBuffer.getMapState())});
        std.debug.print("Buffer size: {}\n", .{mapBuffer.getSize()});
        std.debug.print("Buffer usage: {}\n", .{mapBuffer.getUsage()});
        const result = mapBuffer.getMappedRange(f32, 0, self.bufferSize);
        if (result) |res| {
            std.debug.print("input: {} became {}\n", .{ 0.1, res[0] });
        } else {
            std.debug.print("GetMappedRange failed and returned null\n", .{});
        }
        mapBuffer.unmap();
        self.mapped = false;
        // std.debug.print("done with onCompute\n", .{});
    }
};

fn compute_done(status: wgpu.QueueWorkDoneStatus, userdata: ?*anyopaque) callconv(.C) void {
    const self = @as(*Compute, @ptrCast(@alignCast(userdata)));
    self.compute_done = true;

    if (status != .success) {
        std.debug.print("Compute Failed to complete GPU work (status: {s}).", .{@tagName(status)});
    }
}

fn print_callback(status: wgpu.BufferMapAsyncStatus, userdata: ?*anyopaque) callconv(.C) void {
    std.debug.print("Compute shader done! - status: {}\n", .{status});
    const self = @as(*Compute, @ptrCast(@alignCast(userdata)));
    // const mapBuffer: wgpu.Buffer = self.gctx.lookupResource(self.mapBuffer).?;

    if (status == .success) {
        // const result = mapBuffer.getMappedRange(f32, 0, 64).?;
        // std.debug.print("input: {} became {}\n", .{ 0.1, result[0] });
        // mapBuffer.unmap();
    } else {
        std.debug.print("something went wrong with the compute shader\n", .{});
    }
    self.mapped = true;
}
