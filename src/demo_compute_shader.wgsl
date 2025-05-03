@group(0) @binding(0) var<storage,read> inputBuffer: array<f32,640*480>;
@group(0) @binding(1) var<storage,read_write> outputBuffer: array<u32,640*480>;

struct Sphere {
    center: vec3f,
    radius: f32,
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

struct Material {
    color: vec3f,
}

struct HitRecord {
    t: f32,
    p: vec3f,
    normal: vec3f,
    material: Material,
}

// The function to evaluate for each element of the processed buffer
fn f(x: f32) -> f32 {
    return 20.0 * x + 1.0;
}

@compute @workgroup_size(32)
fn computeStuff(@builtin(global_invocation_id) id: vec3<u32>) {
    // Apply the function f to the buffer element at index id.x:
    outputBuffer[id.x] = u32(f(inputBuffer[id.x]));
}