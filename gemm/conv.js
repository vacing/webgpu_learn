// conv2DMM_4,4,1_linear}_true_false_false_true_3_true_true_8,8,14,4,1,4float32,float32,float32,float32x,W,bias;;false_false_false
gemm_str = /* wgsl */ `
struct vec5 {x: i32, y: i32, z: i32, w: i32, u: i32};
struct vec6 {x: i32, y: i32, z: i32, w: i32, u: i32, v: i32};

// Checks whether coordinates lie within the bounds of the shape.
fn coordsInBounds2D(coord : vec2<i32>, shape : vec2<i32>) -> bool {
  return all(coord >= vec2<i32>(0)) && all(coord < shape);
}
fn coordsInBounds3D(coord : vec3<i32>, shape : vec3<i32>) -> bool {
  return all(coord >= vec3<i32>(0)) && all(coord < shape);
}
fn coordsInBounds4D(coord : vec4<i32>, shape : vec4<i32>) -> bool {
  return all(coord >= vec4<i32>(0)) && all(coord < shape);
}

fn getIndexFromCoords1D(coord : i32, shape : i32) -> i32 {
  return coord;
}
fn getIndexFromCoords2D(coords : vec2<i32>, shape : vec2<i32>) -> i32 {
  return dot(coords, vec2<i32>(shape.y, 1));
}
fn getIndexFromCoords3D(coords : vec3<i32>, shape : vec3<i32>) -> i32 {
  return dot(coords, vec3<i32>(shape.y * shape.z, shape.z, 1));
}
fn getIndexFromCoords4D(coords : vec4<i32>, shape : vec4<i32>) -> i32 {
  return dot(coords, vec4<i32>(
      shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1));
}
fn getIndexFromCoords5D(coords : vec5, shape : vec5) -> i32 {
  let shapeStrides: vec5 = vec5(shape.y * shape.z * shape.w * shape.u, shape.z * shape.w * shape.u, shape.w * shape.u, shape.u, 1);
  return coords.x*shapeStrides.x + coords.y*shapeStrides.y + coords.z*shapeStrides.z + coords.w*shapeStrides.w + coords.u*shapeStrides.u;
}
fn getIndexFromCoords6D(coords : vec6, shape : vec6) -> i32 {
  let shapeStrides: vec6 = vec6(shape.y * shape.z * shape.w * shape.u * shape.v, shape.z * shape.w * shape.u * shape.v, shape.w * shape.u * shape.v, shape.u * shape.v, shape.v, 1);
  return coords.x*shapeStrides.x + coords.y*shapeStrides.y + coords.z*shapeStrides.z + coords.w*shapeStrides.w + coords.u*shapeStrides.u + coords.v*shapeStrides.v;
}

fn idiv(a: i32, b: i32, sign: f32) -> i32 {
  var res: i32 = a / b;
  let modulo: i32 = a % b;
  if (sign < 0. && modulo != 0) {
    res = res - 1;
  }
  return res;
}

// NaN defination in IEEE 754-1985 is :
//   - sign = either 0 or 1.
//   - biased exponent = all 1 bits.
//   - fraction = anything except all 0 bits (since all 0 bits represents infinity).
// https://en.wikipedia.org/wiki/IEEE_754-1985#Representation_of_non-numbers
fn isnan(val: f32) -> bool {
  let floatToUint: u32 = bitcast<u32>(val);
  return (floatToUint & 0x7fffffffu) > 0x7f800000u;
}
fn isnanVec4(val : vec4<f32>) -> vec4<bool> {
  let floatToUint: vec4<u32> = bitcast<vec4<u32>>(val);
  return (floatToUint & vec4<u32>(0x7fffffffu)) > vec4<u32>(0x7f800000u);
}

var<private> localId: vec3<u32>;
var<private> localIndex: u32;
var<private> globalId: vec3<u32>;
var<private> numWorkgroups: vec3<u32>;
var<private> workgroupId: vec3<u32>;

// Only used when the y/z dimension of workgroup size is 1.
fn getGlobalIndex() -> i32 {
  return i32((workgroupId.z * numWorkgroups.x * numWorkgroups.y +
        workgroupId.y * numWorkgroups.x + workgroupId.x) * 64u +
        localIndex);
}

struct Uniforms {
  NAN : f32,
  INFINITY : f32,
  xShape : vec4<i32>,
  xShapeStrides: vec3<i32>,
  wShape : vec4<i32>,
  wShapeStrides: vec3<i32>,
  biasShape : i32,
  biasShapeStrides: i32,
  outShape : vec4<i32>,
  outShapeStrides: vec3<i32>,
  filterDims : vec2<i32>,
  pads : vec2<i32>,
  strides : vec2<i32>,
  dilations : vec2<i32>,
  dimAOuter : i32,
  dimBOuter : i32,
  dimInner : i32,
};

@group(0) @binding(0) var<storage, read_write> result: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

fn isinf(val: f32) -> bool {
  return abs(val) == uniforms.INFINITY;
}


fn getCoordsFromIndex(index : i32) -> vec4<i32> {
  var index2 = index;
  let d0 = index2 / uniforms.outShapeStrides.x;
  index2 = index2 - d0 * uniforms.outShapeStrides.x;
  let d1 = index2 / uniforms.outShapeStrides.y;
  index2 = index2 - d1 * uniforms.outShapeStrides.y;
  let d2 = index2 / uniforms.outShapeStrides.z;
  let d3 = index2 - d2 * uniforms.outShapeStrides.z;
  return vec4<i32>(d0,d1,d2,d3);
}

fn getOutputCoords() -> vec4<i32> {
  let d3 = i32(globalId[0]);
  var index1 = i32(globalId[1]);
  let d1 = index1 / uniforms.outShape.z;
  let d2 = index1 - d1 * uniforms.outShape.z;
  let d0 = i32(globalId[2]);
  return vec4<i32>(d0,d1,d2,d3);
}

fn getOutputIndexFromCoords(coords : vec4<i32>) -> i32 {
  return dot(coords, vec4<i32>(
    uniforms.outShapeStrides.x, uniforms.outShapeStrides.y, uniforms.outShapeStrides.z, 1));
}

fn setOutputAtIndex(flatIndex : i32, value : vec4<f32>) {
  result[flatIndex] = vec4<f32>(value);
}

fn setOutputAtIndexI32(flatIndex : i32, value : vec4<i32>) {
  result[flatIndex] = vec4<f32>(value);
}

fn setOutputAtCoords(d0 : i32, d1 : i32, d2 : i32, d3 : i32, value : vec4<f32>) {
  let flatIndex = getOutputIndexFromCoords(vec4<i32>(d0, d1, d2, d3));
  setOutputAtIndex(flatIndex / 4, value);
}
fn setOutputAtCoordsI32(d0 : i32, d1 : i32, d2 : i32, d3 : i32, value : vec4<i32>) {
  let flatIndex = getOutputIndexFromCoords(vec4<i32>(d0, d1, d2, d3));
  setOutputAtIndexI32(flatIndex / 4, value);
}

fn getXCoordsFromIndex(index : i32) -> vec4<i32> {
  var index2 = index;
  let d0 = index2 / uniforms.xShapeStrides.x;
  index2 = index2 - d0 * uniforms.xShapeStrides.x;
  let d1 = index2 / uniforms.xShapeStrides.y;
  index2 = index2 - d1 * uniforms.xShapeStrides.y;
  let d2 = index2 / uniforms.xShapeStrides.z;
  let d3 = index2 - d2 * uniforms.xShapeStrides.z;
  return vec4<i32>(d0,d1,d2,d3);
}

fn getWCoordsFromIndex(index : i32) -> vec4<i32> {
  var index2 = index;
  let d0 = index2 / uniforms.wShapeStrides.x;
  index2 = index2 - d0 * uniforms.wShapeStrides.x;
  let d1 = index2 / uniforms.wShapeStrides.y;
  index2 = index2 - d1 * uniforms.wShapeStrides.y;
  let d2 = index2 / uniforms.wShapeStrides.z;
  let d3 = index2 - d2 * uniforms.wShapeStrides.z;
  return vec4<i32>(d0,d1,d2,d3);
}

fn getBiasCoordsFromIndex(index : i32) -> i32 { return index; }

fn getX(d0 : i32, d1 : i32, d2 : i32, d3 : i32) -> f32 {
  return f32(x[getIndexFromCoords4D(vec4<i32>(d0,d1,d2,d3),
    uniforms.xShape)]);
}

fn getXByOutputIndex(globalIndex : i32) -> f32 {
  var coords = getCoordsFromIndex(globalIndex);
  return f32(x[getIndexFromCoords4D(vec4<i32>(coords.x, coords.y, coords.z, coords.w), uniforms.xShape)]);
}

fn getXByOutputCoords(coordsIn : vec4<i32>) -> f32 {
  var coords = coordsIn;
  return f32(x[getIndexFromCoords4D(vec4<i32>(coords.x, coords.y, coords.z, coords.w), uniforms.xShape)]);
}

fn getW(d0 : i32, d1 : i32, d2 : i32, d3 : i32) -> vec4<f32> {
  return vec4<f32>(W[getIndexFromCoords4D(vec4<i32>(d0,d1,d2,d3),
    uniforms.wShape) / 4]);
}

fn getWByOutputIndex(globalIndex : i32) -> vec4<f32> {
  var coords = getCoordsFromIndex(globalIndex);
  return vec4<f32>(W[getIndexFromCoords4D(vec4<i32>(coords.x, coords.y, coords.z, coords.w), uniforms.wShape) / 4]);
}

fn getWByOutputCoords(coordsIn : vec4<i32>) -> vec4<f32> {
  var coords = coordsIn;
  return vec4<f32>(W[getIndexFromCoords4D(vec4<i32>(coords.x, coords.y, coords.z, coords.w), uniforms.wShape) / 4]);
}

fn getBias(d0 : i32) -> vec4<f32> {
  return vec4<f32>(bias[getIndexFromCoords1D(i32(d0),
    uniforms.biasShape) / 4]);
}

fn getBiasByOutputIndex(globalIndex : i32) -> vec4<f32> {
  var coords = getCoordsFromIndex(globalIndex);
  return vec4<f32>(bias[getIndexFromCoords1D(i32(coords.w), uniforms.biasShape) / 4]);
}

fn getBiasByOutputCoords(coordsIn : vec4<i32>) -> vec4<f32> {
  var coords = coordsIn;
  return vec4<f32>(bias[getIndexFromCoords1D(i32(coords.w), uniforms.biasShape) / 4]);
}

fn activation(a : vec4<f32>, coords : vec4<i32>) -> vec4<f32> {
  return a;
}

fn mm_readA(batch: i32, row : i32, col : i32) -> vec3<f32> {
  if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
    let inChannels = uniforms.wShape[2];
    let outWidth = uniforms.outShape[2];
    let outRow = row / outWidth;
    let outCol = row % outWidth;

    let WRow = col / (uniforms.filterDims[1] * inChannels);
    let WCol = col / inChannels % uniforms.filterDims[1];
    let xRow = outRow * uniforms.strides[0] + uniforms.dilations[0] * WRow - uniforms.pads[0];
    let xCol = outCol * uniforms.strides[1] + uniforms.dilations[1] * WCol - uniforms.pads[1];
    let xCh = col % inChannels;
    var resData = vec3<f32>(0.0);
    // The bounds checking is always needed since we use it to pad zero for
    // the 'same' padding type.
    if (xRow >= 0 && xRow < uniforms.xShape[1] && xCol >= 0 && xCol < uniforms.xShape[2]) {
      let coord = vec4<i32>(batch, xRow, xCol, xCh);
      let xIndex = getIndexFromCoords4D(coord, uniforms.xShape);
      resData = vec3<f32>(x[xIndex], x[xIndex + 1], x[xIndex + 2]);
    }
    return resData;
  }

  return vec3<f32>(0.0);
}

fn mm_readB(batch: i32, row : i32, col : i32) -> vec4<f32> {
  return W[(row * uniforms.wShape[3] + col) / 4];
}

fn mm_write(batch: i32, row : i32, col : i32, valueIn : vec4<f32>) {
  if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
    var value = valueIn;
    let outWidth = uniforms.outShape[2];
    let coords = vec4<i32>( batch, row / outWidth, row % outWidth, col);
    value = value + getBiasByOutputCoords(coords);
    value = activation(value, coords);
    setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
  }
}

var<workgroup> mm_Asub : array<array<vec3<f32>, 8>, 32>;
var<workgroup> mm_Bsub : array<array<vec4<f32>, 8>, 24>;
fn main() {
  let localRow = i32(localId.y);
  let tileRow = localRow * 4;
  let tileCol = i32(localId.x);

  let globalRow = i32(globalId.y) * 4;
  let globalCol = i32(globalId.x) * 4;
  let batch = i32(globalId.z);
  let batchA = batch;
  let batchB = batch;
  let globalRowStart = i32(workgroupId.y) * 32;

  let numTiles = (uniforms.dimInner - 1) / 24 + 1;
  var kStart = 0;

  var acc: array<vec4<f32>, 4>;

  // Loop over shared dimension.
  let tileRowB = localRow * 3;
  for (var t = 0; t < numTiles; t++) {
    // Load one tile of A into local memory.
    for (var innerRow = 0; innerRow < 4; innerRow++) {
      let inputRow = tileRow + innerRow;
      let inputCol = tileCol;
      mm_Asub[inputRow][inputCol] = mm_readA(batchA,
      globalRow + innerRow,
      kStart + inputCol * 3);
    }

    // Load one tile of B into local memory.
    for (var innerRow = 0; innerRow < 3; innerRow++) {
        let inputRow = tileRowB + innerRow;
        let inputCol = tileCol;
        mm_Bsub[inputRow][inputCol] = mm_readB(batchB, kStart + inputRow, globalCol);
    }
    kStart = kStart + 24;
    workgroupBarrier();

    // Compute acc values for a single thread.
    for (var k = 0; k < 8; k++) {
      let BCached0 = mm_Bsub[k * 3 + 0][tileCol];let BCached1 = mm_Bsub[k * 3 + 1][tileCol];let BCached2 = mm_Bsub[k * 3 + 2][tileCol];
      for (var i = 0; i < 4; i++) {
        let ACached = mm_Asub[tileRow + i][k];
        acc[i] = fma(BCached0, vec4<f32>(ACached[0]), acc[i]);acc[i] = fma(BCached1, vec4<f32>(ACached[1]), acc[i]);acc[i] = fma(BCached2, vec4<f32>(ACached[2]), acc[i]);
      }
    }
    workgroupBarrier();
  }

  for (var innerRow = 0; innerRow < 4; innerRow++) {
    mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
  }
}

@compute @workgroup_size(8, 8, 1)
fn _start(@builtin(local_invocation_id) LocalId : vec3<u32>,
          @builtin(global_invocation_id) GlobalId : vec3<u32>,
          @builtin(local_invocation_index) LocalIndex: u32,
          @builtin(workgroup_id) WorkgroupId : vec3<u32>,
          @builtin(num_workgroups) NumWorkgroups : vec3<u32>) {
  localId = LocalId;
  localIndex = LocalIndex;
  globalId = GlobalId;
  numWorkgroups = NumWorkgroups;
  workgroupId = WorkgroupId;
  main();;
}
`

