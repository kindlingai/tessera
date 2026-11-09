//! Bytes per quantized row and matrix, for each format.

/// Weights per affine int4 or int6 group.
pub const GROUP_SIZE: usize = 32;
/// Weights per NVFP4 block scale.
pub const NVFP4_GROUP_SIZE: usize = 16;
/// An f32 global scale prefixes NVFP4 payloads.
pub const NVFP4_HEADER_BYTES: usize = 4;

/// Bytes for one Q4 block covering `GROUP_SIZE` weights along K.
pub const Q4_BLOCK_BYTES: usize = 4 + GROUP_SIZE / 2; // fp16 scale + fp16 min + 16 nibbles

pub fn q4_row_bytes(in_dim: usize) -> usize {
    let groups = in_dim.div_ceil(GROUP_SIZE);
    groups * Q4_BLOCK_BYTES
}

pub fn q4_matrix_bytes(out_dim: usize, in_dim: usize) -> usize {
    out_dim * q4_row_bytes(in_dim)
}

/// Bytes for one Q6 block covering `GROUP_SIZE` weights along K.
pub const Q6_BLOCK_BYTES: usize = 4 + GROUP_SIZE * 6 / 8; // bf16 scale + bf16 min + 24B codes

pub fn q6_row_bytes(in_dim: usize) -> usize {
    let groups = in_dim.div_ceil(GROUP_SIZE);
    groups * Q6_BLOCK_BYTES
}

pub fn q6_matrix_bytes(out_dim: usize, in_dim: usize) -> usize {
    out_dim * q6_row_bytes(in_dim)
}

/// E2M1 code bytes per row, low nibble first.
pub fn nvfp4_data_row_bytes(in_dim: usize) -> usize {
    in_dim.div_ceil(2)
}

/// E4M3 block scale bytes per row.
pub fn nvfp4_scales_row_bytes(in_dim: usize) -> usize {
    in_dim.div_ceil(NVFP4_GROUP_SIZE)
}

pub fn nvfp4_row_bytes(in_dim: usize) -> usize {
    nvfp4_data_row_bytes(in_dim) + nvfp4_scales_row_bytes(in_dim)
}

pub fn nvfp4_matrix_bytes(out_dim: usize, in_dim: usize) -> usize {
    NVFP4_HEADER_BYTES + out_dim * nvfp4_row_bytes(in_dim)
}

pub fn q8_row_bytes(in_dim: usize) -> usize {
    2 + in_dim // fp16 scale + int8 weights
}

pub fn q8_matrix_bytes(out_dim: usize, in_dim: usize) -> usize {
    out_dim * q8_row_bytes(in_dim)
}
