#![forbid(unsafe_code)]
#![feature(array_methods)]
#![feature(impl_trait_in_assoc_type)]
#![feature(slice_flatten)]
#![feature(slice_as_chunks)]
#![feature(generic_const_exprs)]

pub mod adjugate;
pub mod array_utils;
pub mod camera;
pub mod eigen_vectors;
pub mod homogeneous_equations;
pub mod homography;
pub mod homography_from_points;
pub mod levi_civita;
pub mod projective_primitives;
pub mod scalar_traits;
pub mod tensors;
pub mod triangulator;
