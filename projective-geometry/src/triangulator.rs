use crate::camera::Camera;
use crate::projective_primitives::{Point2D, Point3D};
use crate::scalar_traits::{descale_array, Descale, ScalarAdd, ScalarNeg, ScalarSub, Zero};
use crate::tensors::{Space, Tensor1};
use std::ops::Mul;

pub trait Triangulator<T> {
    fn triangulate<SImage0: Space<3>, SImage1: Space<3>, SWorld: Space<4>>(
        &self,
        camera0: &Camera<T, SWorld, SImage0>,
        camera1: &Camera<T, SWorld, SImage1>,
        point0: &Point2D<T, SImage0>,
        point1: &Point2D<T, SImage1>,
    ) -> Point3D<T, SWorld>;
}

pub struct TriangulatorImpl<T> {
    random_2d_homogeneous_coords: [T; 3],
}

impl<T: Clone + Descale> TriangulatorImpl<T> {
    #[inline]
    pub fn new(random_2d_homogeneous_coords: &[T; 3]) -> Self {
        Self {
            random_2d_homogeneous_coords: descale_array(random_2d_homogeneous_coords),
        }
    }
}

impl<T: Clone + Descale + Zero> Triangulator<T> for TriangulatorImpl<T>
where
    for<'a> &'a T: ScalarNeg<Output = T>,
    for<'a, 'b> &'a T: ScalarAdd<&'b T, Output = T>,
    for<'a, 'b> &'a T: ScalarSub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    fn triangulate<SImage0: Space<3>, SImage1: Space<3>, SWorld: Space<4>>(
        &self,
        camera0: &Camera<T, SWorld, SImage0>,
        camera1: &Camera<T, SWorld, SImage1>,
        point0: &Point2D<T, SImage0>,
        point1: &Point2D<T, SImage1>,
    ) -> Point3D<T, SWorld> {
        let ray0 = camera0.back_project_point(point0);
        let random_point1 = Point2D::from_contra_tensor(&Tensor1::from_raw(
            point1.get_s(),
            self.random_2d_homogeneous_coords.clone(),
        ));
        let line1 = point1.line_to_point(&random_point1);
        let plane1 = camera1.back_project_line(&line1);
        ray0.cross_with_plane(&plane1)
    }
}

#[cfg(test)]
mod tests {
    use crate::camera::Camera;
    use crate::projective_primitives::Point3D;
    use crate::tensors::{CoSpace, Space, Tensor2};
    use crate::triangulator::{Triangulator, TriangulatorImpl};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SImage0;
    impl Space<3> for SImage0 {}

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SImage1;
    impl Space<3> for SImage1 {}

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SWorld;
    impl Space<4> for SWorld {}

    #[test]
    fn test_triangulate() {
        let point3d_normal_coords = [748., 538., 307.];
        let point3d = Point3D::from_normal_coords(&point3d_normal_coords, SWorld);
        let camera0 = Camera::from_tensor(&Tensor2::from_raw(
            SImage0,
            CoSpace(SWorld),
            [
                [0.458, 0.115, 0.053, 0.891],
                [0.156, 0.589, 0.533, 0.843],
                [0.807, 0.386, 0.098, 0.835],
            ],
        ));
        let camera1 = Camera::from_tensor(&Tensor2::from_raw(
            SImage1,
            CoSpace(SWorld),
            [
                [0.667, 0.063, 0.213, 0.727],
                [0.548, 0.104, 0.765, 0.163],
                [0.427, 0.978, 0.517, 0.614],
            ],
        ));
        let point0 = camera0.project_point(&point3d);
        let point1 = camera1.project_point(&point3d);
        let random_2d_homogeneous_coords = [0.7640727990553938, 0.5580193420125891, 1.];
        let triangulator = TriangulatorImpl::new(&random_2d_homogeneous_coords);
        assert_eq!(
            triangulator
                .triangulate(&camera0, &camera1, &point0, &point1)
                .normal_coords()
                .map(|x| x.round()),
            point3d_normal_coords
        );
    }
}
