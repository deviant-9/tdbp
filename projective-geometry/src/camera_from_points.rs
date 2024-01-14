use crate::array_utils::ArrayExt;
use crate::camera::Camera;
use crate::homogeneous_equations::{get_ax_collinear_y_equations_for_a, ExactHomogeneousSolver};
use crate::projective_primitives::{Point2D, Point3D};
use crate::scalar_traits::{Descale, ScalarAdd, ScalarNeg, ScalarSub, Zero};
use crate::tensors::{CoSpace, Space, Tensor2};
use std::ops::Mul;

pub trait CameraFromMinimalPointsSolver<T> {
    fn solve<SWorld: Space<4>, SImage: Space<3>>(
        &self,
        points: &[(Point3D<T, SWorld>, Point2D<T, SImage>); 6],
    ) -> Camera<T, SWorld, SImage>;
}

pub struct CameraFromMinimalPointsSolverImpl<Solver> {
    equations_solver: Solver,
}

impl<Solver> CameraFromMinimalPointsSolverImpl<Solver> {
    #[inline]
    pub fn new<T>(equations_solver: Solver) -> Self
    where
        Solver: ExactHomogeneousSolver<T, { 3 * 4 }, { 3 * 4 - 1 }>,
    {
        Self { equations_solver }
    }
}

impl<T: Clone + Descale + Zero, Solver: ExactHomogeneousSolver<T, { 3 * 4 }, { 3 * 4 - 1 }>>
    CameraFromMinimalPointsSolver<T> for CameraFromMinimalPointsSolverImpl<Solver>
where
    for<'a> &'a T: ScalarNeg<Output = T>,
    for<'a, 'b> &'a T: ScalarAdd<&'b T, Output = T>,
    for<'a, 'b> &'a T: ScalarSub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    fn solve<SWorld: Space<4>, SImage: Space<3>>(
        &self,
        points: &[(Point3D<T, SWorld>, Point2D<T, SImage>); 6],
    ) -> Camera<T, SWorld, SImage> {
        let full_points_a: [[[T; 3 * 4]; 3 - 1]; 6] =
            points.ref_map(|(world_point, image_point)| {
                get_ax_collinear_y_equations_for_a(
                    &world_point.contra_tensor().raw,
                    &image_point.contra_tensor().raw,
                )
            });
        let full_a_slice_ref: &[[T; 3 * 4]] = full_points_a.flatten();
        let a_ref: &[[T; 3 * 4]; 3 * 4 - 1] = full_a_slice_ref[..3 * 4 - 1].try_into().unwrap();
        let c_flat = self.equations_solver.solve(a_ref);
        let (c_slice, _) = c_flat.as_chunks::<4>();
        let c_ref: &[[T; 4]; 3] = c_slice.try_into().unwrap();
        let c: [[T; 4]; 3] = c_ref.clone();
        let s_world = points[0].0.get_s();
        let s_image = points[0].1.get_s();
        Camera::from_tensor(&Tensor2::from_raw(s_image, CoSpace(s_world), c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homogeneous_equations::NoSqrtExactHomogeneousSolverImpl;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SImage;
    impl Space<3> for SImage {}

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SWorld;
    impl Space<4> for SWorld {}

    #[test]
    fn test_camera_from_minimal_points() {
        let world_normal_points: [[f64; 3]; 6] = [
            [958., 575., 342.],
            [301., 199., 856.],
            [449., 758., 426.],
            [147., 246., 541.],
            [540., 535., 938.],
            [640., 581., 274.],
        ];
        let world_points: [Point3D<f64, SWorld>; 6] =
            world_normal_points.ref_map(|p| Point3D::from_normal_coords(p, SWorld));
        let camera: Camera<f64, SWorld, SImage> = Camera::from_tensor(&Tensor2::from_raw(
            SImage,
            CoSpace(SWorld),
            [
                [0.586, 0.597, 0.524, 0.791],
                [0.648, 0.901, 0.182, 0.279],
                [0.936, 0.863, 0.669, 0.323],
            ],
        ));
        let image_points = world_points.ref_map(|p| camera.project_point(p));
        let point_pairs =
            world_points.ref_map_with(&image_points, |world_p, image_p| (*world_p, *image_p));

        let random_vector: [f64; 12] = [
            0.9838754523652482,
            0.70788485086306,
            0.5710167952184118,
            0.32444202581192705,
            0.25274759116253953,
            0.4071973855434946,
            0.6249834315362917,
            0.18915254445155127,
            0.2525818280867961,
            0.5811966727935087,
            0.7446823820031155,
            0.7597920711317897,
        ];
        let equations_solver = NoSqrtExactHomogeneousSolverImpl::new(&random_vector);
        let solver = CameraFromMinimalPointsSolverImpl::new(equations_solver);
        let solved_camera = solver.solve(&point_pairs);
        let solved_image_points = world_points.ref_map(|p| solved_camera.project_point(p));

        let image_normal_points_times_1000 =
            image_points.ref_map(|p| p.normal_coords().ref_map(|x| (x * 1000.).round()));
        let solved_image_normal_points_times_1000 =
            solved_image_points.ref_map(|p| p.normal_coords().ref_map(|x| (x * 1000.).round()));
        assert_eq!(
            solved_image_normal_points_times_1000,
            image_normal_points_times_1000
        );
    }
}
