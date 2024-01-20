use crate::array_utils::ArrayExt;
use crate::fundamental_matrix::F;
use crate::homography::H;
use crate::projective_primitives::{Line2D, Line3D, Plane3D, Point2D, Point3D};
use crate::scalar_traits::{Descale, ScalarAdd, ScalarNeg, ScalarSub, Zero};
use crate::tensors::{CoSpace, Space, Tensor2};
use std::ops::Mul;

#[derive(Copy, Clone, Debug)]
pub struct Camera<T, SWorld: Space<4>, SImage: Space<3>>(
    Tensor2<T, SImage, CoSpace<4, SWorld>, 3, 4>,
);

impl<T: Clone + Descale + Zero, SWorld: Space<4>, SImage: Space<3>> Camera<T, SWorld, SImage>
where
    for<'a> &'a T: ScalarNeg<Output = T>,
    for<'a, 'b> &'a T: ScalarAdd<&'b T, Output = T>,
    for<'a, 'b> &'a T: ScalarSub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    #[inline]
    pub fn from_tensor(tensor: &Tensor2<T, SImage, CoSpace<4, SWorld>, 3, 4>) -> Self {
        Camera(tensor.descale())
    }

    #[inline]
    pub fn tensor(&self) -> Tensor2<T, SImage, CoSpace<4, SWorld>, 3, 4> {
        self.0.clone()
    }

    #[inline]
    pub fn center(&self) -> Point3D<T, SWorld> {
        let [plane_x, plane_y, plane_inf] =
            self.0.sub_tensors().ref_map(|t| Plane3D::from_co_tensor(t));
        plane_x.cross_with_planes(&plane_y, &plane_inf)
    }

    #[inline]
    pub fn project_point(&self, point: &Point3D<T, SWorld>) -> Point2D<T, SImage> {
        Point2D::from_contra_tensor(&self.0.contract_tensor1_10(&point.contra_tensor()))
    }

    #[inline]
    pub fn project_line(&self, line: &Line3D<T, SWorld>) -> Line2D<T, SImage> {
        let line_tensor = line.contra_tensor();
        Line2D::from_contra_tensor(
            &self
                .0
                .contract_tensor2_10(&self.0.contract_tensor2_10(&line_tensor).swap10()),
        )
    }

    #[inline]
    pub fn back_project_point(&self, point: &Point2D<T, SImage>) -> Line3D<T, SWorld> {
        Line3D::from_co_tensor(
            &point
                .co_tensor()
                .contract_tensor2_10(&self.0)
                .swap10()
                .contract_tensor2_10(&self.0),
        )
    }

    #[inline]
    pub fn back_project_line(&self, line: &Line2D<T, SImage>) -> Plane3D<T, SWorld> {
        Plane3D::from_co_tensor(&line.co_tensor().contract_tensor2_00(&self.0))
    }

    pub fn fundamental_matrix<SImageRhs: Space<3>>(
        &self,
        rhs: &Camera<T, SWorld, SImageRhs>,
    ) -> F<T, SImage, SImageRhs> {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        struct SInfinity;
        impl Space<3> for SInfinity {}

        let e1 = rhs.project_point(&self.center());
        let h0_raw = self.0.raw.ref_map(|row| {
            let sub_row: &[T; 3] = row[0..3].try_into().unwrap();
            sub_row.clone()
        });
        let h0 = Tensor2::from_raw(self.0.s0.clone(), CoSpace(SInfinity), h0_raw);
        let h1_raw = rhs.0.raw.ref_map(|row| {
            let sub_row: &[T; 3] = row[0..3].try_into().unwrap();
            sub_row.clone()
        });
        let h1 = Tensor2::from_raw(rhs.0.s0.clone(), CoSpace(SInfinity), h1_raw);
        let some_h = H::from_tensor(&h1.contract_tensor2_10(&h0.adjugate()));
        some_h.fundamental_matrix_for_e1(&e1)
    }
}

#[cfg(test)]
mod tests {
    use crate::array_utils::ArrayExt;
    use crate::camera::Camera;
    use crate::projective_primitives::{Point2D, Point3D};
    use crate::tensors::{CoSpace, Space, Tensor2};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SWorld;
    impl Space<4> for SWorld {}

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SImage0;
    impl Space<3> for SImage0 {}

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SImage1;
    impl Space<3> for SImage1 {}

    #[test]
    fn test_fundamental_matrix() {
        let point_3d = Point3D::from_normal_coords(&[531., 660., 864.], SWorld);
        let camera0 = Camera::from_tensor(&Tensor2::from_raw(
            SImage0,
            CoSpace(SWorld),
            [
                [0.679, 0.581, 0.359, 0.714],
                [0.479, 0.025, 0.603, 0.336],
                [0.053, 0.521, 0.413, 0.586],
            ],
        ));
        let camera1 = Camera::from_tensor(&Tensor2::from_raw(
            SImage1,
            CoSpace(SWorld),
            [
                [0.747, 0.787, 0.668, 0.444],
                [0.419, 0.631, 0.925, 0.007],
                [0.737, 0.075, 0.975, 0.618],
            ],
        ));
        let point0 = camera0.project_point(&point_3d);
        let point1 = camera1.project_point(&point_3d);
        let normal_point1_times_1000 = point1.normal_coords().ref_map(|x| (x * 1000.).round());
        let origin1 = Point2D::from_normal_coords(&[0., 0.], SImage1);
        let line_with_point1 = origin1.line_to_point(&point1);
        let f = camera0.fundamental_matrix(&camera1);
        let line1 = f.project_point(&point0);
        let point_to_test = line1.cross_with_line(&line_with_point1);
        let normal_point_to_test_times_1000 = point_to_test
            .normal_coords()
            .ref_map(|x| (x * 1000.).round());
        assert_eq!(normal_point_to_test_times_1000, normal_point1_times_1000);
    }
}
