use crate::array_utils::ArrayExt;
use crate::fundamental_matrix::F;
use crate::homography::H;
use crate::projective_primitives::{Line2D, Line3D, Plane3D, Point2D, Point3D};
use crate::scalar_traits::{Descale, Hypot, ScalarAdd, ScalarNeg, ScalarSub, Zero};
use crate::tensors::{CoSpace, Space, Tensor2};
use std::ops::{Div, Mul};

#[derive(Copy, Clone, Debug)]
pub struct Camera<T, SWorld: Space<4>, SImage: Space<3>>(
    Tensor2<T, SImage, CoSpace<4, SWorld>, 3, 4>,
);

#[derive(Copy, Clone, Debug)]
pub struct CameraInternalParams<T> {
    pub fx: T,
    pub fy: T,
    pub x0: T,
    pub y0: T,
    pub s: T,
}

#[derive(Copy, Clone, Debug)]
pub struct CameraExternalParams<T, SWorld: Space<4>> {
    pub center: Point3D<T, SWorld>,
    pub ax_sin: T,
    pub ax_cos: T,
    pub ay_sin: T,
    pub ay_cos: T,
    pub az_sin: T,
    pub az_cos: T,
}

#[derive(Copy, Clone, Debug)]
pub struct CameraParams<T, SWorld: Space<4>> {
    pub internal: CameraInternalParams<T>,
    pub external: CameraExternalParams<T, SWorld>,
}

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

    pub fn get_params(&self) -> CameraParams<T, SWorld>
    where
        T: Hypot<T, Output = T>,
        for<'a, 'b> &'a T: Div<&'b T, Output = T>,
    {
        let [[m00, m01, m02, _], [m10, m11, m12, _], [m20, m21, m22, _]] = self.0.raw.clone();

        let ((ax_sin, ax_cos), m22) = Self::get_sin_cos_to_zero_x_and_new_y(&m21, &m22);
        drop(m21);
        let (m01, m02) = (
            &(&m01 * &ax_cos) - &(&m02 * &ax_sin),
            &(&m01 * &ax_sin) + &(&m02 * &ax_cos),
        );
        let (m11, m12) = (
            &(&m11 * &ax_cos) - &(&m12 * &ax_sin),
            &(&m11 * &ax_sin) + &(&m12 * &ax_cos),
        );

        let ((ay_sin, ay_cos), m22) = Self::get_sin_cos_to_zero_x_and_new_y(&m20, &m22);
        drop(m20);
        let (m00, m02) = (
            &(&m00 * &ay_cos) - &(&m02 * &ay_sin),
            &(&m00 * &ay_sin) + &(&m02 * &ay_cos),
        );
        let (m10, m12) = (
            &(&m10 * &ay_cos) - &(&m12 * &ay_sin),
            &(&m10 * &ay_sin) + &(&m12 * &ay_cos),
        );

        let ((az_sin, az_cos), m11) = Self::get_sin_cos_to_zero_x_and_new_y(&m10, &m11);
        drop(m10);
        let (m00, m01) = (
            &(&m00 * &az_cos) - &(&m01 * &az_sin),
            &(&m00 * &az_sin) + &(&m01 * &az_cos),
        );

        CameraParams {
            internal: CameraInternalParams {
                fx: &m00 / &m22,
                fy: &m11 / &m22,
                x0: &m02 / &m22,
                y0: &m12 / &m22,
                s: &m01 / &m22,
            },
            external: CameraExternalParams {
                center: self.center(),
                ax_sin,
                ax_cos,
                ay_sin: -&ay_sin,
                ay_cos,
                az_sin,
                az_cos,
            },
        }
    }

    #[inline]
    fn get_sin_cos_to_zero_x_and_new_y(x: &T, y: &T) -> ((T, T), T)
    where
        T: Hypot<T, Output = T>,
        for<'a, 'b> &'a T: Div<&'b T, Output = T>,
    {
        let abs = x.hypot(y);
        let sin = x / &abs;
        let cos = y / &abs;
        ((sin, cos), abs)
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

    #[test]
    fn test_get_params() {
        let fx = 0.317448;
        let fy = 0.126929;
        let x0 = 0.449578;
        let y0 = 0.881717;
        let s = 0.251931;
        let k = Tensor2::from_raw(
            SImage0,
            CoSpace(SImage0),
            [[fx, s, x0], [0., fy, y0], [0., 0., 1.]],
        );
        let ax: f64 = 1.344503;
        let ay: f64 = 0.314033;
        let az: f64 = 2.917331;
        let (ax_sin, ax_cos) = ax.sin_cos();
        let (ay_sin, ay_cos) = ay.sin_cos();
        let (az_sin, az_cos) = az.sin_cos();
        let rx = Tensor2::from_raw(
            SImage0,
            CoSpace(SImage0),
            [[1., 0., 0.], [0., ax_cos, -ax_sin], [0., ax_sin, ax_cos]],
        );
        let ry = Tensor2::from_raw(
            SImage0,
            CoSpace(SImage0),
            [[ay_cos, 0., ay_sin], [0., 1., 0.], [-ay_sin, 0., ay_cos]],
        );
        let rz = Tensor2::from_raw(
            SImage0,
            CoSpace(SImage0),
            [[az_cos, -az_sin, 0.], [az_sin, az_cos, 0.], [0., 0., 1.]],
        );
        let r = rz.contract_tensor2_10(&ry).contract_tensor2_10(&rx);
        let cx = 0.984337;
        let cy = 0.564489;
        let cz = 0.487036;
        let pre_m = Tensor2::from_raw(
            SImage0,
            CoSpace(SWorld),
            [[1., 0., 0., -cx], [0., 1., 0., -cy], [0., 0., 1., -cz]],
        );
        let m = k
            .contract_tensor2_10(&r)
            .contract_tensor2_10(&pre_m)
            .scale(0.279980);
        let camera = Camera::from_tensor(&m);
        let params = camera.get_params();
        assert_eq!(
            (fx * 1000000.).round(),
            (params.internal.fx * 1000000.).round()
        );
        assert_eq!(
            (fy * 1000000.).round(),
            (params.internal.fy * 1000000.).round()
        );
        assert_eq!(
            (x0 * 1000000.).round(),
            (params.internal.x0 * 1000000.).round()
        );
        assert_eq!(
            (y0 * 1000000.).round(),
            (params.internal.y0 * 1000000.).round()
        );
        assert_eq!(
            (s * 1000000.).round(),
            (params.internal.s * 1000000.).round()
        );
        let params_center_normal_coords = params.external.center.normal_coords();
        assert_eq!(
            (cx * 1000000.).round(),
            (params_center_normal_coords[0] * 1000000.).round()
        );
        assert_eq!(
            (cy * 1000000.).round(),
            (params_center_normal_coords[1] * 1000000.).round()
        );
        assert_eq!(
            (cz * 1000000.).round(),
            (params_center_normal_coords[2] * 1000000.).round()
        );
        let params_rx = Tensor2::from_raw(
            SImage0,
            CoSpace(SImage0),
            [
                [1., 0., 0.],
                [0., params.external.ax_cos, -params.external.ax_sin],
                [0., params.external.ax_sin, params.external.ax_cos],
            ],
        );
        let params_ry = Tensor2::from_raw(
            SImage0,
            CoSpace(SImage0),
            [
                [params.external.ay_cos, 0., params.external.ay_sin],
                [0., 1., 0.],
                [-params.external.ay_sin, 0., params.external.ay_cos],
            ],
        );
        let params_rz = Tensor2::from_raw(
            SImage0,
            CoSpace(SImage0),
            [
                [params.external.az_cos, -params.external.az_sin, 0.],
                [params.external.az_sin, params.external.az_cos, 0.],
                [0., 0., 1.],
            ],
        );
        let params_r = params_rz
            .contract_tensor2_10(&params_ry)
            .contract_tensor2_10(&params_rx);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    (r.raw[i][j] * 1000000.).round(),
                    (params_r.raw[i][j] * 1000000.).round()
                );
            }
        }
    }
}
