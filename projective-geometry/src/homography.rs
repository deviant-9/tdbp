use crate::camera::Camera;
use crate::fundamental_matrix::F;
use crate::projective_primitives::{Line2D, Line3D, Plane3D, Point, Point2D};
use crate::scalar_traits::{Descale, ScalarAdd, ScalarNeg, ScalarSub, Zero};
use crate::tensors::{CoSpace, Space, Tensor2};
use std::ops::Mul;

#[derive(Copy, Clone, Debug)]
pub struct H<T, SIn: Space<N>, SOut: Space<N>, const N: usize>(
    Tensor2<T, SOut, CoSpace<N, SIn>, N, N>,
);

impl<T: Clone + Descale + Zero, SIn: Space<N>, SOut: Space<N>, const N: usize> H<T, SIn, SOut, N>
where
    for<'a> &'a T: ScalarNeg<Output = T>,
    for<'a, 'b> &'a T: ScalarAdd<&'b T, Output = T>,
    for<'a, 'b> &'a T: ScalarSub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    #[inline]
    pub fn from_tensor(tensor: &Tensor2<T, SOut, CoSpace<N, SIn>, N, N>) -> Self {
        H(tensor.descale())
    }

    #[inline]
    pub fn tensor(&self) -> Tensor2<T, SOut, CoSpace<N, SIn>, N, N> {
        self.0.clone()
    }

    #[inline]
    pub fn transfer_point(&self, point: &Point<T, SIn, N>) -> Point<T, SOut, N> {
        Point::from_contra_tensor(&self.0.contract_tensor1_10(&point.contra_tensor()))
    }
}

impl<T: Clone + Descale + Zero, SIn: Space<2>, SOut: Space<2>> H<T, SIn, SOut, 2>
where
    for<'a> &'a T: ScalarNeg<Output = T>,
    for<'a, 'b> &'a T: ScalarAdd<&'b T, Output = T>,
    for<'a, 'b> &'a T: ScalarSub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    #[inline]
    pub fn inverse(&self) -> H<T, SOut, SIn, 2> {
        H::from_tensor(&self.0.adjugate())
    }
}

impl<T: Clone + Descale + Zero, SIn: Space<3>, SOut: Space<3>> H<T, SIn, SOut, 3>
where
    for<'a> &'a T: ScalarNeg<Output = T>,
    for<'a, 'b> &'a T: ScalarAdd<&'b T, Output = T>,
    for<'a, 'b> &'a T: ScalarSub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    #[inline]
    pub fn inverse(&self) -> H<T, SOut, SIn, 3> {
        H::from_tensor(&self.0.adjugate())
    }

    #[inline]
    pub fn back_transfer_line(&self, line: &Line2D<T, SOut>) -> Line2D<T, SIn> {
        Line2D::from_co_tensor(&line.co_tensor().contract_tensor2_00(&self.0))
    }

    #[inline]
    pub fn transfer_camera<SWorld: Space<4>>(
        &self,
        camera: Camera<T, SWorld, SIn>,
    ) -> Camera<T, SWorld, SOut> {
        Camera::from_tensor(&self.0.contract_tensor2_10(&camera.tensor()))
    }

    #[inline]
    pub fn fundamental_matrix_for_e1(&self, e1: &Point2D<T, SOut>) -> F<T, SIn, SOut> {
        F::from_tensor(&e1.co_tensor().contract_tensor2_10(&self.0))
    }
}

impl<T: Clone + Descale + Zero, SIn: Space<4>, SOut: Space<4>> H<T, SIn, SOut, 4>
where
    for<'a> &'a T: ScalarNeg<Output = T>,
    for<'a, 'b> &'a T: ScalarAdd<&'b T, Output = T>,
    for<'a, 'b> &'a T: ScalarSub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    #[inline]
    pub fn inverse(&self) -> H<T, SOut, SIn, 4> {
        H::from_tensor(&self.0.adjugate())
    }

    #[inline]
    pub fn transfer_line(&self, line: &Line3D<T, SIn>) -> Line3D<T, SOut> {
        Line3D::from_contra_tensor(
            &self
                .0
                .contract_tensor2_10(&self.0.contract_tensor2_10(&line.contra_tensor()).swap10()),
        )
    }

    #[inline]
    pub fn back_transfer_line(&self, line: &Line3D<T, SOut>) -> Line3D<T, SIn> {
        Line3D::from_co_tensor(
            &line
                .co_tensor()
                .contract_tensor2_10(&self.0)
                .swap10()
                .contract_tensor2_10(&self.0),
        )
    }

    #[inline]
    pub fn back_transfer_plane(&self, plane: &Plane3D<T, SOut>) -> Plane3D<T, SIn> {
        Plane3D::from_co_tensor(&plane.co_tensor().contract_tensor2_00(&self.0))
    }

    #[inline]
    pub fn back_transfer_camera<SImage: Space<3>>(
        &self,
        camera: &Camera<T, SOut, SImage>,
    ) -> Camera<T, SIn, SImage> {
        Camera::from_tensor(&camera.tensor().contract_tensor2_10(&self.0))
    }
}
