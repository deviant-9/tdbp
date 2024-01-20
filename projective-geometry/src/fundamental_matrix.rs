use crate::projective_primitives::{Line2D, Point2D};
use crate::scalar_traits::{Descale, ScalarAdd, ScalarNeg, ScalarSub, Zero};
use crate::tensors::{CoSpace, Space, Tensor2};
use std::ops::Mul;

#[derive(Copy, Clone, Debug)]
pub struct F<T, SImage0: Space<3>, SImage1: Space<3>>(
    Tensor2<T, CoSpace<3, SImage1>, CoSpace<3, SImage0>, 3, 3>,
);

impl<T: Clone + Descale + Zero, SImage0: Space<3>, SImage1: Space<3>> F<T, SImage0, SImage1>
where
    for<'a> &'a T: ScalarNeg<Output = T>,
    for<'a, 'b> &'a T: ScalarAdd<&'b T, Output = T>,
    for<'a, 'b> &'a T: ScalarSub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    #[inline]
    pub fn from_tensor(
        tensor: &Tensor2<T, CoSpace<3, SImage1>, CoSpace<3, SImage0>, 3, 3>,
    ) -> Self {
        F(tensor.descale())
    }

    #[inline]
    pub fn tensor(&self) -> Tensor2<T, CoSpace<3, SImage1>, CoSpace<3, SImage0>, 3, 3> {
        self.0.clone()
    }

    #[inline]
    pub fn swap10(&self) -> F<T, SImage1, SImage0> {
        F(self.0.swap10())
    }

    #[inline]
    pub fn project_point(&self, point: &Point2D<T, SImage0>) -> Line2D<T, SImage1> {
        Line2D::from_co_tensor(&self.0.contract_tensor1_10(&point.contra_tensor()))
    }

    #[inline]
    pub fn back_project_point(&self, point: &Point2D<T, SImage1>) -> Line2D<T, SImage0> {
        Line2D::from_co_tensor(&self.0.swap10().contract_tensor1_10(&point.contra_tensor()))
    }
}
