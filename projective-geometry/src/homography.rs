use crate::camera::Camera;
use crate::projective_primitives::{Line2D, Line3D, Plane3D, Point1D, Point2D, Point3D, T};
use crate::tensors::{CoSpace, Space, Tensor2};

#[derive(Debug)]
pub struct H<SIn: Space<N>, SOut: Space<N>, const N: usize>(
    Tensor2<T, SOut, CoSpace<N, SIn>, N, N>,
);

impl<SIn: Space<N>, SOut: Space<N>, const N: usize> H<SIn, SOut, N> {
    #[inline]
    pub fn from_tensor(tensor: &Tensor2<T, SOut, CoSpace<N, SIn>, N, N>) -> Self {
        H(tensor.descale())
    }

    #[inline]
    pub fn tensor(&self) -> Tensor2<T, SOut, CoSpace<N, SIn>, N, N> {
        self.0.clone()
    }
}

impl<SIn: Space<2>, SOut: Space<2>> H<SIn, SOut, 2> {
    #[inline]
    pub fn inverse(&self) -> H<SOut, SIn, 2> {
        H::from_tensor(&self.0.up_to_scale_inverse())
    }

    #[inline]
    pub fn transfer_point(&self, point: &Point1D<SIn>) -> Point1D<SOut> {
        Point1D::from_contra_tensor(&self.0.contract_tensor1_10(&point.contra_tensor()))
    }
}

impl<SIn: Space<3>, SOut: Space<3>> H<SIn, SOut, 3> {
    #[inline]
    pub fn inverse(&self) -> H<SOut, SIn, 3> {
        H::from_tensor(&self.0.up_to_scale_inverse())
    }

    #[inline]
    pub fn transfer_point(&self, point: &Point2D<SIn>) -> Point2D<SOut> {
        Point2D::from_contra_tensor(&self.0.contract_tensor1_10(&point.contra_tensor()))
    }

    #[inline]
    pub fn back_transfer_line(&self, line: &Line2D<SOut>) -> Line2D<SIn> {
        Line2D::from_co_tensor(&line.co_tensor().contract_tensor2_00(&self.0))
    }

    #[inline]
    pub fn transfer_camera<SWorld: Space<4>>(
        &self,
        camera: Camera<SWorld, SIn>,
    ) -> Camera<SWorld, SOut> {
        Camera::from_tensor(&self.0.contract_tensor2_10(&camera.tensor()))
    }
}

impl<SIn: Space<4>, SOut: Space<4>> H<SIn, SOut, 4> {
    #[inline]
    pub fn inverse(&self) -> H<SOut, SIn, 4> {
        H::from_tensor(&self.0.up_to_scale_inverse())
    }

    #[inline]
    pub fn transfer_point(&self, point: &Point3D<SIn>) -> Point3D<SOut> {
        Point3D::from_contra_tensor(&self.0.contract_tensor1_10(&point.contra_tensor()))
    }

    #[inline]
    pub fn transfer_line(&self, line: &Line3D<SIn>) -> Line3D<SOut> {
        Line3D::from_contra_tensor(
            &self
                .0
                .contract_tensor2_10(&self.0.contract_tensor2_10(&line.contra_tensor()).swap10()),
        )
    }

    #[inline]
    pub fn back_transfer_line(&self, line: &Line3D<SOut>) -> Line3D<SIn> {
        Line3D::from_co_tensor(
            &line
                .co_tensor()
                .contract_tensor2_10(&self.0)
                .swap10()
                .contract_tensor2_10(&self.0),
        )
    }

    #[inline]
    pub fn back_transfer_plane(&self, plane: &Plane3D<SOut>) -> Plane3D<SIn> {
        Plane3D::from_co_tensor(&plane.co_tensor().contract_tensor2_00(&self.0))
    }

    #[inline]
    pub fn back_transfer_camera<SImage: Space<3>>(
        &self,
        camera: &Camera<SOut, SImage>,
    ) -> Camera<SIn, SImage> {
        Camera::from_tensor(&camera.tensor().contract_tensor2_10(&self.0))
    }
}
