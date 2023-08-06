use crate::array_utils::ArrayExt;
use crate::projective_primitives::{Line2D, Line3D, Plane3D, Point2D, Point3D, T};
use crate::tensors::{CoSpace, Space, Tensor2};

pub struct Camera<SWorld: Space<4>, SImage: Space<3>>(Tensor2<T, SImage, CoSpace<4, SWorld>, 3, 4>);

impl<SWorld: Space<4>, SImage: Space<3>> Camera<SWorld, SImage> {
    #[inline]
    pub fn from_tensor(tensor: &Tensor2<T, SImage, CoSpace<4, SWorld>, 3, 4>) -> Self {
        Camera(tensor.descale())
    }

    #[inline]
    pub fn tensor(&self) -> Tensor2<T, SImage, CoSpace<4, SWorld>, 3, 4> {
        self.0.clone()
    }

    #[inline]
    pub fn center(&self) -> Point3D<SWorld> {
        let [plane_x, plane_y, plane_inf] =
            self.0.sub_tensors().ref_map(|t| Plane3D::from_co_tensor(t));
        plane_x.cross_with_planes(&plane_y, &plane_inf)
    }

    #[inline]
    pub fn project_point(&self, point: &Point3D<SWorld>) -> Point2D<SImage> {
        Point2D::from_contra_tensor(&self.0.contract_tensor1_10(&point.contra_tensor()))
    }

    #[inline]
    pub fn project_line(&self, line: &Line3D<SWorld>) -> Line2D<SImage> {
        let line_tensor = line.contra_tensor();
        Line2D::from_contra_tensor(
            &self
                .0
                .contract_tensor2_10(&self.0.contract_tensor2_10(&line_tensor).swap10()),
        )
    }

    #[inline]
    pub fn back_project_point(&self, point: &Point2D<SImage>) -> Line3D<SWorld> {
        Line3D::from_co_tensor(
            &point
                .co_tensor()
                .contract_tensor2_10(&self.0)
                .swap10()
                .contract_tensor2_10(&self.0),
        )
    }

    #[inline]
    pub fn back_project_line(&self, line: &Line2D<SImage>) -> Plane3D<SWorld> {
        Plane3D::from_co_tensor(&line.co_tensor().contract_tensor2_00(&self.0))
    }
}
