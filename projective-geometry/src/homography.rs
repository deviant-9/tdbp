use crate::array_utils::ArrayExt;
use crate::camera::Camera;
use crate::homogeneous_equations::{SolveExactHomogeneousExt, SolveHomogeneousExt};
use crate::projective_primitives::{Line2D, Line3D, Plane3D, Point1D, Point2D, Point3D};
use crate::scalar_traits::{Descale, ScalarAdd, ScalarNeg, ScalarSub, Zero};
use crate::tensors::{CoSpace, Space, Tensor2};
use std::array::from_fn;
use std::ops::{Div, Mul};

#[derive(Debug)]
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

    #[inline]
    pub fn transfer_point(&self, point: &Point1D<T, SIn>) -> Point1D<T, SOut> {
        Point1D::from_contra_tensor(&self.0.contract_tensor1_10(&point.contra_tensor()))
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
    pub fn transfer_point(&self, point: &Point2D<T, SIn>) -> Point2D<T, SOut> {
        Point2D::from_contra_tensor(&self.0.contract_tensor1_10(&point.contra_tensor()))
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
    pub fn transfer_point(&self, point: &Point3D<T, SIn>) -> Point3D<T, SOut> {
        Point3D::from_contra_tensor(&self.0.contract_tensor1_10(&point.contra_tensor()))
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

macro_rules! from_exact_points_impl {
    ($n:expr, $point_type:ident) => {
        impl<T: Clone + Descale + Zero, SIn: Space<$n>, SOut: Space<$n>> H<T, SIn, SOut, $n>
        where
            for<'a> &'a T: ScalarNeg<Output = T>,
            for<'a, 'b> &'a T: ScalarAdd<&'b T, Output = T>,
            for<'a, 'b> &'a T: ScalarSub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
        {
            pub fn from_exact_points(
                points: &[($point_type<T, SIn>, $point_type<T, SOut>); $n + 1],
                random_vector: &[T; $n * $n],
            ) -> Self {
                let points_a: [[[T; $n * $n]; $n - 1]; $n + 1] =
                    points.ref_map(|(in_point, out_point)| {
                        let in_tensor = in_point.contra_tensor();
                        let in_raw = &in_tensor.raw;
                        let out_tensor = out_point.contra_tensor();
                        let out_raw = &out_tensor.raw;
                        from_fn(|equation_i| {
                            let j = equation_i + 1;
                            let mut a_ij_flat: [T; $n * $n] = from_fn(|_| T::zero());
                            let (a_ij, _) = a_ij_flat.as_chunks_mut::<$n>();
                            for k in 0..$n {
                                a_ij[0][k] = &in_raw[k] * &out_raw[j];
                                a_ij[j][k] = -&(&out_raw[0] * &in_raw[k]);
                            }
                            a_ij_flat
                        })
                    });
                let a_slice = points_a.flatten();
                let a: [[T; $n * $n]; $n * $n - 1] = from_fn(|i| a_slice[i].clone());
                let h_flat = a.solve_exact_homogeneous(random_vector);
                let (h_slice, _) = h_flat.as_chunks::<$n>();
                let h: [[T; $n]; $n] = from_fn(|i| h_slice[i].clone());
                let s_in = points[0].0.contra_tensor().s0.clone();
                let s_out = points[0].1.contra_tensor().s0.clone();
                H::from_tensor(&Tensor2::from_raw(s_out, CoSpace(s_in), h))
            }
        }
    };
}

from_exact_points_impl!(2, Point1D);
from_exact_points_impl!(3, Point2D);
from_exact_points_impl!(4, Point3D);

#[derive(Debug)]
pub enum FromPointsError {
    NotEnoughPoints,
}

macro_rules! from_points_impl {
    ($n:expr, $point_type:ident) => {
        impl<T: Clone + Descale + Zero, SIn: Space<$n>, SOut: Space<$n>> H<T, SIn, SOut, $n>
        where
            for<'a> &'a T: ScalarNeg<Output = T>,
            for<'a, 'b> &'a T: ScalarAdd<&'b T, Output = T>,
            for<'a, 'b> &'a T: ScalarSub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
            for<'a, 'b> &'a T: Div<&'b T, Output = T>,
            for<'a, 'b> &'a T: PartialOrd<&'b T>,
        {
            pub fn from_points(
                points: &[($point_type<T, SIn>, $point_type<T, SOut>)],
                random_vector: &[T; $n * $n],
            ) -> Result<Self, FromPointsError> {
                if points.len() < $n + 1 {
                    return Err(FromPointsError::NotEnoughPoints);
                }
                let mut a: Vec<[T; $n * $n]> =
                    Vec::with_capacity(points.len() * ($n * $n - $n) / 2);
                for (in_point, out_point) in points {
                    let in_tensor = in_point.contra_tensor();
                    let in_raw = &in_tensor.raw;
                    let out_tensor = out_point.contra_tensor();
                    let out_raw = &out_tensor.raw;
                    for i in 0usize..($n - 1) {
                        for j in (i + 1)..$n {
                            let mut a_ij_flat: [T; $n * $n] = from_fn(|_| T::zero());
                            let (a_ij, _) = a_ij_flat.as_chunks_mut::<$n>();
                            for k in 0..$n {
                                a_ij[i][k] = &in_raw[k] * &out_raw[j];
                                a_ij[j][k] = -&(&out_raw[i] * &in_raw[k]);
                            }
                            a.push(a_ij_flat);
                        }
                    }
                }
                let h_flat = a
                    .as_slice()
                    .solve_homogeneous(random_vector)
                    .expect("We already checked points.len()");
                let (h_slice, _) = h_flat.as_chunks::<$n>();
                let h: [[T; $n]; $n] = from_fn(|i| h_slice[i].clone());
                let s_in = points[0].0.contra_tensor().s0.clone();
                let s_out = points[0].1.contra_tensor().s0.clone();
                Ok(H::from_tensor(&Tensor2::from_raw(s_out, CoSpace(s_in), h)))
            }
        }
    };
}

from_points_impl!(2, Point1D);
from_points_impl!(3, Point2D);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_utils::ArrayExt;
    use crate::tensors::Space;

    type T = f64;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct H2InS;
    impl Space<2> for H2InS {}
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct H2OutS;
    impl Space<2> for H2OutS {}

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct H3InS;
    impl Space<3> for H3InS {}
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct H3OutS;
    impl Space<3> for H3OutS {}

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct H4InS;
    impl Space<4> for H4InS {}
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct H4OutS;
    impl Space<4> for H4OutS {}

    const RANDOM_VECTOR: [T; 16] = [
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
        0.5885218756022766,
        0.6004945401135157,
        0.1306402894815769,
        0.013365123679884294,
    ];

    #[test]
    fn test_h2_from_exact_points() {
        let normal_points = [([111.], [766.]), ([791.], [969.]), ([247.], [625.])];
        let points = normal_points.ref_map(|(p_in, p_out)| {
            (
                Point1D::from_normal_coords(p_in, H2InS),
                Point1D::from_normal_coords(p_out, H2OutS),
            )
        });
        let random_vector = from_fn(|i| RANDOM_VECTOR[i].clone());
        let h = H::<T, H2InS, H2OutS, 2>::from_exact_points(&points, &random_vector);
        assert_eq!(
            h.transfer_point(&points[0].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[0].1
        );
        assert_eq!(
            h.transfer_point(&points[1].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[1].1
        );
        assert_eq!(
            h.transfer_point(&points[2].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[2].1
        );
    }

    #[test]
    fn test_h2_from_points() {
        let normal_points = [([111.], [766.]), ([791.], [969.]), ([247.], [625.])];
        let points = normal_points.ref_map(|(p_in, p_out)| {
            (
                Point1D::from_normal_coords(p_in, H2InS),
                Point1D::from_normal_coords(p_out, H2OutS),
            )
        });
        let random_vector = from_fn(|i| RANDOM_VECTOR[i].clone());
        let h = H::<T, H2InS, H2OutS, 2>::from_points(&points, &random_vector).unwrap();
        assert_eq!(
            h.transfer_point(&points[0].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[0].1
        );
        assert_eq!(
            h.transfer_point(&points[1].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[1].1
        );
        assert_eq!(
            h.transfer_point(&points[2].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[2].1
        );
    }

    #[test]
    fn test_h3_from_exact_points() {
        let normal_points = [
            ([67., 60.], [69., 80.]),
            ([13., 19.], [17., 49.]),
            ([98., 44.], [72., 53.]),
            ([42., 40.], [68., 90.]),
        ];
        let points = normal_points.ref_map(|(p_in, p_out)| {
            (
                Point2D::from_normal_coords(p_in, H3InS),
                Point2D::from_normal_coords(p_out, H3OutS),
            )
        });
        let random_vector = from_fn(|i| RANDOM_VECTOR[i].clone());
        let h = H::<T, H3InS, H3OutS, 3>::from_exact_points(&points, &random_vector);
        assert_eq!(
            h.transfer_point(&points[0].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[0].1
        );
        assert_eq!(
            h.transfer_point(&points[1].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[1].1
        );
        assert_eq!(
            h.transfer_point(&points[2].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[2].1
        );
        assert_eq!(
            h.transfer_point(&points[3].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[3].1
        );
    }

    #[test]
    fn test_h3_from_points() {
        let normal_points = [
            ([67., 60.], [69., 80.]),
            ([13., 19.], [17., 49.]),
            ([98., 44.], [72., 53.]),
            ([42., 40.], [68., 90.]),
        ];
        let points = normal_points.ref_map(|(p_in, p_out)| {
            (
                Point2D::from_normal_coords(p_in, H3InS),
                Point2D::from_normal_coords(p_out, H3OutS),
            )
        });
        let random_vector = from_fn(|i| RANDOM_VECTOR[i].clone());
        let h = H::<T, H3InS, H3OutS, 3>::from_points(&points, &random_vector).unwrap();
        assert_eq!(
            h.transfer_point(&points[0].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[0].1
        );
        assert_eq!(
            h.transfer_point(&points[1].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[1].1
        );
        assert_eq!(
            h.transfer_point(&points[2].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[2].1
        );
        assert_eq!(
            h.transfer_point(&points[3].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[3].1
        );
    }

    #[test]
    fn test_h4_from_exact_points() {
        let normal_points = [
            ([604., 743., 887.], [318., 103., 160.]),
            ([981., 721., 460.], [137., 942., 895.]),
            ([580., 926., 298.], [168., 256., 226.]),
            ([612., 501., 859.], [498., 901., 330.]),
            ([696., 228., 969.], [856., 905., 568.]),
        ];
        let points = normal_points.ref_map(|(p_in, p_out)| {
            (
                Point3D::from_normal_coords(p_in, H4InS),
                Point3D::from_normal_coords(p_out, H4OutS),
            )
        });
        let random_vector = from_fn(|i| RANDOM_VECTOR[i].clone());
        let h = H::<T, H4InS, H4OutS, 4>::from_exact_points(&points, &random_vector);
        assert_eq!(
            h.transfer_point(&points[0].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[0].1
        );
        assert_eq!(
            h.transfer_point(&points[1].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[1].1
        );
        assert_eq!(
            h.transfer_point(&points[2].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[2].1
        );
        assert_eq!(
            h.transfer_point(&points[3].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[3].1
        );
        assert_eq!(
            h.transfer_point(&points[4].0)
                .normal_coords()
                .map(|x| x.round()),
            normal_points[4].1
        );
    }
}
