use crate::array_utils::ArrayExt;
use crate::homogeneous_equations::{
    get_ax_collinear_y_equations_for_a, ExactHomogeneousSolver, HomogeneousSolver,
};
use crate::homography::H;
use crate::projective_primitives::Point;
use crate::scalar_traits::{Descale, ScalarAdd, ScalarNeg, ScalarSub, Zero};
use crate::tensors::{CoSpace, Space, Tensor2};
use std::ops::{Div, Mul};

pub trait HFromExactPointsSolver<T, const N: usize, const POINTS_N: usize> {
    fn solve<SIn: Space<N>, SOut: Space<N>>(
        &self,
        points: &[(Point<T, SIn, N>, Point<T, SOut, N>); POINTS_N],
    ) -> H<T, SIn, SOut, N>;
}

pub struct HFromExactPointsSolverImpl<Solver> {
    equations_solver: Solver,
}

impl<Solver> HFromExactPointsSolverImpl<Solver> {
    #[inline]
    pub fn new<T, const N_SQR: usize, const N_SQR_MINUS_1: usize>(equations_solver: Solver) -> Self
    where
        Solver: ExactHomogeneousSolver<T, N_SQR, N_SQR_MINUS_1>,
    {
        Self { equations_solver }
    }
}

macro_rules! from_exact_points_impl {
    ($n:expr) => {
        impl<
                T: Clone + Descale + Zero,
                Solver: ExactHomogeneousSolver<T, { $n * $n }, { $n * $n - 1 }>,
            > HFromExactPointsSolver<T, $n, { $n + 1 }> for HFromExactPointsSolverImpl<Solver>
        where
            for<'a> &'a T: ScalarNeg<Output = T>,
            for<'a, 'b> &'a T: ScalarAdd<&'b T, Output = T>,
            for<'a, 'b> &'a T: ScalarSub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
        {
            fn solve<SIn: Space<$n>, SOut: Space<$n>>(
                &self,
                points: &[(Point<T, SIn, $n>, Point<T, SOut, $n>); $n + 1],
            ) -> H<T, SIn, SOut, $n> {
                let points_a: [[[T; $n * $n]; $n - 1]; $n + 1] =
                    points.ref_map(|(in_point, out_point)| {
                        get_ax_collinear_y_equations_for_a(
                            &in_point.contra_tensor().raw,
                            &out_point.contra_tensor().raw,
                        )
                    });
                let a_slice_ref: &[[T; $n * $n]] = points_a.flatten();
                let a_ref: &[[T; $n * $n]; $n * $n - 1] = a_slice_ref.try_into().unwrap();
                let h_flat = self.equations_solver.solve(a_ref);
                let (h_slice, _) = h_flat.as_chunks::<$n>();
                let h_ref: &[[T; $n]; $n] = h_slice.try_into().unwrap();
                let h: [[T; $n]; $n] = h_ref.clone();
                let s_in = points[0].0.get_s();
                let s_out = points[0].1.get_s();
                H::from_tensor(&Tensor2::from_raw(s_out, CoSpace(s_in), h))
            }
        }
    };
}

from_exact_points_impl!(2);
from_exact_points_impl!(3);
from_exact_points_impl!(4);

#[derive(Debug)]
pub enum FromPointsError {
    NotEnoughPoints,
}

pub trait HFromPointsSolver<T, const N: usize> {
    fn solve<SIn: Space<N>, SOut: Space<N>>(
        &self,
        points: &[(Point<T, SIn, N>, Point<T, SOut, N>)],
    ) -> Result<H<T, SIn, SOut, N>, FromPointsError>;
}

pub struct HFromPointsSolverImpl<Solver> {
    equations_solver: Solver,
}

impl<Solver> HFromPointsSolverImpl<Solver> {
    #[inline]
    pub fn new<T, const N_SQR: usize>(equations_solver: Solver) -> Self
    where
        Solver: HomogeneousSolver<T, N_SQR>,
    {
        Self { equations_solver }
    }
}

macro_rules! from_points_impl {
    ($n:expr) => {
        impl<T: Clone + Descale + Zero, Solver: HomogeneousSolver<T, { $n * $n }>>
            HFromPointsSolver<T, $n> for HFromPointsSolverImpl<Solver>
        where
            for<'a> &'a T: ScalarNeg<Output = T>,
            for<'a, 'b> &'a T: ScalarAdd<&'b T, Output = T>,
            for<'a, 'b> &'a T: ScalarSub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
            for<'a, 'b> &'a T: Div<&'b T, Output = T>,
            for<'a, 'b> &'a T: PartialOrd<&'b T>,
        {
            fn solve<SIn: Space<$n>, SOut: Space<$n>>(
                &self,
                points: &[(Point<T, SIn, $n>, Point<T, SOut, $n>)],
            ) -> Result<H<T, SIn, SOut, $n>, FromPointsError> {
                if points.len() < $n + 1 {
                    return Err(FromPointsError::NotEnoughPoints);
                }
                let points_a: Box<[[[T; $n * $n]; $n - 1]]> = points
                    .iter()
                    .map(|(in_point, out_point)| {
                        get_ax_collinear_y_equations_for_a(
                            &in_point.contra_tensor().raw,
                            &out_point.contra_tensor().raw,
                        )
                    })
                    .collect();
                let h_flat = self
                    .equations_solver
                    .solve(points_a.flatten())
                    .expect("We already checked points.len()");
                let (h_slice, _) = h_flat.as_chunks::<$n>();
                let h_ref: &[[T; $n]; $n] = h_slice.try_into().unwrap();
                let h: [[T; $n]; $n] = h_ref.clone();
                let s_in = points[0].0.get_s();
                let s_out = points[0].1.get_s();
                Ok(H::from_tensor(&Tensor2::from_raw(s_out, CoSpace(s_in), h)))
            }
        }
    };
}

from_points_impl!(2);
from_points_impl!(3);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_utils::ArrayExt;
    use crate::eigen_vectors::{MaxEigenValueVectorSolverImpl, MinEigenValueVectorSolverImpl};
    use crate::homogeneous_equations::{HomogeneousSolverImpl, NoSqrtExactHomogeneousSolverImpl};
    use crate::projective_primitives::{Point1D, Point2D, Point3D};
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
        let random_vector: &[f64; 2 * 2] = RANDOM_VECTOR[..2 * 2].try_into().unwrap();
        let equations_solver = NoSqrtExactHomogeneousSolverImpl::new(random_vector);
        let solver = HFromExactPointsSolverImpl::new(equations_solver);
        let h = solver.solve(&points);
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
        let random_vector: &[f64; 2 * 2] = RANDOM_VECTOR[..2 * 2].try_into().unwrap();
        let max_solver = MaxEigenValueVectorSolverImpl::new(random_vector);
        let min_solver = MinEigenValueVectorSolverImpl::new(max_solver);
        let equations_solver = HomogeneousSolverImpl::new(min_solver);
        let h_solver = HFromPointsSolverImpl::new(equations_solver);
        let h = h_solver.solve::<H2InS, H2OutS>(&points).unwrap();
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
        let random_vector: &[f64; 3 * 3] = RANDOM_VECTOR[..3 * 3].try_into().unwrap();
        let equations_solver = NoSqrtExactHomogeneousSolverImpl::new(random_vector);
        let solver = HFromExactPointsSolverImpl::new(equations_solver);
        let h = solver.solve(&points);
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
        let random_vector: &[f64; 3 * 3] = RANDOM_VECTOR[..3 * 3].try_into().unwrap();
        let max_solver = MaxEigenValueVectorSolverImpl::new(random_vector);
        let min_solver = MinEigenValueVectorSolverImpl::new(max_solver);
        let equations_solver = HomogeneousSolverImpl::new(min_solver);
        let h_solver = HFromPointsSolverImpl::new(equations_solver);
        let h = h_solver.solve::<H3InS, H3OutS>(&points).unwrap();
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
        let random_vector: &[f64; 4 * 4] = RANDOM_VECTOR[..4 * 4].try_into().unwrap();
        let equations_solver = NoSqrtExactHomogeneousSolverImpl::new(random_vector);
        let solver = HFromExactPointsSolverImpl::new(equations_solver);
        let h = solver.solve(&points);
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
