use crate::array_utils::ArrayExt;
use crate::eigen_vectors::MinEigenValueVectorSolver;
use crate::scalar_traits::{Descale, Sqrt, Zero};
use crate::tensors::{CoSpace, Space, Tensor2};
use std::array::from_fn;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait ExactHomogeneousSolver<T, const N: usize, const EQUATIONS_N: usize> {
    fn solve(&self, matrix: &[[T; N]; EQUATIONS_N]) -> [T; N];
}

pub struct WithSqrtExactHomogeneousSolverImpl<T, const N: usize> {
    random_vector: [T; N],
}

impl<T: Clone + Descale, const N: usize> WithSqrtExactHomogeneousSolverImpl<T, N> {
    #[inline]
    pub fn new(random_vector: &[T; N]) -> Self {
        Self {
            random_vector: descale(random_vector),
        }
    }
}

macro_rules! solve_exact_homogeneous_impl_with_sqrt {
    ($n:expr) => {
        impl<T: Clone + Zero + Descale + Sqrt<Output = T>> ExactHomogeneousSolver<T, $n, { $n - 1 }>
            for WithSqrtExactHomogeneousSolverImpl<T, $n>
        where
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
            for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
            for<'a, 'b> &'a T: Div<&'b T, Output = T>,
        {
            fn solve(&self, matrix: &[[T; $n]; $n - 1]) -> [T; $n] {
                let mut a = matrix.clone();
                for i in 0..a.len() {
                    let (a_i, a_prev_all) = a[0..=i].split_last_mut().unwrap();
                    for a_prev in a_prev_all {
                        add_to_orthogonality(a_i, a_prev);
                    }
                    let curr_abs = abs(a_i);
                    for curr_k in a_i.iter_mut() {
                        *curr_k = &*curr_k / &curr_abs;
                    }
                }
                let mut random_vector = self.random_vector.clone();
                for a_i in a.iter() {
                    add_to_orthogonality(&mut random_vector, a_i);
                }
                descale(&random_vector)
            }
        }
    };
}

pub struct NoSqrtExactHomogeneousSolverImpl<T, const N: usize> {
    random_vector: [T; N],
}

impl<T: Clone + Descale, const N: usize> NoSqrtExactHomogeneousSolverImpl<T, N> {
    #[inline]
    pub fn new(random_vector: &[T; N]) -> Self {
        Self {
            random_vector: descale(random_vector),
        }
    }
}

macro_rules! solve_exact_homogeneous_impl_without_sqrt {
    ($n:expr) => {
        impl<T: Clone + Zero + Descale> ExactHomogeneousSolver<T, $n, { $n - 1 }>
            for NoSqrtExactHomogeneousSolverImpl<T, $n>
        where
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
            for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
        {
            fn solve(&self, matrix: &[[T; $n]; $n - 1]) -> [T; $n] {
                let mut a = matrix.clone();
                let mut abs_sqrs: [Option<T>; $n - 1] = from_fn(|_| None);
                for i in 0..a.len() {
                    let (a_i, a_prev_all) = a[0..=i].split_last_mut().unwrap();
                    *a_i = descale(a_i);
                    for prev_i in 0..a_prev_all.len() {
                        add_to_orthogonality_special(
                            a_i,
                            &a_prev_all[prev_i],
                            (&abs_sqrs[prev_i]).as_ref().unwrap(),
                        );
                        *a_i = descale(a_i);
                    }
                    abs_sqrs[i] = Some(dot_product(a_i, a_i));
                }
                let mut random_vector = self.random_vector.clone();
                for i in 0..a.len() {
                    add_to_orthogonality_special(
                        &mut random_vector,
                        &a[i],
                        (&abs_sqrs[i]).as_ref().unwrap(),
                    );
                }
                descale(&random_vector)
            }
        }
    };
}

macro_rules! solve_exact_homogeneous_impl {
    ($n:expr) => {
        solve_exact_homogeneous_impl_with_sqrt!($n);
        solve_exact_homogeneous_impl_without_sqrt!($n);
    };
}

solve_exact_homogeneous_impl!(2);
solve_exact_homogeneous_impl!(3);
solve_exact_homogeneous_impl!(4);
solve_exact_homogeneous_impl!(5);
solve_exact_homogeneous_impl!(6);
solve_exact_homogeneous_impl!(7);
solve_exact_homogeneous_impl!(8);
solve_exact_homogeneous_impl!(9);
solve_exact_homogeneous_impl!(10);
solve_exact_homogeneous_impl!(11);
solve_exact_homogeneous_impl!(12);
solve_exact_homogeneous_impl!(13);
solve_exact_homogeneous_impl!(14);
solve_exact_homogeneous_impl!(15);
solve_exact_homogeneous_impl!(16);

#[inline]
fn descale<T: Descale, const N: usize>(v: &[T; N]) -> [T; N] {
    let factor = Descale::descaling_factor(v.iter());
    v.ref_map(|x| x.descale(&factor))
}

#[inline]
fn add_to_orthogonality<T: Zero, const N: usize>(lhs: &mut [T; N], normalized_rhs: &[T; N])
where
    for<'a, 'b> &'a T: Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    let proj = dot_product(lhs, normalized_rhs);
    for (l, r) in lhs.iter_mut().zip(normalized_rhs.iter()) {
        *l = &*l - &(&proj * r);
    }
}

#[inline]
fn add_to_orthogonality_special<T: Zero, const N: usize>(
    lhs: &mut [T; N],
    rhs: &[T; N],
    rhs_abs_sqr: &T,
) where
    for<'a, 'b> &'a T: Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    let proj = dot_product(lhs, rhs);
    for (l, r) in lhs.iter_mut().zip(rhs.iter()) {
        *l = &(&*l * &rhs_abs_sqr) - &(&proj * r);
    }
}

#[inline]
fn abs<T: Zero + Sqrt<Output = T>, const N: usize>(v: &[T; N]) -> T
where
    for<'a, 'b> &'a T: Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    dot_product(v, v).sqrt()
}

#[inline]
fn dot_product<T: Zero, const N: usize>(lhs: &[T; N], rhs: &[T; N]) -> T
where
    for<'a, 'b> &'a T: Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    lhs.iter()
        .zip(rhs.iter())
        .map(|(l, r)| l * r)
        .fold(T::zero(), |acc, x| &acc + &x)
}

#[derive(Debug)]
pub enum SolveHomogeneousError {
    NotEnoughEquations,
}

pub trait HomogeneousSolver<T, const N: usize> {
    fn solve(&self, matrix: &[[T; N]]) -> Result<[T; N], SolveHomogeneousError>;
}

pub struct HomogeneousSolverImpl<MinSolver> {
    min_solver: MinSolver,
}

impl<MinSolver> HomogeneousSolverImpl<MinSolver> {
    #[inline]
    pub fn new(min_solver: MinSolver) -> Self {
        Self { min_solver }
    }
}

macro_rules! solve_homogeneous_impl {
    ($n:expr) => {
        impl<T: Clone + Zero + Descale, MinSolver: MinEigenValueVectorSolver<T, $n>>
            HomogeneousSolver<T, $n> for HomogeneousSolverImpl<MinSolver>
        where
            for<'a> &'a T: Neg<Output = T>,
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
            for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
            for<'a, 'b> &'a T: Div<&'b T, Output = T>,
            for<'a, 'b> &'a T: PartialOrd<&'b T>,
        {
            fn solve(&self, matrix: &[[T; $n]]) -> Result<[T; $n], SolveHomogeneousError> {
                if matrix.len() < $n - 1 {
                    return Err(SolveHomogeneousError::NotEnoughEquations);
                }
                let m: Tensor2<T, SolutionSpace<$n>, CoSpace<$n, SolutionSpace<$n>>, $n, $n> =
                    Tensor2::from_raw(
                        SolutionSpace,
                        CoSpace(SolutionSpace),
                        from_fn(|i| {
                            from_fn(|j| {
                                matrix
                                    .iter()
                                    .map(|a_k| &a_k[i])
                                    .zip(matrix.iter().map(|a_k| &a_k[j]))
                                    .fold(T::zero(), |acc, (a_ki, a_kj)| &acc + &(a_ki * a_kj))
                            })
                        }),
                    );
                Ok(self.min_solver.min_eigen_value_vector(&m).raw.clone())
            }
        }
    };
}

solve_homogeneous_impl!(2);
solve_homogeneous_impl!(3);
solve_homogeneous_impl!(4);
solve_homogeneous_impl!(5);
solve_homogeneous_impl!(6);
solve_homogeneous_impl!(7);
solve_homogeneous_impl!(8);
solve_homogeneous_impl!(9);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SolutionSpace<const N: usize>;

impl<const N: usize> Space<N> for SolutionSpace<N> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eigen_vectors::{MaxEigenValueVectorSolverImpl, MinEigenValueVectorSolverImpl};

    #[test]
    fn test_solve_exact_homogeneous_3_with_sqrt() {
        let random_vector: [f64; 3] = [1., 2., 3.];
        do_test_solve_exact_homogeneous_3(&WithSqrtExactHomogeneousSolverImpl::new(&random_vector));
    }

    #[test]
    fn test_solve_exact_homogeneous_3_without_sqrt() {
        let random_vector: [f64; 3] = [1., 2., 3.];
        do_test_solve_exact_homogeneous_3(&NoSqrtExactHomogeneousSolverImpl::new(&random_vector));
    }

    fn do_test_solve_exact_homogeneous_3<Solver: ExactHomogeneousSolver<f64, 3, 2>>(
        solver: &Solver,
    ) {
        let a: [[f64; 3]; 2] = [[965., 880., 756.], [470., 332., 822.]];
        let null_vector = solver.solve(&a);
        let null_vector_abs = abs(&null_vector);
        assert!(null_vector_abs >= 0.1, "null vector is too short");
        assert!(
            null_vector_abs < (null_vector.len() as f64 * 10.),
            "null vector is too long"
        );
        assert!(dot_product(&a[0], &null_vector).abs() < 0.000000001);
        assert!(dot_product(&a[1], &null_vector).abs() < 0.000000001);
    }

    #[test]
    fn test_solve_exact_homogeneous_4_with_sqrt() {
        let random_vector: [f64; 4] = [1., 2., 3., 4.];
        do_test_solve_exact_homogeneous_4(&WithSqrtExactHomogeneousSolverImpl::new(&random_vector));
    }

    #[test]
    fn test_solve_exact_homogeneous_4_without_sqrt() {
        let random_vector: [f64; 4] = [1., 2., 3., 4.];
        do_test_solve_exact_homogeneous_4(&NoSqrtExactHomogeneousSolverImpl::new(&random_vector));
    }

    fn do_test_solve_exact_homogeneous_4<Solver: ExactHomogeneousSolver<f64, 4, 3>>(
        solver: &Solver,
    ) {
        let a: [[f64; 4]; 3] = [
            [965., 880., 756., 295.],
            [470., 332., 822., 748.],
            [529., 139., 729., 625.],
        ];
        let null_vector = solver.solve(&a);
        let null_vector_abs = abs(&null_vector);
        assert!(null_vector_abs >= 0.1, "null vector is too short");
        assert!(
            null_vector_abs < (null_vector.len() as f64 * 10.),
            "null vector is too long"
        );
        assert!(dot_product(&a[0], &null_vector).abs() < 0.000000001);
        assert!(dot_product(&a[1], &null_vector).abs() < 0.000000001);
    }

    #[test]
    fn test_solve_homogeneous() {
        let a: [[f64; 3]; 2] = [[965., 880., 756.], [470., 332., 822.]];
        let random_vector: [f64; 3] = [1., 2., 3.];
        let max_solver = MaxEigenValueVectorSolverImpl::new(&random_vector);
        let min_solver = MinEigenValueVectorSolverImpl::new(max_solver);
        let solver = HomogeneousSolverImpl::new(min_solver);
        let null_vector = solver.solve(&a).unwrap();
        let null_vector_abs = abs(&null_vector);
        assert!(null_vector_abs >= 1., "null vector is too short");
        assert!(
            null_vector_abs < (null_vector.len() as f64 * 10.),
            "null vector is too long"
        );
        assert!(dot_product(&a[0], &null_vector).abs() < 0.000000001);
        assert!(dot_product(&a[1], &null_vector).abs() < 0.000000001);
    }

    fn abs(v: &[f64]) -> f64 {
        dot_product(v, v).sqrt()
    }

    fn dot_product(lhs: &[f64], rhs: &[f64]) -> f64 {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum()
    }
}
