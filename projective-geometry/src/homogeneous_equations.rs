use crate::eigen_vectors::MinEigenValueVectorSolver;
use crate::scalar_traits::{descale_array, Descale, Sqrt, Zero};
use crate::tensors::{CoSpace, Space, Tensor2};
use std::array::from_fn;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait ExactHomogeneousSolver<T, const VARS_N: usize, const EQUATIONS_N: usize> {
    fn solve(&self, matrix: &[[T; VARS_N]; EQUATIONS_N]) -> [T; VARS_N];
}

pub struct WithSqrtExactHomogeneousSolverImpl<T, const VARS_N: usize> {
    random_vector: [T; VARS_N],
}

impl<T: Clone + Descale, const VARS_N: usize> WithSqrtExactHomogeneousSolverImpl<T, VARS_N> {
    #[inline]
    pub fn new(random_vector: &[T; VARS_N]) -> Self {
        Self {
            random_vector: descale_array(random_vector),
        }
    }
}

macro_rules! solve_exact_homogeneous_impl_with_sqrt {
    ($vars_n:expr) => {
        impl<T: Clone + Zero + Descale + Sqrt<Output = T>>
            ExactHomogeneousSolver<T, $vars_n, { $vars_n - 1 }>
            for WithSqrtExactHomogeneousSolverImpl<T, $vars_n>
        where
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
            for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
            for<'a, 'b> &'a T: Div<&'b T, Output = T>,
        {
            fn solve(&self, matrix: &[[T; $vars_n]; $vars_n - 1]) -> [T; $vars_n] {
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
                descale_array(&random_vector)
            }
        }
    };
}

pub struct NoSqrtExactHomogeneousSolverImpl<T, const VARS_N: usize> {
    random_vector: [T; VARS_N],
}

impl<T: Clone + Descale, const VARS_N: usize> NoSqrtExactHomogeneousSolverImpl<T, VARS_N> {
    #[inline]
    pub fn new(random_vector: &[T; VARS_N]) -> Self {
        Self {
            random_vector: descale_array(random_vector),
        }
    }
}

macro_rules! solve_exact_homogeneous_impl_without_sqrt {
    ($vars_n:expr) => {
        impl<T: Clone + Zero + Descale> ExactHomogeneousSolver<T, $vars_n, { $vars_n - 1 }>
            for NoSqrtExactHomogeneousSolverImpl<T, $vars_n>
        where
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
            for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
        {
            fn solve(&self, matrix: &[[T; $vars_n]; $vars_n - 1]) -> [T; $vars_n] {
                let mut a = matrix.clone();
                let mut abs_sqrs: [Option<T>; $vars_n - 1] = from_fn(|_| None);
                for i in 0..a.len() {
                    let (a_i, a_prev_all) = a[0..=i].split_last_mut().unwrap();
                    *a_i = descale_array(a_i);
                    for prev_i in 0..a_prev_all.len() {
                        add_to_orthogonality_special(
                            a_i,
                            &a_prev_all[prev_i],
                            (&abs_sqrs[prev_i]).as_ref().unwrap(),
                        );
                        *a_i = descale_array(a_i);
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
                descale_array(&random_vector)
            }
        }
    };
}

macro_rules! solve_exact_homogeneous_impl {
    ($vars_n:expr) => {
        solve_exact_homogeneous_impl_with_sqrt!($vars_n);
        solve_exact_homogeneous_impl_without_sqrt!($vars_n);
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

pub trait HomogeneousSolver<T, const VARS_N: usize> {
    fn solve(&self, matrix: &[[T; VARS_N]]) -> Result<[T; VARS_N], SolveHomogeneousError>;
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
    ($vars_n:expr) => {
        impl<T: Clone + Zero + Descale, MinSolver: MinEigenValueVectorSolver<T, $vars_n>>
            HomogeneousSolver<T, $vars_n> for HomogeneousSolverImpl<MinSolver>
        where
            for<'a> &'a T: Neg<Output = T>,
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
            for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
            for<'a, 'b> &'a T: Div<&'b T, Output = T>,
            for<'a, 'b> &'a T: PartialOrd<&'b T>,
        {
            fn solve(
                &self,
                matrix: &[[T; $vars_n]],
            ) -> Result<[T; $vars_n], SolveHomogeneousError> {
                if matrix.len() < $vars_n - 1 {
                    return Err(SolveHomogeneousError::NotEnoughEquations);
                }
                let m: Tensor2<
                    T,
                    SolutionSpace<$vars_n>,
                    CoSpace<$vars_n, SolutionSpace<$vars_n>>,
                    $vars_n,
                    $vars_n,
                > = Tensor2::from_raw(
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
                Ok(self.min_solver.min_eigen_value_vector(&m).raw)
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
struct SolutionSpace<const VARS_N: usize>;

impl<const VARS_N: usize> Space<VARS_N> for SolutionSpace<VARS_N> {}

pub fn get_ax_collinear_y_equations_for_a<T: Zero, const X_N: usize, const Y_N: usize>(
    x: &[T; X_N],
    y: &[T; Y_N],
) -> [[T; X_N * Y_N]; Y_N - 1]
where
    for<'a> &'a T: Neg<Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    from_fn(|equation_i| {
        let j = equation_i + 1;
        let mut a_ij_flat: [T; X_N * Y_N] = from_fn(|_| T::zero());
        let (a_ij, _) = a_ij_flat.as_chunks_mut::<X_N>();
        for k in 0..X_N {
            a_ij[0][k] = &x[k] * &y[j];
            a_ij[j][k] = -&(&y[0] * &x[k]);
        }
        a_ij_flat
    })
}

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

    #[test]
    fn test_get_ax_collinear_y_equations_for_a() {
        let a: [[f64; 4]; 3] = [
            [28., 186., 546., 821.],
            [366., 550., 801., 588.],
            [361., 61., 256., 386.],
        ];
        let x: [f64; 4] = [910., 359., 268., 989.];
        let y: [f64; 3] = [
            dot_product(&a[0], &x),
            dot_product(&a[1], &x),
            dot_product(&a[2], &x),
        ];
        let equations: [[f64; 3 * 4]; 3 - 1] = get_ax_collinear_y_equations_for_a(&x, &y);
        let equations_abs = abs(equations.flatten());
        assert!(
            equations_abs >= 0.1,
            "equations coefficients are all too small"
        );
        let flat_a = a.flatten();
        for equation in &equations {
            assert!(dot_product(equation, flat_a) < 0.000000001);
        }
    }

    fn abs(v: &[f64]) -> f64 {
        dot_product(v, v).sqrt()
    }

    fn dot_product(lhs: &[f64], rhs: &[f64]) -> f64 {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum()
    }
}
