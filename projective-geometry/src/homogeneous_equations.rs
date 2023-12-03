use crate::array_utils::ArrayExt;
use crate::scalar_traits::{Descale, Sqrt, Zero};
use crate::tensors::{CoSpace, Space, Tensor1, Tensor2};
use std::array::from_fn;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait SolveExactHomogeneousExt<RandomVector> {
    type Output;

    fn solve_exact_homogeneous(&self, random_vector: &RandomVector) -> Self::Output;
}

macro_rules! solve_exact_homogeneous_impl_with_sqrt {
    ($n:expr) => {
        impl<T: Clone + Zero + Descale + Sqrt<Output = T>> SolveExactHomogeneousExt<[T; $n]>
            for [[T; $n]; $n - 1]
        where
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
            for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
            for<'a, 'b> &'a T: Div<&'b T, Output = T>,
        {
            type Output = [T; $n];

            fn solve_exact_homogeneous(&self, random_vector: &[T; $n]) -> Self::Output {
                let mut a = self.clone();
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
                let mut random_vector = random_vector.clone();
                for a_i in a.iter() {
                    add_to_orthogonality(&mut random_vector, a_i);
                }
                descale(&random_vector)
            }
        }
    };
}

macro_rules! solve_exact_homogeneous_impl_without_sqrt {
    ($n:expr) => {
        impl<T: Clone + Zero + Descale> SolveExactHomogeneousExt<[T; $n]> for [[T; $n]; $n - 1]
        where
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
            for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
        {
            type Output = [T; $n];

            fn solve_exact_homogeneous(&self, random_vector: &[T; $n]) -> Self::Output {
                let mut a = self.clone();
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
                let mut random_vector = random_vector.clone();
                random_vector = descale(&random_vector);
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

pub trait SolveHomogeneousExt<RandomVector> {
    type Output;

    fn solve_homogeneous(
        &self,
        random_vector: &RandomVector,
    ) -> Result<Self::Output, SolveHomogeneousError>;
}

macro_rules! solve_homogeneous_impl {
    ($n:expr) => {
        impl<T: Clone + Zero + Descale> SolveHomogeneousExt<[T; $n]> for [[T; $n]]
        where
            for<'a> &'a T: Neg<Output = T>,
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
            for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
            for<'a, 'b> &'a T: Div<&'b T, Output = T>,
            for<'a, 'b> &'a T: PartialOrd<&'b T>,
        {
            type Output = [T; $n];

            fn solve_homogeneous(
                &self,
                random_vector: &[T; $n],
            ) -> Result<Self::Output, SolveHomogeneousError> {
                if self.len() < $n - 1 {
                    return Err(SolveHomogeneousError::NotEnoughEquations);
                }
                let m: Tensor2<T, SolutionSpace<$n>, CoSpace<$n, SolutionSpace<$n>>, $n, $n> =
                    Tensor2::from_raw(
                        SolutionSpace,
                        CoSpace(SolutionSpace),
                        from_fn(|i| {
                            from_fn(|j| {
                                self.iter()
                                    .map(|a_k| &a_k[i])
                                    .zip(self.iter().map(|a_k| &a_k[j]))
                                    .fold(T::zero(), |acc, (a_ki, a_kj)| &acc + &(a_ki * a_kj))
                            })
                        }),
                    );
                let random_vector = Tensor1::from_raw(SolutionSpace, random_vector.clone());
                Ok(m.min_eigen_value_vector(&random_vector).raw.clone())
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

    #[test]
    fn test_solve_exact_homogeneous_3() {
        let a: [[f64; 3]; 2] = [[965., 880., 756.], [470., 332., 822.]];
        let random_vector: [f64; 3] = [1., 2., 3.];
        let null_vector = a.solve_exact_homogeneous(&random_vector);
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
    fn test_solve_exact_homogeneous_4() {
        let a: [[f64; 4]; 3] = [[965., 880., 756., 295.], [470., 332., 822., 748.], [529., 139., 729., 625.]];
        let random_vector: [f64; 4] = [1., 2., 3., 4.];
        let null_vector = a.solve_exact_homogeneous(&random_vector);
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
        let null_vector = a.solve_homogeneous(&random_vector).unwrap();
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
