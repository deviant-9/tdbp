use crate::scalar_traits::{descale_array, Descale, Zero};
use crate::tensors::{CoSpace, Space, Tensor1, Tensor2};
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait MaxEigenValueVectorSolver<T, const N: usize> {
    fn max_eigen_value_vector<S: Space<N>>(
        &self,
        tensor: &Tensor2<T, S, CoSpace<N, S>, N, N>,
    ) -> Tensor1<T, S, N>;
}

pub struct MaxEigenValueVectorSolverImpl<T, const N: usize> {
    random_vector: [T; N],
}

impl<T: Descale, const N: usize> MaxEigenValueVectorSolverImpl<T, N> {
    #[inline]
    pub fn new(random_vector: &[T; N]) -> Self {
        Self {
            random_vector: descale_array(random_vector),
        }
    }
}

impl<T: Clone + Zero + Descale, const N: usize> MaxEigenValueVectorSolver<T, N>
    for MaxEigenValueVectorSolverImpl<T, N>
where
    for<'a, 'b> &'a T: Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: Div<&'b T, Output = T>,
    for<'a, 'b> &'a T: PartialOrd<&'b T>,
{
    fn max_eigen_value_vector<S: Space<N>>(
        &self,
        tensor: &Tensor2<T, S, CoSpace<N, S>, N, N>,
    ) -> Tensor1<T, S, N> {
        assert_eq!(tensor.s0, tensor.s1.0);
        let m = tensor.descale();
        let random_vector = Tensor1::from_raw(tensor.s0.clone(), self.random_vector.clone());
        let result = m.contract_tensor1_10(&random_vector);
        let mut ratio = &dot_square(&result) / &dot_square(&random_vector);
        let mut result = result.descale();
        loop {
            let new_result = m.contract_tensor1_10(&result);
            let new_ratio = &dot_square(&new_result) / &dot_square(&result);
            result = new_result.descale();
            if &new_ratio <= &ratio {
                return result;
            }
            ratio = new_ratio;
        }
    }
}

#[inline]
fn dot_square<T: Clone + Zero, S: Space<N>, const N: usize>(v: &Tensor1<T, S, N>) -> T
where
    for<'a, 'b> &'a T: Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    dot_product(&v, &v)
}

#[inline]
fn dot_product<T: Clone + Zero, S: Space<N>, const N: usize>(
    lhs: &Tensor1<T, S, N>,
    rhs: &Tensor1<T, S, N>,
) -> T
where
    for<'a, 'b> &'a T: Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
{
    Tensor1::from_raw(CoSpace(lhs.s0.clone()), lhs.raw.clone()).contract_tensor1_00(rhs)
}

pub trait MinEigenValueVectorSolver<T, const N: usize> {
    fn min_eigen_value_vector<S: Space<N>>(
        &self,
        tensor: &Tensor2<T, S, CoSpace<N, S>, N, N>,
    ) -> Tensor1<T, S, N>;
}

pub struct MinEigenValueVectorSolverImpl<MaxSolver> {
    max_solver: MaxSolver,
}

impl<MaxSolver> MinEigenValueVectorSolverImpl<MaxSolver> {
    #[inline]
    pub fn new(max_solver: MaxSolver) -> Self {
        Self { max_solver }
    }
}

macro_rules! min_eigen_value_vector_impl {
    ($n:expr) => {
        impl<MaxSolver: MaxEigenValueVectorSolver<T, $n>, T: Clone + Descale>
            MinEigenValueVectorSolver<T, $n> for MinEigenValueVectorSolverImpl<MaxSolver>
        where
            for<'a> &'a T: Neg<Output = T>,
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
            for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
        {
            #[inline]
            fn min_eigen_value_vector<S: Space<$n>>(
                &self,
                tensor: &Tensor2<T, S, CoSpace<$n, S>, $n, $n>,
            ) -> Tensor1<T, S, $n> {
                assert_eq!(tensor.s0, tensor.s1.0);
                self.max_solver
                    .max_eigen_value_vector(&tensor.descale().adjugate())
            }
        }
    };
}

min_eigen_value_vector_impl!(2);
min_eigen_value_vector_impl!(3);
min_eigen_value_vector_impl!(4);
min_eigen_value_vector_impl!(5);
min_eigen_value_vector_impl!(6);
min_eigen_value_vector_impl!(7);
min_eigen_value_vector_impl!(8);
min_eigen_value_vector_impl!(9);

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S3;
    impl Space<3> for S3 {}

    #[test]
    fn test_max_eigen_value_vector() {
        let m = Tensor2::from_raw(
            S3,
            CoSpace(S3),
            [[539., 406., 602.], [406., 863., 949.], [602., 949., 604.]],
        );
        let random_vector = [13., 17., -20.];
        let max_solver = MaxEigenValueVectorSolverImpl::new(&random_vector);
        let eigen_vector = max_solver.max_eigen_value_vector(&m);
        let multiplied = m.contract_tensor1_10(&eigen_vector);
        let angle_cos = angle_cos(&eigen_vector.raw, &multiplied.raw);
        assert!(
            (angle_cos.abs() - 1.).abs() < 0.000000001,
            "cos(angle) = {}",
            angle_cos
        );
        let eigen_value = dot_product(&multiplied.raw, &eigen_vector.raw)
            / dot_product(&eigen_vector.raw, &eigen_vector.raw);
        assert!(
            (eigen_value - 2025.9460699086276).abs() < 0.000000001,
            "eigen_value = {}",
            eigen_value
        );
    }

    #[test]
    fn test_min_eigen_value_vector() {
        let m = Tensor2::from_raw(
            S3,
            CoSpace(S3),
            [[539., 406., 602.], [406., 863., 949.], [602., 949., 604.]],
        );
        let random_vector = [13., 17., -20.];
        let max_solver = MaxEigenValueVectorSolverImpl::new(&random_vector);
        let min_solver = MinEigenValueVectorSolverImpl::new(max_solver);
        let eigen_vector = min_solver.min_eigen_value_vector(&m);
        let eigen_vector_abs = abs(&eigen_vector.raw);
        assert!(eigen_vector_abs >= 1., "eigen vector is too short");
        assert!(
            eigen_vector_abs < eigen_vector.raw.len() as f64,
            "eigen vector is too long"
        );
        let multiplied = m.contract_tensor1_10(&eigen_vector);
        let angle_cos = angle_cos(&eigen_vector.raw, &multiplied.raw);
        assert!(
            (angle_cos.abs() - 1.).abs() < 0.000000001,
            "cos(angle) = {}",
            angle_cos
        );
        let eigen_value = dot_product(&multiplied.raw, &eigen_vector.raw)
            / dot_product(&eigen_vector.raw, &eigen_vector.raw);
        assert!(
            (eigen_value - 264.9193066718052).abs() < 0.000000001,
            "eigen_value = {}",
            eigen_value
        );
    }

    fn angle_cos(lhs: &[f64], rhs: &[f64]) -> f64 {
        assert_eq!(lhs.len(), rhs.len());
        dot_product(lhs, rhs) / (abs(lhs) * abs(rhs))
    }

    fn abs(v: &[f64]) -> f64 {
        dot_product(v, v).sqrt()
    }

    fn dot_product(lhs: &[f64], rhs: &[f64]) -> f64 {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum()
    }
}
