use crate::scalar_traits::{Descale, Zero};
use crate::tensors::{CoSpace, Space, Tensor1, Tensor2};
use std::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! eigen_vector_impl {
    ($n:expr) => {
        impl<T: Clone + Zero + Descale, S: Space<$n>> Tensor2<T, S, CoSpace<$n, S>, $n, $n>
        where
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
            for<'a, 'b> &'a T: Div<&'b T, Output = T>,
            for<'a, 'b> &'a T: PartialOrd<&'b T>,
        {
            pub fn max_eigen_value_vector(
                &self,
                random_vector: &Tensor1<T, S, $n>,
            ) -> Tensor1<T, S, $n> {
                assert_eq!(self.s0, self.s1.0);
                let m = self.descale();
                let random_vector = random_vector.descale();
                let result = m.contract_tensor1_10(&random_vector);
                let mut ratio = &Self::dot_square(&result) / &Self::dot_square(&random_vector);
                let mut result = result.descale();
                loop {
                    let new_result = m.contract_tensor1_10(&result);
                    let new_ratio = &Self::dot_square(&new_result) / &Self::dot_square(&result);
                    result = new_result.descale();
                    if &new_ratio <= &ratio {
                        return result;
                    }
                    ratio = new_ratio;
                }
            }

            #[inline]
            fn dot_square(v: &Tensor1<T, S, $n>) -> T {
                Self::dot_product(&v, &v)
            }

            #[inline]
            fn dot_product(lhs: &Tensor1<T, S, $n>, rhs: &Tensor1<T, S, $n>) -> T {
                Tensor1::from_raw(CoSpace(lhs.s0.clone()), lhs.raw.clone()).contract_tensor1_00(rhs)
            }

            #[inline]
            pub fn min_eigen_value_vector(
                &self,
                random_vector: &Tensor1<T, S, $n>,
            ) -> Tensor1<T, S, $n>
            where
                for<'a> &'a T: Neg<Output = T>,
                for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            {
                assert_eq!(self.s0, self.s1.0);
                self.descale()
                    .adjugate()
                    .max_eigen_value_vector(random_vector)
            }
        }
    };
}

eigen_vector_impl!(2);
eigen_vector_impl!(3);
eigen_vector_impl!(4);
eigen_vector_impl!(5);
eigen_vector_impl!(6);
eigen_vector_impl!(7);
eigen_vector_impl!(8);
eigen_vector_impl!(9);

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S3;
    impl Space<3> for S3 {}

    #[test]
    fn test_max_eigen_value_vector() {
        let random_vector = Tensor1::from_raw(S3, [13., 17., -20.]);
        let m = Tensor2::from_raw(
            S3,
            CoSpace(S3),
            [[539., 406., 602.], [406., 863., 949.], [602., 949., 604.]],
        );
        let eigen_vector = m.max_eigen_value_vector(&random_vector);
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
        let random_vector = Tensor1::from_raw(S3, [13., 17., -20.]);
        let m = Tensor2::from_raw(
            S3,
            CoSpace(S3),
            [[539., 406., 602.], [406., 863., 949.], [602., 949., 604.]],
        );
        let eigen_vector = m.min_eigen_value_vector(&random_vector);
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
