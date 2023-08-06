use crate::tensors::{CoSpace, Space, Tensor2};
use std::ops::{Add, Mul, Neg, Sub};

impl<T: Clone, S0: Space<2>, S1: Space<2>> Tensor2<T, S0, CoSpace<2, S1>, 2, 2>
where
    for<'a> &'a T: Neg<Output = T>,
{
    #[inline]
    pub fn up_to_scale_inverse(&self) -> Tensor2<T, S1, CoSpace<2, S0>, 2, 2> {
        let raw = &self.raw;
        Tensor2::from_raw(
            self.s1.0.clone(),
            CoSpace(self.s0.clone()),
            [
                [raw[1][1].clone(), -&raw[0][1]],
                [-&raw[1][0], raw[0][0].clone()],
            ],
        )
    }
}

impl<T: Clone, S0: Space<3>, S1: Space<3>> Tensor2<T, S0, CoSpace<3, S1>, 3, 3>
where
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
{
    pub fn up_to_scale_inverse(&self) -> Tensor2<T, S1, CoSpace<3, S0>, 3, 3> {
        let raw = &self.raw;
        let minor = |i0: usize, i1: usize, j0: usize, j1: usize| {
            &(&raw[i0][j0] * &raw[i1][j1]) - &(&raw[i0][j1] * &raw[i1][j0])
        };
        Tensor2::from_raw(
            self.s1.0.clone(),
            CoSpace(self.s0.clone()),
            [
                [minor(1, 2, 1, 2), minor(2, 0, 1, 2), minor(0, 1, 1, 2)],
                [minor(1, 2, 2, 0), minor(2, 0, 2, 0), minor(0, 1, 2, 0)],
                [minor(1, 2, 0, 1), minor(2, 0, 0, 1), minor(0, 1, 0, 1)],
            ],
        )
    }
}

impl<T: Clone, S0: Space<4>, S1: Space<4>> Tensor2<T, S0, CoSpace<4, S1>, 4, 4>
where
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Add<&'b T, Output = T>,
{
    pub fn up_to_scale_inverse(&self) -> Tensor2<T, S1, CoSpace<4, S0>, 4, 4> {
        let raw = &self.raw;
        let minor2 = |i0: usize, i1: usize, j0: usize, j1: usize| {
            &(&raw[i0][j0] * &raw[i1][j1]) - &(&raw[i0][j1] * &raw[i1][j0])
        };
        let minor = |i0: usize, i1: usize, i2: usize, j0: usize, j1: usize, j2: usize| {
            &(&(&raw[i0][j0] * &minor2(i1, i2, j1, j2)) + &(&raw[i0][j1] * &minor2(i1, i2, j2, j0)))
                + &(&raw[i0][j2] * &minor2(i1, i2, j0, j1))
        };
        Tensor2::from_raw(
            self.s1.0.clone(),
            CoSpace(self.s0.clone()),
            [
                [
                    minor(1, 2, 3, 1, 2, 3),
                    minor(0, 3, 2, 1, 2, 3),
                    minor(3, 0, 1, 1, 2, 3),
                    minor(2, 1, 0, 1, 2, 3),
                ],
                [
                    minor(1, 2, 3, 0, 3, 2),
                    minor(0, 3, 2, 0, 3, 2),
                    minor(3, 0, 1, 0, 3, 2),
                    minor(2, 1, 0, 0, 3, 2),
                ],
                [
                    minor(1, 2, 3, 3, 0, 1),
                    minor(0, 3, 2, 3, 0, 1),
                    minor(3, 0, 1, 3, 0, 1),
                    minor(2, 1, 0, 3, 0, 1),
                ],
                [
                    minor(1, 2, 3, 2, 1, 0),
                    minor(0, 3, 2, 2, 1, 0),
                    minor(3, 0, 1, 2, 1, 0),
                    minor(2, 1, 0, 2, 1, 0),
                ],
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projective_primitives::T;
    use crate::scalar_traits::Zero;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S0_2;
    impl Space<2> for S0_2 {}
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S1_2;
    impl Space<2> for S1_2 {}

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S0_3;
    impl Space<3> for S0_3 {}
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S1_3;
    impl Space<3> for S1_3 {}

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S0_4;
    impl Space<4> for S0_4 {}
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S1_4;
    impl Space<4> for S1_4 {}

    macro_rules! assert_homogenous_eq {
        ($lhs:expr, $rhs:expr) => {
            match (&$lhs, &$rhs) {
                (lhs, rhs) => {
                    assert!(homogenous_vectors_equal(lhs, rhs), "{:?} != {:?}", lhs, rhs);
                }
            }
        };
    }

    fn homogenous_vectors_equal(v0: &[T], v1: &[T]) -> bool {
        assert_eq!(
            v0.len(),
            v1.len(),
            "homogenous vectors have different lengths"
        );
        let v0_max = v0.iter().fold(
            T::zero(),
            |acc, x| if x.abs() > acc.abs() { *x } else { acc },
        );
        assert_ne!(v0_max, 0., "first homogenous vector is zero vector");
        let v1_max = v1.iter().fold(
            T::zero(),
            |acc, x| if x.abs() > acc.abs() { *x } else { acc },
        );
        assert_ne!(v1_max, 0., "second homogenous vector is zero vector");
        let v0_fixed: Vec<T> = v0.iter().map(|x| x * v1_max).collect();
        let v1_fixed: Vec<T> = v1.iter().map(|x| x * v0_max).collect();
        v0_fixed == v1_fixed
    }

    #[test]
    fn test_tensor2_2_inverse_up_to_scale() {
        let so_identity = Tensor2::from_raw(S0_2, CoSpace(S0_2), [[1., 0.], [0., 1.]]);
        let t = Tensor2::from_raw(S0_2, CoSpace(S1_2), [[97., 17.], [41., 37.]]);
        let inverse = t.up_to_scale_inverse();
        assert_homogenous_eq!(
            t.contract_tensor2_10(&inverse).raw.flatten(),
            so_identity.raw.flatten()
        );
    }

    #[test]
    fn test_tensor2_3_inverse_up_to_scale() {
        let so_identity = Tensor2::from_raw(
            S0_3,
            CoSpace(S0_3),
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        );
        let t = Tensor2::from_raw(
            S0_3,
            CoSpace(S1_3),
            [[85., 16., 96.], [19., 64., 13.], [16., 42., 67.]],
        );
        let inverse = t.up_to_scale_inverse();
        assert_homogenous_eq!(
            t.contract_tensor2_10(&inverse).raw.flatten(),
            so_identity.raw.flatten()
        );
    }

    #[test]
    fn test_tensor2_4_inverse_up_to_scale() {
        let so_identity = Tensor2::from_raw(
            S0_4,
            CoSpace(S0_4),
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
        );
        let t = Tensor2::from_raw(
            S0_4,
            CoSpace(S1_4),
            [
                [75., 87., 55., 32.],
                [16., 25., 39., 36.],
                [22., 72., 10., 63.],
                [35., 84., 71., 26.],
            ],
        );
        let inverse = t.up_to_scale_inverse();
        assert_homogenous_eq!(
            t.contract_tensor2_10(&inverse).raw.flatten(),
            so_identity.raw.flatten()
        );
    }
}
