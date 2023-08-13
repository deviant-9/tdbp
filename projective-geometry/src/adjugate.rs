use crate::array_utils::ArrayExt;
use crate::tensors::{CoSpace, Space, Tensor2};
use std::array::from_fn;
use std::ops::{Add, Mul, Neg, Sub};

impl<T: Clone, S0: Space<2>, S1: Space<2>> Tensor2<T, S0, CoSpace<2, S1>, 2, 2>
where
    for<'a> &'a T: Neg<Output = T>,
{
    #[inline]
    pub fn adjugate(&self) -> Tensor2<T, S1, CoSpace<2, S0>, 2, 2> {
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

macro_rules! adjugate_impl {
    ($n:expr) => {
        impl<T: Clone, S0: Space<$n>, S1: Space<$n>> Tensor2<T, S0, CoSpace<$n, S1>, $n, $n>
        where
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
            for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
            for<'a> &'a T: Neg<Output = T>,
        {
            pub fn adjugate(&self) -> Tensor2<T, S1, CoSpace<$n, S0>, $n, $n> {
                Tensor2::from_raw(
                    self.s1.0.clone(),
                    CoSpace(self.s0.clone()),
                    from_fn(|i| {
                        from_fn(|j| {
                            let cofactor = self.raw.sub_matrix(j, i).det();
                            if ((i % 2) ^ (j % 2)) == 0 {
                                cofactor
                            } else {
                                -&cofactor
                            }
                        })
                    }),
                )
            }
        }
    };
}

adjugate_impl!(3);
adjugate_impl!(4);
adjugate_impl!(5);
adjugate_impl!(6);
adjugate_impl!(7);
adjugate_impl!(8);
adjugate_impl!(9);

trait DetExt {
    type Output;
    fn det(&self) -> Self::Output;
}

impl<T> DetExt for [[T; 2]; 2]
where
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
{
    type Output = T;

    #[inline]
    fn det(&self) -> Self::Output {
        &(&self[0][0] * &self[1][1]) - &(&self[0][1] * &self[1][0])
    }
}

macro_rules! det_ext_impl {
    ($n:expr) => {
        impl<T: Clone> DetExt for [[T; $n]; $n]
        where
            for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
            for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
            for<'a, 'b> &'a T: Add<&'b T, Output = T>,
        {
            type Output = T;

            #[inline]
            fn det(&self) -> Self::Output {
                let without_0_col: [[T; $n - 1]; $n] =
                    self.ref_map(|row| from_fn(|j| row[j + 1].clone()));
                let without_00: &[[T; $n - 1]; $n - 1] = without_0_col[1..].try_into().unwrap();
                let mut result: T = &self[0][0] * &without_00.det();
                for det_i in 1..self.len() {
                    let without_i0: [[T; $n - 1]; $n - 1] = from_fn(|i| {
                        let fixed_i = if i < det_i { i } else { i + 1 };
                        without_0_col[fixed_i].clone()
                    });
                    let term: T = &self[det_i][0] * &without_i0.det();
                    if det_i % 2 == 0 {
                        result = &result + &term;
                    } else {
                        result = &result - &term;
                    }
                }
                result
            }
        }
    };
}

det_ext_impl!(3);
det_ext_impl!(4);
det_ext_impl!(5);
det_ext_impl!(6);
det_ext_impl!(7);
det_ext_impl!(8);

trait SubMatrixExt {
    type Output;
    fn sub_matrix(&self, omit_i: usize, omit_j: usize) -> Self::Output;
}

macro_rules! sub_matrix_ext_impl {
    ($n0:expr, $n1:expr) => {
        impl<T: Clone> SubMatrixExt for [[T; $n1]; $n0] {
            type Output = [[T; $n1 - 1]; $n0 - 1];

            #[inline]
            fn sub_matrix(&self, omit_i: usize, omit_j: usize) -> Self::Output {
                assert!(omit_i < $n0);
                assert!(omit_j < $n1);
                from_fn(|i| {
                    let fixed_i = if i < omit_i { i } else { i + 1 };
                    from_fn(|j| {
                        let fixed_j = if j < omit_j { j } else { j + 1 };
                        self[fixed_i][fixed_j].clone()
                    })
                })
            }
        }
    };
}

sub_matrix_ext_impl!(3, 3);
sub_matrix_ext_impl!(4, 4);
sub_matrix_ext_impl!(5, 5);
sub_matrix_ext_impl!(6, 6);
sub_matrix_ext_impl!(7, 7);
sub_matrix_ext_impl!(8, 8);
sub_matrix_ext_impl!(9, 9);

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

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S0_5;
    impl Space<5> for S0_5 {}
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S1_5;
    impl Space<5> for S1_5 {}

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
    fn test_tensor2_2_adjugate() {
        let so_identity = Tensor2::from_raw(S0_2, CoSpace(S0_2), [[1., 0.], [0., 1.]]);
        let t = Tensor2::from_raw(S0_2, CoSpace(S1_2), [[97., 17.], [41., 37.]]);
        let adjugate = t.adjugate();
        assert_homogenous_eq!(
            t.contract_tensor2_10(&adjugate).raw.flatten(),
            so_identity.raw.flatten()
        );
    }

    #[test]
    fn test_tensor2_3_adjugate() {
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
        let adjugate = t.adjugate();
        assert_homogenous_eq!(
            t.contract_tensor2_10(&adjugate).raw.flatten(),
            so_identity.raw.flatten()
        );
    }

    #[test]
    fn test_tensor2_4_adjugate() {
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
        let adjugate = t.adjugate();
        assert_homogenous_eq!(
            t.contract_tensor2_10(&adjugate).raw.flatten(),
            so_identity.raw.flatten()
        );
    }

    #[test]
    fn test_tensor2_5_adjugate() {
        let so_identity = Tensor2::from_raw(
            S0_5,
            CoSpace(S0_5),
            [
                [1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 1., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 1.],
            ],
        );
        let t = Tensor2::from_raw(
            S0_5,
            CoSpace(S1_5),
            [
                [75., 87., 55., 32., 44.],
                [16., 25., 39., 36., 55.],
                [22., 72., 10., 63., 58.],
                [35., 84., 71., 26., 94.],
                [67., 58., 89., 97., 25.],
            ],
        );
        let adjugate = t.adjugate();
        assert_homogenous_eq!(
            t.contract_tensor2_10(&adjugate).raw.flatten(),
            so_identity.raw.flatten()
        );
    }
}
