use crate::scalar_traits::Zero;
use crate::tensors::{CoSpace, Space, Tensor1, Tensor2, Tensor3};
use std::ops::{Add, Neg, Sub};

impl<T: Clone, S0: Space<2>> Tensor1<T, S0, 2>
where
    for<'a> &'a T: Neg<Output = T>,
{
    #[inline]
    pub fn co_levi_contract_0(&self) -> Tensor1<T, CoSpace<2, S0>, 2> {
        let raw = &self.raw;
        Tensor1::from_raw(CoSpace(self.s0.clone()), [raw[1].clone(), -&raw[0]])
    }
}

impl<T: Clone, S0: Space<2>> Tensor1<T, CoSpace<2, S0>, 2>
where
    for<'a> &'a T: Neg<Output = T>,
{
    #[inline]
    pub fn contra_levi_contract_0(&self) -> Tensor1<T, S0, 2> {
        let raw = &self.raw;
        Tensor1::from_raw(self.s0.0.clone(), [raw[1].clone(), -&raw[0]])
    }
}

impl<T: Clone + Zero, S0: Space<3>> Tensor1<T, S0, 3>
where
    for<'a> &'a T: Neg<Output = T>,
{
    #[inline]
    pub fn co_levi_contract_0(&self) -> Tensor2<T, CoSpace<3, S0>, CoSpace<3, S0>, 3, 3> {
        let raw = &self.raw;
        Tensor2::from_raw(
            CoSpace(self.s0.clone()),
            CoSpace(self.s0.clone()),
            [
                [T::zero(), raw[2].clone(), -&raw[1]],
                [-&raw[2], T::zero(), raw[0].clone()],
                [raw[1].clone(), -&raw[0], T::zero()],
            ],
        )
    }
}

impl<T: Clone + Zero, S0: Space<3>> Tensor1<T, CoSpace<3, S0>, 3>
where
    for<'a> &'a T: Neg<Output = T>,
{
    #[inline]
    pub fn contra_levi_contract_0(&self) -> Tensor2<T, S0, S0, 3, 3> {
        let raw = &self.raw;
        Tensor2::from_raw(
            self.s0.0.clone(),
            self.s0.0.clone(),
            [
                [T::zero(), raw[2].clone(), -&raw[1]],
                [-&raw[2], T::zero(), raw[0].clone()],
                [raw[1].clone(), -&raw[0], T::zero()],
            ],
        )
    }
}

impl<T: Clone + Zero, S0: Space<4>> Tensor1<T, S0, 4>
where
    for<'a> &'a T: Neg<Output = T>,
{
    #[inline]
    pub fn co_levi_contract_0(
        &self,
    ) -> Tensor3<T, CoSpace<4, S0>, CoSpace<4, S0>, CoSpace<4, S0>, 4, 4, 4> {
        let raw = &self.raw;
        Tensor3::from_raw(
            CoSpace(self.s0.clone()),
            CoSpace(self.s0.clone()),
            CoSpace(self.s0.clone()),
            [
                [
                    [T::zero(), T::zero(), T::zero(), T::zero()],
                    [T::zero(), T::zero(), raw[3].clone(), -&raw[2]],
                    [T::zero(), -&raw[3], T::zero(), raw[1].clone()],
                    [T::zero(), raw[2].clone(), -&raw[1], T::zero()],
                ],
                [
                    [T::zero(), T::zero(), -&raw[3], raw[2].clone()],
                    [T::zero(), T::zero(), T::zero(), T::zero()],
                    [raw[3].clone(), T::zero(), T::zero(), -&raw[0]],
                    [-&raw[2], T::zero(), raw[0].clone(), T::zero()],
                ],
                [
                    [T::zero(), raw[3].clone(), T::zero(), -&raw[1]],
                    [-&raw[3], T::zero(), T::zero(), raw[0].clone()],
                    [T::zero(), T::zero(), T::zero(), T::zero()],
                    [raw[1].clone(), -&raw[0], T::zero(), T::zero()],
                ],
                [
                    [T::zero(), -&raw[2], raw[1].clone(), T::zero()],
                    [raw[2].clone(), T::zero(), -&raw[0], T::zero()],
                    [-&raw[1], raw[0].clone(), T::zero(), T::zero()],
                    [T::zero(), T::zero(), T::zero(), T::zero()],
                ],
            ],
        )
    }
}

impl<T: Clone + Zero, S0: Space<4>> Tensor1<T, CoSpace<4, S0>, 4>
where
    for<'a> &'a T: Neg<Output = T>,
{
    #[inline]
    pub fn contra_levi_contract_0(&self) -> Tensor3<T, S0, S0, S0, 4, 4, 4> {
        let raw = &self.raw;
        Tensor3::from_raw(
            self.s0.0.clone(),
            self.s0.0.clone(),
            self.s0.0.clone(),
            [
                [
                    [T::zero(), T::zero(), T::zero(), T::zero()],
                    [T::zero(), T::zero(), raw[3].clone(), -&raw[2]],
                    [T::zero(), -&raw[3], T::zero(), raw[1].clone()],
                    [T::zero(), raw[2].clone(), -&raw[1], T::zero()],
                ],
                [
                    [T::zero(), T::zero(), -&raw[3], raw[2].clone()],
                    [T::zero(), T::zero(), T::zero(), T::zero()],
                    [raw[3].clone(), T::zero(), T::zero(), -&raw[0]],
                    [-&raw[2], T::zero(), raw[0].clone(), T::zero()],
                ],
                [
                    [T::zero(), raw[3].clone(), T::zero(), -&raw[1]],
                    [-&raw[3], T::zero(), T::zero(), raw[0].clone()],
                    [T::zero(), T::zero(), T::zero(), T::zero()],
                    [raw[1].clone(), -&raw[0], T::zero(), T::zero()],
                ],
                [
                    [T::zero(), -&raw[2], raw[1].clone(), T::zero()],
                    [raw[2].clone(), T::zero(), -&raw[0], T::zero()],
                    [-&raw[1], raw[0].clone(), T::zero(), T::zero()],
                    [T::zero(), T::zero(), T::zero(), T::zero()],
                ],
            ],
        )
    }
}

impl<T: Clone + Zero, S0: Space<3>> Tensor2<T, S0, S0, 3, 3>
where
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
{
    #[inline]
    pub fn co_levi_contract_01(&self) -> Tensor1<T, CoSpace<3, S0>, 3> {
        assert_eq!(self.s0, self.s1);
        let raw = &self.raw;
        Tensor1::from_raw(
            CoSpace(self.s0.clone()),
            [
                &raw[1][2] - &raw[2][1],
                &raw[2][0] - &raw[0][2],
                &raw[0][1] - &raw[1][0],
            ],
        )
    }
}

impl<T: Clone + Zero, S0: Space<3>> Tensor2<T, CoSpace<3, S0>, CoSpace<3, S0>, 3, 3>
where
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
{
    #[inline]
    pub fn contra_levi_contract_01(&self) -> Tensor1<T, S0, 3> {
        assert_eq!(self.s0, self.s1);
        let raw = &self.raw;
        Tensor1::from_raw(
            self.s0.0.clone(),
            [
                &raw[1][2] - &raw[2][1],
                &raw[2][0] - &raw[0][2],
                &raw[0][1] - &raw[1][0],
            ],
        )
    }
}

impl<T: Clone + Zero, S0: Space<4>> Tensor2<T, S0, S0, 4, 4>
where
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
{
    #[inline]
    pub fn co_levi_contract_01(&self) -> Tensor2<T, CoSpace<4, S0>, CoSpace<4, S0>, 4, 4> {
        assert_eq!(self.s0, self.s1);
        let raw = &self.raw;
        Tensor2::from_raw(
            CoSpace(self.s0.clone()),
            CoSpace(self.s0.clone()),
            [
                [
                    T::zero(),
                    &raw[2][3] - &raw[3][2],
                    &raw[3][1] - &raw[1][3],
                    &raw[1][2] - &raw[2][1],
                ],
                [
                    &raw[3][2] - &raw[2][3],
                    T::zero(),
                    &raw[0][3] - &raw[3][0],
                    &raw[2][0] - &raw[0][2],
                ],
                [
                    &raw[1][3] - &raw[3][1],
                    &raw[3][0] - &raw[0][3],
                    T::zero(),
                    &raw[0][1] - &raw[1][0],
                ],
                [
                    &raw[2][1] - &raw[1][2],
                    &raw[0][2] - &raw[2][0],
                    &raw[1][0] - &raw[0][1],
                    T::zero(),
                ],
            ],
        )
    }
}

impl<T: Clone + Zero, S0: Space<4>> Tensor2<T, CoSpace<4, S0>, CoSpace<4, S0>, 4, 4>
where
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
{
    #[inline]
    pub fn contra_levi_contract_01(&self) -> Tensor2<T, S0, S0, 4, 4> {
        assert_eq!(self.s0, self.s1);
        let raw = &self.raw;
        Tensor2::from_raw(
            self.s0.0.clone(),
            self.s0.0.clone(),
            [
                [
                    T::zero(),
                    &raw[2][3] - &raw[3][2],
                    &raw[3][1] - &raw[1][3],
                    &raw[1][2] - &raw[2][1],
                ],
                [
                    &raw[3][2] - &raw[2][3],
                    T::zero(),
                    &raw[0][3] - &raw[3][0],
                    &raw[2][0] - &raw[0][2],
                ],
                [
                    &raw[1][3] - &raw[3][1],
                    &raw[3][0] - &raw[0][3],
                    T::zero(),
                    &raw[0][1] - &raw[1][0],
                ],
                [
                    &raw[2][1] - &raw[1][2],
                    &raw[0][2] - &raw[2][0],
                    &raw[1][0] - &raw[0][1],
                    T::zero(),
                ],
            ],
        )
    }
}

impl<T: Clone + Zero, S0: Space<4>> Tensor3<T, S0, S0, S0, 4, 4, 4>
where
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Add<&'b T, Output = T>,
{
    #[inline]
    pub fn co_levi_contract_012(&self) -> Tensor1<T, CoSpace<4, S0>, 4> {
        assert_eq!(self.s0, self.s1);
        assert_eq!(self.s1, self.s2);
        let raw = &self.raw;
        Tensor1::from_raw(
            CoSpace(self.s0.clone()),
            [
                &(&(&raw[1][2][3] + &raw[2][3][1]) + &raw[3][1][2])
                    - &(&(&raw[1][3][2] + &raw[2][1][3]) + &raw[3][2][1]),
                &(&(&raw[0][3][2] + &raw[2][0][3]) + &raw[3][2][0])
                    - &(&(&raw[0][2][3] + &raw[2][3][0]) + &raw[3][0][2]),
                &(&(&raw[0][1][3] + &raw[1][3][0]) + &raw[3][0][1])
                    - &(&(&raw[0][3][1] + &raw[1][0][3]) + &raw[3][1][0]),
                &(&(&raw[0][2][1] + &raw[1][0][2]) + &raw[2][1][0])
                    - &(&(&raw[0][1][2] + &raw[1][2][0]) + &raw[2][0][1]),
            ],
        )
    }
}

impl<T: Clone + Zero, S0: Space<4>>
    Tensor3<T, CoSpace<4, S0>, CoSpace<4, S0>, CoSpace<4, S0>, 4, 4, 4>
where
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Add<&'b T, Output = T>,
{
    #[inline]
    pub fn contra_levi_contract_012(&self) -> Tensor1<T, S0, 4> {
        assert_eq!(self.s0, self.s1);
        assert_eq!(self.s1, self.s2);
        let raw = &self.raw;
        Tensor1::from_raw(
            self.s0.0.clone(),
            [
                &(&(&raw[1][2][3] + &raw[2][3][1]) + &raw[3][1][2])
                    - &(&(&raw[1][3][2] + &raw[2][1][3]) + &raw[3][2][1]),
                &(&(&raw[0][3][2] + &raw[2][0][3]) + &raw[3][2][0])
                    - &(&(&raw[0][2][3] + &raw[2][3][0]) + &raw[3][0][2]),
                &(&(&raw[0][1][3] + &raw[1][3][0]) + &raw[3][0][1])
                    - &(&(&raw[0][3][1] + &raw[1][0][3]) + &raw[3][1][0]),
                &(&(&raw[0][2][1] + &raw[1][0][2]) + &raw[2][1][0])
                    - &(&(&raw[0][1][2] + &raw[1][2][0]) + &raw[2][0][1]),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensors::Tensor1;
    use std::array::from_fn;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S0_2D;
    impl Space<2> for S0_2D {}
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S0_3D;
    impl Space<3> for S0_3D {}
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S0_4D;
    impl Space<4> for S0_4D {}

    #[test]
    fn test_tensor1_2d_co_levi_contract_0() {
        let coords = [1, 2];
        assert_eq!(tensor1_2d_levi_contract_0(&coords), [2, -1]);
        assert_eq!(
            Tensor1::from_raw(S0_2D, coords.clone()).co_levi_contract_0(),
            Tensor1::from_raw(CoSpace(S0_2D), tensor1_2d_levi_contract_0(&coords))
        );
    }

    #[test]
    fn test_tensor1_2d_contra_levi_contract_0() {
        let coords = [1, 2];
        assert_eq!(
            Tensor1::from_raw(CoSpace(S0_2D), coords.clone()).contra_levi_contract_0(),
            Tensor1::from_raw(S0_2D, tensor1_2d_levi_contract_0(&coords))
        );
    }

    #[test]
    fn test_tensor1_3d_co_levi_contract_0() {
        let coords = [1, 2, 3];
        assert_eq!(
            tensor1_3d_levi_contract_0(&coords),
            [[0, 3, -2], [-3, 0, 1], [2, -1, 0]]
        );
        assert_eq!(
            Tensor1::from_raw(S0_3D, coords.clone()).co_levi_contract_0(),
            Tensor2::from_raw(
                CoSpace(S0_3D),
                CoSpace(S0_3D),
                tensor1_3d_levi_contract_0(&coords)
            )
        );
    }

    #[test]
    fn test_tensor1_3d_contra_levi_contract_0() {
        let coords = [1, 2, 3];
        assert_eq!(
            Tensor1::from_raw(CoSpace(S0_3D), coords.clone()).contra_levi_contract_0(),
            Tensor2::from_raw(S0_3D, S0_3D, tensor1_3d_levi_contract_0(&coords))
        );
    }

    #[test]
    fn test_tensor1_4d_co_levi_contract_0() {
        let coords = [1, 2, 3, 4];
        assert_eq!(
            Tensor1::from_raw(S0_4D, coords.clone()).co_levi_contract_0(),
            Tensor3::from_raw(
                CoSpace(S0_4D),
                CoSpace(S0_4D),
                CoSpace(S0_4D),
                tensor1_4d_levi_contract_0(&coords)
            )
        );
    }

    #[test]
    fn test_tensor1_4d_contra_levi_contract_0() {
        let coords = [1, 2, 3, 4];
        assert_eq!(
            Tensor1::from_raw(CoSpace(S0_4D), coords.clone()).contra_levi_contract_0(),
            Tensor3::from_raw(S0_4D, S0_4D, S0_4D, tensor1_4d_levi_contract_0(&coords))
        );
    }

    #[test]
    fn test_tensor2_3d_co_levi_contract_01() {
        let coords = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(
            Tensor2::from_raw(S0_3D, S0_3D, coords.clone()).co_levi_contract_01(),
            Tensor1::from_raw(CoSpace(S0_3D), tensor2_3d_levi_contract_01(&coords))
        );
    }

    #[test]
    fn test_tensor2_3d_contra_levi_contract_01() {
        let coords = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(
            Tensor2::from_raw(CoSpace(S0_3D), CoSpace(S0_3D), coords.clone())
                .contra_levi_contract_01(),
            Tensor1::from_raw(S0_3D, tensor2_3d_levi_contract_01(&coords))
        );
    }

    #[test]
    fn test_tensor2_4d_co_levi_contract_01() {
        let coords = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ];
        assert_eq!(
            Tensor2::from_raw(S0_4D, S0_4D, coords.clone()).co_levi_contract_01(),
            Tensor2::from_raw(
                CoSpace(S0_4D),
                CoSpace(S0_4D),
                tensor2_4d_levi_contract_01(&coords)
            )
        );
    }

    #[test]
    fn test_tensor2_4d_contra_levi_contract_01() {
        let coords = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ];
        assert_eq!(
            Tensor2::from_raw(CoSpace(S0_4D), CoSpace(S0_4D), coords.clone())
                .contra_levi_contract_01(),
            Tensor2::from_raw(S0_4D, S0_4D, tensor2_4d_levi_contract_01(&coords))
        );
    }

    #[test]
    fn test_tensor3_4d_co_levi_contract_012() {
        let coords = [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            [
                [17, 18, 19, 20],
                [21, 22, 23, 24],
                [25, 26, 27, 28],
                [29, 30, 31, 32],
            ],
            [
                [33, 34, 35, 36],
                [37, 38, 39, 40],
                [41, 42, 43, 44],
                [45, 46, 47, 48],
            ],
            [
                [49, 50, 51, 52],
                [53, 54, 55, 56],
                [57, 58, 59, 60],
                [61, 62, 63, 64],
            ],
        ];
        assert_eq!(
            Tensor3::from_raw(S0_4D, S0_4D, S0_4D, coords.clone()).co_levi_contract_012(),
            Tensor1::from_raw(CoSpace(S0_4D), tensor3_4d_levi_contract_012(&coords))
        );
    }

    #[test]
    fn test_tensor3_4d_contra_levi_contract_012() {
        let coords = [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            [
                [17, 18, 19, 20],
                [21, 22, 23, 24],
                [25, 26, 27, 28],
                [29, 30, 31, 32],
            ],
            [
                [33, 34, 35, 36],
                [37, 38, 39, 40],
                [41, 42, 43, 44],
                [45, 46, 47, 48],
            ],
            [
                [49, 50, 51, 52],
                [53, 54, 55, 56],
                [57, 58, 59, 60],
                [61, 62, 63, 64],
            ],
        ];
        assert_eq!(
            Tensor3::from_raw(
                CoSpace(S0_4D),
                CoSpace(S0_4D),
                CoSpace(S0_4D),
                coords.clone()
            )
            .contra_levi_contract_012(),
            Tensor1::from_raw(S0_4D, tensor3_4d_levi_contract_012(&coords))
        );
    }

    fn tensor1_2d_levi_contract_0(t: &[i32; 2]) -> [i32; 2] {
        from_fn(|i0| (0..t.len()).map(|i1| e(&[i0, i1]) * t[i1]).sum())
    }

    fn tensor1_3d_levi_contract_0(t: &[i32; 3]) -> [[i32; 3]; 3] {
        from_fn(|i0| from_fn(|i1| (0..t.len()).map(|i2| e(&[i0, i1, i2]) * t[i2]).sum()))
    }

    fn tensor1_4d_levi_contract_0(t: &[i32; 4]) -> [[[i32; 4]; 4]; 4] {
        from_fn(|i0| {
            from_fn(|i1| from_fn(|i2| (0..t.len()).map(|i3| e(&[i0, i1, i2, i3]) * t[i3]).sum()))
        })
    }

    fn tensor2_3d_levi_contract_01(t: &[[i32; 3]; 3]) -> [i32; 3] {
        from_fn(|i0| {
            (0..t.len())
                .map(|i1| {
                    (0..t[0].len())
                        .map(|i2| e(&[i0, i1, i2]) * t[i1][i2])
                        .sum::<i32>()
                })
                .sum()
        })
    }

    fn tensor2_4d_levi_contract_01(t: &[[i32; 4]; 4]) -> [[i32; 4]; 4] {
        from_fn(|i0| {
            from_fn(|i1| {
                (0..t.len())
                    .map(|i2| {
                        (0..t[0].len())
                            .map(|i3| e(&[i0, i1, i2, i3]) * t[i2][i3])
                            .sum::<i32>()
                    })
                    .sum()
            })
        })
    }

    fn tensor3_4d_levi_contract_012(t: &[[[i32; 4]; 4]; 4]) -> [i32; 4] {
        from_fn(|i0| {
            (0..t.len())
                .map(|i1| {
                    (0..t[0].len())
                        .map(|i2| {
                            (0..t[0][0].len())
                                .map(|i3| e(&[i0, i1, i2, i3]) * t[i1][i2][i3])
                                .sum::<i32>()
                        })
                        .sum::<i32>()
                })
                .sum()
        })
    }

    fn e(indices: &[usize]) -> i32 {
        let mut inversions = 0u32;
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                if indices[i] == indices[j] {
                    return 0;
                } else if indices[i] > indices[j] {
                    inversions += 1;
                }
            }
        }
        (-1i32).pow(inversions)
    }
}
