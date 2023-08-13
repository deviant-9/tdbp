use crate::array_utils::ArrayExt;
use crate::scalar_traits::{Descale, ScalarAdd, ScalarNeg, ScalarSub, Zero};
use std::array::from_fn;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Mul, Neg, Sub};

// Types implemented this trait are intended to be zero-sized
pub trait Space<const N: usize>: Clone + Debug + Eq {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoSpace<const N: usize, S: Space<N>>(pub S);

impl<const N: usize, S: Space<N>> Space<N> for CoSpace<N, S> {}

pub trait DualSpace<Rhs> {
    fn check_duality(&self, rhs: &Rhs);
}

impl<const N: usize, S: Space<N>> DualSpace<S> for CoSpace<N, S> {
    #[inline]
    fn check_duality(&self, rhs: &S) {
        assert_eq!(&self.0, rhs);
    }
}

impl<const N: usize, S: Space<N>> DualSpace<CoSpace<N, S>> for S {
    #[inline]
    fn check_duality(&self, rhs: &CoSpace<N, S>) {
        assert_eq!(self, &rhs.0);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor1<T, S0: Space<N0>, const N0: usize> {
    pub s0: S0,
    pub raw: [T; N0],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor2<T, S0: Space<N0>, S1: Space<N1>, const N0: usize, const N1: usize> {
    pub s0: S0,
    pub s1: S1,
    pub raw: [[T; N1]; N0],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor3<
    T,
    S0: Space<N0>,
    S1: Space<N1>,
    S2: Space<N2>,
    const N0: usize,
    const N1: usize,
    const N2: usize,
> {
    pub s0: S0,
    pub s1: S1,
    pub s2: S2,
    pub raw: [[[T; N2]; N1]; N0],
}

impl<T, S0: Space<N0>, const N0: usize> Tensor1<T, S0, N0> {
    #[inline]
    pub fn from_raw(s0: S0, raw: [T; N0]) -> Self {
        Self { s0, raw }
    }

    #[inline]
    pub fn zero(s0: S0) -> Self
    where
        T: Zero,
    {
        Self::from_raw(s0, from_fn(|_| T::zero()))
    }

    #[inline]
    pub fn descale(&self) -> Self
    where
        T: Descale,
    {
        let factor = T::descaling_factor(self.raw.iter());
        Tensor1::from_raw(self.s0.clone(), self.raw.ref_map(|x| x.descale(&factor)))
    }

    #[inline]
    pub fn scale<'a, Factor: Copy>(
        &'a self,
        factor: Factor,
    ) -> Tensor1<<&'a T as Mul<Factor>>::Output, S0, N0>
    where
        &'a T: Mul<Factor>,
    {
        Tensor1::from_raw(self.s0.clone(), self.raw.scale1(factor))
    }

    #[inline]
    pub fn mul_tensor1<'l, 'r, RhsT, RhsS0: Space<RHS_N0>, const RHS_N0: usize>(
        &'l self,
        rhs: &'r Tensor1<RhsT, RhsS0, RHS_N0>,
    ) -> Tensor2<<&'r RhsT as Mul<&'l T>>::Output, S0, RhsS0, N0, RHS_N0>
    where
        &'r RhsT: Mul<&'l T>,
    {
        Tensor2::from_raw(
            self.s0.clone(),
            rhs.s0.clone(),
            self.raw.ref_map(|l| rhs.raw.scale1(l)),
        )
    }

    #[inline]
    pub fn mul_tensor2<
        'l,
        'r,
        RhsT,
        RhsS0: Space<RHS_N0>,
        RhsS1: Space<RHS_N1>,
        const RHS_N0: usize,
        const RHS_N1: usize,
    >(
        &'l self,
        rhs: &'r Tensor2<RhsT, RhsS0, RhsS1, RHS_N0, RHS_N1>,
    ) -> Tensor3<<&'r RhsT as Mul<&'l T>>::Output, S0, RhsS0, RhsS1, N0, RHS_N0, RHS_N1>
    where
        &'r RhsT: Mul<&'l T>,
    {
        Tensor3::from_raw(
            self.s0.clone(),
            rhs.s0.clone(),
            rhs.s1.clone(),
            self.raw.ref_map(|l| rhs.raw.scale2(l)),
        )
    }

    #[inline]
    pub fn contract_tensor1_00<'l, 'r, ResultT: Zero, RhsT, RhsS0: Space<N0> + DualSpace<S0>>(
        &'l self,
        rhs: &'r Tensor1<RhsT, RhsS0, N0>,
    ) -> ResultT
    where
        &'r RhsT: Mul<&'l T>,
        for<'a, 'b> &'a ResultT: Add<&'b <&'r RhsT as Mul<&'l T>>::Output, Output = ResultT>,
    {
        rhs.s0.check_duality(&self.s0);
        self.raw
            .array_zip_iter(&rhs.raw)
            .map(|(l, r)| r * l)
            .fold(ResultT::zero(), |s, x| &s + &x)
    }

    #[inline]
    pub fn contract_tensor2_00<
        'l,
        'r,
        ResultT: Zero,
        RhsT,
        RhsS0: Space<N0> + DualSpace<S0>,
        RhsS1: Space<RHS_N1>,
        const RHS_N1: usize,
    >(
        &'l self,
        rhs: &'r Tensor2<RhsT, RhsS0, RhsS1, N0, RHS_N1>,
    ) -> Tensor1<ResultT, RhsS1, RHS_N1>
    where
        &'r RhsT: Mul<&'l T>,
        for<'a, 'b> &'a ResultT: Add<&'b <&'r RhsT as Mul<&'l T>>::Output, Output = ResultT>,
    {
        rhs.s0.check_duality(&self.s0);
        Tensor1::from_raw(
            rhs.s1.clone(),
            self.raw
                .array_zip_iter(&rhs.raw)
                .map(|(l, rv)| rv.scale1(l))
                .arrays1_sum(),
        )
    }

    #[inline]
    pub fn contract_tensor3_00<
        'l,
        'r,
        ResultT: Zero,
        RhsT,
        RhsS0: Space<N0> + DualSpace<S0>,
        RhsS1: Space<RHS_N1>,
        RhsS2: Space<RHS_N2>,
        const RHS_N1: usize,
        const RHS_N2: usize,
    >(
        &'l self,
        rhs: &'r Tensor3<RhsT, RhsS0, RhsS1, RhsS2, N0, RHS_N1, RHS_N2>,
    ) -> Tensor2<ResultT, RhsS1, RhsS2, RHS_N1, RHS_N2>
    where
        &'r RhsT: Mul<&'l T>,
        for<'a, 'b> &'a ResultT: Add<&'b <&'r RhsT as Mul<&'l T>>::Output, Output = ResultT>,
    {
        rhs.s0.check_duality(&self.s0);
        Tensor2::from_raw(
            rhs.s1.clone(),
            rhs.s2.clone(),
            self.raw
                .array_zip_iter(&rhs.raw)
                .map(|(l, rvv)| rvv.scale2(l))
                .arrays2_sum(),
        )
    }
}

impl<T, S0: Space<N0>, S1: Space<N1>, const N0: usize, const N1: usize> Tensor2<T, S0, S1, N0, N1> {
    #[inline]
    pub fn from_raw(s0: S0, s1: S1, raw: [[T; N1]; N0]) -> Self {
        Self { s0, s1, raw }
    }

    #[inline]
    pub fn sub_tensors(&self) -> [Tensor1<T, S1, N1>; N0]
    where
        T: Clone,
    {
        self.raw
            .ref_map(|x| Tensor1::from_raw(self.s1.clone(), x.clone()))
    }

    #[inline]
    pub fn zero(s0: S0, s1: S1) -> Self
    where
        T: Zero,
    {
        Self::from_raw(s0, s1, from_fn(|_| from_fn(|_| T::zero())))
    }

    #[inline]
    pub fn descale(&self) -> Self
    where
        T: Descale,
    {
        let factor = T::descaling_factor(self.raw.flatten().iter());
        Tensor2::from_raw(
            self.s0.clone(),
            self.s1.clone(),
            self.raw.ref_map(|xv| xv.ref_map(|x| x.descale(&factor))),
        )
    }

    #[inline]
    pub fn scale<'a, Factor: Copy>(
        &'a self,
        factor: Factor,
    ) -> Tensor2<<&'a T as Mul<Factor>>::Output, S0, S1, N0, N1>
    where
        &'a T: Mul<Factor>,
    {
        Tensor2::from_raw(self.s0.clone(), self.s1.clone(), self.raw.scale2(factor))
    }

    #[inline]
    pub fn swap10(&self) -> Tensor2<T, S1, S0, N1, N0>
    where
        T: Clone,
    {
        Tensor2::from_raw(
            self.s1.clone(),
            self.s0.clone(),
            from_fn(|i1| from_fn(|i0| self.raw[i0][i1].clone())),
        )
    }

    #[inline]
    pub fn mul_tensor1<'l, 'r, RhsT, RhsS0: Space<RHS_N0>, const RHS_N0: usize>(
        &'l self,
        rhs: &'r Tensor1<RhsT, RhsS0, RHS_N0>,
    ) -> Tensor3<<&'r RhsT as Mul<&'l T>>::Output, S0, S1, RhsS0, N0, N1, RHS_N0>
    where
        &'r RhsT: Mul<&'l T>,
    {
        Tensor3::from_raw(
            self.s0.clone(),
            self.s1.clone(),
            rhs.s0.clone(),
            self.raw.ref_map(|lv| lv.ref_map(|l| rhs.raw.scale1(l))),
        )
    }

    #[inline]
    pub fn contract_tensor1_10<'l, 'r, ResultT: Zero, RhsT, RhsS0: Space<N1> + DualSpace<S1>>(
        &'l self,
        rhs: &'r Tensor1<RhsT, RhsS0, N1>,
    ) -> Tensor1<ResultT, S0, N0>
    where
        &'r RhsT: Mul<&'l T>,
        for<'a, 'b> &'a ResultT: Add<&'b <&'r RhsT as Mul<&'l T>>::Output, Output = ResultT>,
    {
        rhs.s0.check_duality(&self.s1);
        Tensor1::from_raw(
            self.s0.clone(),
            self.raw.ref_map(|lv| {
                lv.array_zip_iter(&rhs.raw)
                    .map(|(l, r)| r * l)
                    .fold(ResultT::zero(), |s, x| &s + &x)
            }),
        )
    }

    #[inline]
    pub fn contract_tensor2_10<
        'l,
        'r,
        ResultT: Zero,
        RhsT,
        RhsS0: Space<N1> + DualSpace<S1>,
        RhsS1: Space<RHS_N1>,
        const RHS_N1: usize,
    >(
        &'l self,
        rhs: &'r Tensor2<RhsT, RhsS0, RhsS1, N1, RHS_N1>,
    ) -> Tensor2<ResultT, S0, RhsS1, N0, RHS_N1>
    where
        &'r RhsT: Mul<&'l T>,
        for<'a, 'b> &'a ResultT: Add<&'b <&'r RhsT as Mul<&'l T>>::Output, Output = ResultT>,
    {
        rhs.s0.check_duality(&self.s1);
        Tensor2::from_raw(
            self.s0.clone(),
            rhs.s1.clone(),
            self.raw.ref_map(|lv| {
                lv.array_zip_iter(&rhs.raw)
                    .map(|(l, rv)| rv.scale1(l))
                    .arrays1_sum()
            }),
        )
    }

    #[inline]
    pub fn contract_tensor3_10<
        'l,
        'r,
        ResultT: Zero,
        RhsT,
        RhsS0: Space<N1> + DualSpace<S1>,
        RhsS1: Space<RHS_N1>,
        RhsS2: Space<RHS_N2>,
        const RHS_N1: usize,
        const RHS_N2: usize,
    >(
        &'l self,
        rhs: &'r Tensor3<RhsT, RhsS0, RhsS1, RhsS2, N1, RHS_N1, RHS_N2>,
    ) -> Tensor3<ResultT, S0, RhsS1, RhsS2, N0, RHS_N1, RHS_N2>
    where
        &'r RhsT: Mul<&'l T>,
        for<'a, 'b> &'a ResultT: Add<&'b <&'r RhsT as Mul<&'l T>>::Output, Output = ResultT>,
    {
        rhs.s0.check_duality(&self.s1);
        Tensor3::from_raw(
            self.s0.clone(),
            rhs.s1.clone(),
            rhs.s2.clone(),
            self.raw.ref_map(|lv| {
                lv.array_zip_iter(&rhs.raw)
                    .map(|(l, rvv)| rvv.scale2(l))
                    .arrays2_sum()
            }),
        )
    }
}

impl<
        T,
        S0: Space<N0>,
        S1: Space<N1>,
        S2: Space<N2>,
        const N0: usize,
        const N1: usize,
        const N2: usize,
    > Tensor3<T, S0, S1, S2, N0, N1, N2>
{
    #[inline]
    pub fn from_raw(s0: S0, s1: S1, s2: S2, raw: [[[T; N2]; N1]; N0]) -> Self {
        Self { s0, s1, s2, raw }
    }

    #[inline]
    pub fn sub_tensors(&self) -> [Tensor2<T, S1, S2, N1, N2>; N0]
    where
        T: Clone,
    {
        self.raw
            .ref_map(|x| Tensor2::from_raw(self.s1.clone(), self.s2.clone(), x.clone()))
    }

    #[inline]
    pub fn zero(s0: S0, s1: S1, s2: S2) -> Self
    where
        T: Zero,
    {
        Self::from_raw(s0, s1, s2, from_fn(|_| from_fn(|_| from_fn(|_| T::zero()))))
    }

    #[inline]
    pub fn descale(&self) -> Self
    where
        T: Descale,
    {
        let factor = T::descaling_factor(self.raw.flatten().flatten().iter());
        Tensor3::from_raw(
            self.s0.clone(),
            self.s1.clone(),
            self.s2.clone(),
            self.raw
                .ref_map(|xvv| xvv.ref_map(|xv| xv.ref_map(|x| x.descale(&factor)))),
        )
    }

    #[inline]
    pub fn scale<'a, Factor: Copy>(
        &'a self,
        factor: Factor,
    ) -> Tensor3<<&'a T as Mul<Factor>>::Output, S0, S1, S2, N0, N1, N2>
    where
        &'a T: Mul<Factor>,
    {
        Tensor3::from_raw(
            self.s0.clone(),
            self.s1.clone(),
            self.s2.clone(),
            self.raw.scale3(factor),
        )
    }

    #[inline]
    pub fn swap021(&self) -> Tensor3<T, S0, S2, S1, N0, N2, N1>
    where
        T: Clone,
    {
        Tensor3::from_raw(
            self.s0.clone(),
            self.s2.clone(),
            self.s1.clone(),
            from_fn(|i0| from_fn(|i2| from_fn(|i1| self.raw[i0][i1][i2].clone()))),
        )
    }

    #[inline]
    pub fn swap102(&self) -> Tensor3<T, S1, S0, S2, N1, N0, N2>
    where
        T: Clone,
    {
        Tensor3::from_raw(
            self.s1.clone(),
            self.s0.clone(),
            self.s2.clone(),
            from_fn(|i1| from_fn(|i0| from_fn(|i2| self.raw[i0][i1][i2].clone()))),
        )
    }

    #[inline]
    pub fn swap120(&self) -> Tensor3<T, S1, S2, S0, N1, N2, N0>
    where
        T: Clone,
    {
        Tensor3::from_raw(
            self.s1.clone(),
            self.s2.clone(),
            self.s0.clone(),
            from_fn(|i1| from_fn(|i2| from_fn(|i0| self.raw[i0][i1][i2].clone()))),
        )
    }

    #[inline]
    pub fn swap201(&self) -> Tensor3<T, S2, S0, S1, N2, N0, N1>
    where
        T: Clone,
    {
        Tensor3::from_raw(
            self.s2.clone(),
            self.s0.clone(),
            self.s1.clone(),
            from_fn(|i2| from_fn(|i0| from_fn(|i1| self.raw[i0][i1][i2].clone()))),
        )
    }

    #[inline]
    pub fn swap210(&self) -> Tensor3<T, S2, S1, S0, N2, N1, N0>
    where
        T: Clone,
    {
        Tensor3::from_raw(
            self.s2.clone(),
            self.s1.clone(),
            self.s0.clone(),
            from_fn(|i2| from_fn(|i1| from_fn(|i0| self.raw[i0][i1][i2].clone()))),
        )
    }

    #[inline]
    pub fn contract_tensor1_20<
        'l,
        'r,
        ResultT: Sum<<&'r RhsT as Mul<&'l T>>::Output>,
        RhsT,
        RhsS0: Space<N2> + DualSpace<S2>,
    >(
        &'l self,
        rhs: &'r Tensor1<RhsT, RhsS0, N2>,
    ) -> Tensor2<ResultT, S0, S1, N0, N1>
    where
        &'r RhsT: Mul<&'l T>,
    {
        rhs.s0.check_duality(&self.s2);
        Tensor2::from_raw(
            self.s0.clone(),
            self.s1.clone(),
            self.raw.ref_map(|lvv| {
                lvv.ref_map(|lv| lv.array_zip_iter(&rhs.raw).map(|(l, r)| r * l).sum())
            }),
        )
    }

    #[inline]
    pub fn contract_tensor2_20<
        'l,
        'r,
        ResultT: Zero,
        RhsT,
        RhsS0: Space<N2> + DualSpace<S2>,
        RhsS1: Space<RHS_N1>,
        const RHS_N1: usize,
    >(
        &'l self,
        rhs: &'r Tensor2<RhsT, RhsS0, RhsS1, N2, RHS_N1>,
    ) -> Tensor3<ResultT, S0, S1, RhsS1, N0, N1, RHS_N1>
    where
        &'r RhsT: Mul<&'l T>,
        for<'a, 'b> &'a ResultT: Add<&'b <&'r RhsT as Mul<&'l T>>::Output, Output = ResultT>,
    {
        rhs.s0.check_duality(&self.s2);
        Tensor3::from_raw(
            self.s0.clone(),
            self.s1.clone(),
            rhs.s1.clone(),
            self.raw.ref_map(|lvv| {
                lvv.ref_map(|lv| {
                    lv.array_zip_iter(&rhs.raw)
                        .map(|(l, rv)| rv.scale1(l))
                        .arrays1_sum()
                })
            }),
        )
    }
}

impl<'l, 'r, LhsT, RhsT, S0: Space<N0>, const N0: usize> Add<&'r Tensor1<RhsT, S0, N0>>
    for &'l Tensor1<LhsT, S0, N0>
where
    &'l LhsT: ScalarAdd<&'r RhsT>,
{
    type Output = Tensor1<<&'l LhsT as Add<&'r RhsT>>::Output, S0, N0>;

    #[inline]
    fn add(self, rhs: &'r Tensor1<RhsT, S0, N0>) -> Self::Output {
        assert_eq!(self.s0, rhs.s0);
        Tensor1::from_raw(
            self.s0.clone(),
            self.raw.ref_map_with(&rhs.raw, |l, r| l + r),
        )
    }
}

impl<'l, 'r, LhsT, RhsT, S0: Space<N0>, S1: Space<N1>, const N0: usize, const N1: usize>
    Add<&'r Tensor2<RhsT, S0, S1, N0, N1>> for &'l Tensor2<LhsT, S0, S1, N0, N1>
where
    &'l LhsT: ScalarAdd<&'r RhsT>,
{
    type Output = Tensor2<<&'l LhsT as Add<&'r RhsT>>::Output, S0, S1, N0, N1>;

    #[inline]
    fn add(self, rhs: &'r Tensor2<RhsT, S0, S1, N0, N1>) -> Self::Output {
        assert_eq!(self.s0, rhs.s0);
        assert_eq!(self.s1, rhs.s1);
        Tensor2::from_raw(
            self.s0.clone(),
            self.s1.clone(),
            self.raw
                .ref_map_with(&rhs.raw, |lv, rv| lv.ref_map_with(rv, |l, r| l + r)),
        )
    }
}

impl<
        'l,
        'r,
        LhsT,
        RhsT,
        S0: Space<N0>,
        S1: Space<N1>,
        S2: Space<N2>,
        const N0: usize,
        const N1: usize,
        const N2: usize,
    > Add<&'r Tensor3<RhsT, S0, S1, S2, N0, N1, N2>> for &'l Tensor3<LhsT, S0, S1, S2, N0, N1, N2>
where
    &'l LhsT: ScalarAdd<&'r RhsT>,
{
    type Output = Tensor3<<&'l LhsT as Add<&'r RhsT>>::Output, S0, S1, S2, N0, N1, N2>;

    #[inline]
    fn add(self, rhs: &'r Tensor3<RhsT, S0, S1, S2, N0, N1, N2>) -> Self::Output {
        assert_eq!(self.s0, rhs.s0);
        assert_eq!(self.s1, rhs.s1);
        assert_eq!(self.s2, rhs.s2);
        Tensor3::from_raw(
            self.s0.clone(),
            self.s1.clone(),
            self.s2.clone(),
            self.raw.ref_map_with(&rhs.raw, |lvv, rvv| {
                lvv.ref_map_with(rvv, |lv, rv| lv.ref_map_with(rv, |l, r| l + r))
            }),
        )
    }
}

impl<'l, 'r, LhsT, RhsT, S0: Space<N0>, const N0: usize> Sub<&'r Tensor1<RhsT, S0, N0>>
    for &'l Tensor1<LhsT, S0, N0>
where
    &'l LhsT: ScalarSub<&'r RhsT>,
{
    type Output = Tensor1<<&'l LhsT as Sub<&'r RhsT>>::Output, S0, N0>;

    #[inline]
    fn sub(self, rhs: &'r Tensor1<RhsT, S0, N0>) -> Self::Output {
        assert_eq!(self.s0, rhs.s0);
        Tensor1::from_raw(
            self.s0.clone(),
            self.raw.ref_map_with(&rhs.raw, |l, r| l - r),
        )
    }
}

impl<'l, 'r, LhsT, RhsT, S0: Space<N0>, S1: Space<N1>, const N0: usize, const N1: usize>
    Sub<&'r Tensor2<RhsT, S0, S1, N0, N1>> for &'l Tensor2<LhsT, S0, S1, N0, N1>
where
    &'l LhsT: ScalarSub<&'r RhsT>,
{
    type Output = Tensor2<<&'l LhsT as Sub<&'r RhsT>>::Output, S0, S1, N0, N1>;

    #[inline]
    fn sub(self, rhs: &'r Tensor2<RhsT, S0, S1, N0, N1>) -> Self::Output {
        assert_eq!(self.s0, rhs.s0);
        assert_eq!(self.s1, rhs.s1);
        Tensor2::from_raw(
            self.s0.clone(),
            self.s1.clone(),
            self.raw
                .ref_map_with(&rhs.raw, |lv, rv| lv.ref_map_with(rv, |l, r| l - r)),
        )
    }
}

impl<
        'l,
        'r,
        LhsT,
        RhsT,
        S0: Space<N0>,
        S1: Space<N1>,
        S2: Space<N2>,
        const N0: usize,
        const N1: usize,
        const N2: usize,
    > Sub<&'r Tensor3<RhsT, S0, S1, S2, N0, N1, N2>> for &'l Tensor3<LhsT, S0, S1, S2, N0, N1, N2>
where
    &'l LhsT: ScalarSub<&'r RhsT>,
{
    type Output = Tensor3<<&'l LhsT as Sub<&'r RhsT>>::Output, S0, S1, S2, N0, N1, N2>;

    #[inline]
    fn sub(self, rhs: &'r Tensor3<RhsT, S0, S1, S2, N0, N1, N2>) -> Self::Output {
        assert_eq!(self.s0, rhs.s0);
        assert_eq!(self.s1, rhs.s1);
        assert_eq!(self.s2, rhs.s2);
        Tensor3::from_raw(
            self.s0.clone(),
            self.s1.clone(),
            self.s2.clone(),
            self.raw.ref_map_with(&rhs.raw, |lvv, rvv| {
                lvv.ref_map_with(rvv, |lv, rv| lv.ref_map_with(rv, |l, r| l - r))
            }),
        )
    }
}

impl<'a, T, S0: Space<N0>, const N0: usize> Neg for &'a Tensor1<T, S0, N0>
where
    &'a T: ScalarNeg,
{
    type Output = Tensor1<<&'a T as Neg>::Output, S0, N0>;

    #[inline]
    fn neg(self) -> Self::Output {
        Tensor1::from_raw(self.s0.clone(), self.raw.ref_map(|x| -x))
    }
}

impl<'a, T, S0: Space<N0>, S1: Space<N1>, const N0: usize, const N1: usize> Neg
    for &'a Tensor2<T, S0, S1, N0, N1>
where
    &'a T: ScalarNeg,
{
    type Output = Tensor2<<&'a T as Neg>::Output, S0, S1, N0, N1>;

    #[inline]
    fn neg(self) -> Self::Output {
        Tensor2::from_raw(
            self.s0.clone(),
            self.s1.clone(),
            self.raw.ref_map(|xv| xv.ref_map(|x| -x)),
        )
    }
}

impl<
        'a,
        T,
        S0: Space<N0>,
        S1: Space<N1>,
        S2: Space<N2>,
        const N0: usize,
        const N1: usize,
        const N2: usize,
    > Neg for &'a Tensor3<T, S0, S1, S2, N0, N1, N2>
where
    &'a T: ScalarNeg,
{
    type Output = Tensor3<<&'a T as Neg>::Output, S0, S1, S2, N0, N1, N2>;

    #[inline]
    fn neg(self) -> Self::Output {
        Tensor3::from_raw(
            self.s0.clone(),
            self.s1.clone(),
            self.s2.clone(),
            self.raw.ref_map(|xvv| xvv.ref_map(|xv| xv.ref_map(|x| -x))),
        )
    }
}

trait Array1Ext<'a, T: 'a, const N0: usize>
where
    Self: 'a,
{
    fn scale1<Factor: Copy>(self, factor: Factor) -> [<&'a T as Mul<Factor>>::Output; N0]
    where
        &'a T: Mul<Factor>;
}

impl<'a, T: 'a, const N0: usize> Array1Ext<'a, T, N0> for &'a [T; N0] {
    #[inline]
    fn scale1<Factor: Copy>(self, factor: Factor) -> [<&'a T as Mul<Factor>>::Output; N0]
    where
        &'a T: Mul<Factor>,
    {
        self.ref_map(|x| x * factor)
    }
}

trait Array2Ext<'a, T: 'a, const N0: usize, const N1: usize>
where
    Self: 'a,
{
    fn scale2<Factor: Copy>(self, factor: Factor) -> [[<&'a T as Mul<Factor>>::Output; N1]; N0]
    where
        &'a T: Mul<Factor>;
}

impl<'a, T: 'a, const N0: usize, const N1: usize> Array2Ext<'a, T, N0, N1> for &'a [[T; N1]; N0] {
    #[inline]
    fn scale2<Factor: Copy>(self, factor: Factor) -> [[<&'a T as Mul<Factor>>::Output; N1]; N0]
    where
        &'a T: Mul<Factor>,
    {
        self.ref_map(|xv| xv.ref_map(|x| x * factor))
    }
}

trait Array3Ext<'a, T: 'a, const N0: usize, const N1: usize, const N2: usize>
where
    Self: 'a,
{
    fn scale3<Factor: Copy>(
        self,
        factor: Factor,
    ) -> [[[<&'a T as Mul<Factor>>::Output; N2]; N1]; N0]
    where
        &'a T: Mul<Factor>;
}

impl<'a, T: 'a, const N0: usize, const N1: usize, const N2: usize> Array3Ext<'a, T, N0, N1, N2>
    for &'a [[[T; N2]; N1]; N0]
{
    #[inline]
    fn scale3<Factor: Copy>(
        self,
        factor: Factor,
    ) -> [[[<&'a T as Mul<Factor>>::Output; N2]; N1]; N0]
    where
        &'a T: Mul<Factor>,
    {
        self.ref_map(|xvv| xvv.ref_map(|xv| xv.ref_map(|x| x * factor)))
    }
}

trait Arrays1IterExt<T, const N0: usize> {
    fn arrays1_sum<ResultT: Zero>(self) -> [ResultT; N0]
    where
        for<'l, 'r> &'l ResultT: Add<&'r T, Output = ResultT>;
}

impl<IterT: Iterator<Item = [T; N0]>, T, const N0: usize> Arrays1IterExt<T, N0> for IterT {
    #[inline]
    fn arrays1_sum<ResultT: Zero>(self) -> [ResultT; N0]
    where
        for<'l, 'r> &'l ResultT: Add<&'r T, Output = ResultT>,
    {
        self.fold(from_fn(|_| ResultT::zero()), |sv, xv| {
            sv.ref_map_with(&xv, |s, x| s + x)
        })
    }
}

trait Arrays2IterExt<T, const N0: usize, const N1: usize> {
    fn arrays2_sum<ResultT: Zero>(self) -> [[ResultT; N1]; N0]
    where
        for<'l, 'r> &'l ResultT: Add<&'r T, Output = ResultT>;
}

impl<IterT: Iterator<Item = [[T; N1]; N0]>, T, const N0: usize, const N1: usize>
    Arrays2IterExt<T, N0, N1> for IterT
{
    #[inline]
    fn arrays2_sum<ResultT: Zero>(self) -> [[ResultT; N1]; N0]
    where
        for<'l, 'r> &'l ResultT: Add<&'r T, Output = ResultT>,
    {
        self.fold(from_fn(|_| from_fn(|_| ResultT::zero())), |svv, xvv| {
            svv.ref_map_with(&xvv, |sv, xv| sv.ref_map_with(&xv, |s, x| s + x))
        })
    }
}

trait Arrays3IterExt<T, const N0: usize, const N1: usize, const N2: usize> {
    fn arrays3_sum<ResultT: Zero>(self) -> [[[ResultT; N2]; N1]; N0]
    where
        for<'l, 'r> &'l ResultT: Add<&'r T, Output = ResultT>;
}

impl<
        IterT: Iterator<Item = [[[T; N2]; N1]; N0]>,
        T,
        const N0: usize,
        const N1: usize,
        const N2: usize,
    > Arrays3IterExt<T, N0, N1, N2> for IterT
{
    #[inline]
    fn arrays3_sum<ResultT: Zero>(self) -> [[[ResultT; N2]; N1]; N0]
    where
        for<'l, 'r> &'l ResultT: Add<&'r T, Output = ResultT>,
    {
        self.fold(
            from_fn(|_| from_fn(|_| from_fn(|_| ResultT::zero()))),
            |svvv, xvvv| {
                svvv.ref_map_with(&xvvv, |svv, xvv| {
                    svv.ref_map_with(&xvv, |sv, xv| sv.ref_map_with(&xv, |s, x| s + x))
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S0;
    impl Space<2> for S0 {}
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S1;
    impl Space<2> for S1 {}
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S2;
    impl Space<2> for S2 {}
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S3;
    impl Space<2> for S3 {}

    #[test]
    fn test_tensor2_sub_tensors() {
        assert_eq!(
            Tensor2::from_raw(S0, S1, [[10, 20], [30, 40]]).sub_tensors(),
            [
                Tensor1::from_raw(S1, [10, 20]),
                Tensor1::from_raw(S1, [30, 40])
            ]
        );
    }

    #[test]
    fn test_tensor3_sub_tensors() {
        assert_eq!(
            Tensor3::from_raw(S0, S1, S2, [[[10, 20], [30, 40]], [[50, 60], [70, 80]]])
                .sub_tensors(),
            [
                Tensor2::from_raw(S1, S2, [[10, 20], [30, 40]]),
                Tensor2::from_raw(S1, S2, [[50, 60], [70, 80]])
            ]
        );
    }

    #[test]
    fn test_tensor1_zero() {
        let t1 = Tensor1::zero(S0);
        assert_eq!(t1, Tensor1::from_raw(S0, [0, 0]));
    }

    #[test]
    fn test_tensor2_zero() {
        let t2 = Tensor2::zero(S0, S1);
        assert_eq!(t2, Tensor2::from_raw(S0, S1, [[0, 0], [0, 0]]));
    }

    #[test]
    fn test_tensor3_zero() {
        let t3 = Tensor3::zero(S0, S1, S2);
        assert_eq!(
            t3,
            Tensor3::from_raw(S0, S1, S2, [[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        );
    }

    #[test]
    fn test_tensor1_descale() {
        let t1 = Tensor1::from_raw(S0, [10. * 12345., 20. * 12345.]);
        assert!(homogenous_vectors_equal(&t1.clone().descale().raw, &t1.raw));
    }

    #[test]
    fn test_tensor2_descale() {
        let t2 = Tensor2::from_raw(
            S0,
            S1,
            [[10. * 12345., 20. * 12345.], [30. * 12345., 40. * 12345.]],
        );
        assert!(homogenous_vectors_equal(
            &t2.clone().descale().raw.flatten(),
            &t2.raw.flatten()
        ));
    }

    #[test]
    fn test_tensor3_descale() {
        let t3 = Tensor3::from_raw(
            S0,
            S1,
            S2,
            [
                [[10. * 12345., 20. * 12345.], [30. * 12345., 40. * 12345.]],
                [[50. * 12345., 60. * 12345.], [70. * 12345., 80. * 12345.]],
            ],
        );
        assert!(homogenous_vectors_equal(
            &t3.clone().descale().raw.flatten().flatten(),
            &t3.raw.flatten().flatten()
        ));
    }

    #[test]
    fn test_tensor1_scale() {
        let t1 = Tensor1::from_raw(S0, [10, 20]);
        assert_eq!(t1.scale(&3), Tensor1::from_raw(S0, [30, 60]));
    }

    #[test]
    fn test_tensor2_scale() {
        let t2 = Tensor2::from_raw(S0, S1, [[10, 20], [30, 40]]);
        assert_eq!(
            t2.scale(&3),
            Tensor2::from_raw(S0, S1, [[30, 60], [90, 120]])
        );
    }

    #[test]
    fn test_tensor3_scale() {
        let t3 = Tensor3::from_raw(S0, S1, S2, [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]);
        assert_eq!(
            t3.scale(&3),
            Tensor3::from_raw(
                S0,
                S1,
                S2,
                [[[30, 60], [90, 120]], [[150, 180], [210, 240]]]
            )
        );
    }

    #[test]
    fn test_tensor1_mul_tensor1() {
        let t1 = Tensor1::from_raw(S0, [10, 20]);
        let t2 = Tensor1::from_raw(S1, [30, 40]);
        assert_eq!(
            t1.mul_tensor1(&t2),
            Tensor2::from_raw(S0, S1, [[300, 400], [600, 800]])
        );
    }

    #[test]
    fn test_tensor1_mul_tensor2() {
        let t1 = Tensor1::from_raw(S0, [10, 20]);
        let t2 = Tensor2::from_raw(S1, S2, [[30, 40], [50, 60]]);
        assert_eq!(
            t1.mul_tensor2(&t2),
            Tensor3::from_raw(
                S0,
                S1,
                S2,
                [[[300, 400], [500, 600]], [[600, 800], [1000, 1200]]]
            )
        );
    }

    #[test]
    fn test_tensor2_mul_tensor1() {
        let t1 = Tensor2::from_raw(S0, S1, [[10, 20], [30, 40]]);
        let t2 = Tensor1::from_raw(S2, [50, 60]);
        assert_eq!(
            t1.mul_tensor1(&t2),
            Tensor3::from_raw(
                S0,
                S1,
                S2,
                [[[500, 600], [1000, 1200]], [[1500, 1800], [2000, 2400]]]
            )
        );
    }

    #[test]
    fn test_tensor1_contract_tensor1() {
        let t1 = Tensor1::from_raw(S0, [10, 20]);
        let t2 = Tensor1::from_raw(CoSpace(S0), [30, 40]);
        assert_eq!(t1.contract_tensor1_00(&t2), 1100);
    }

    #[test]
    fn test_tensor1_contract_tensor2() {
        let t1 = Tensor1::from_raw(S0, [10, 20]);
        let t2 = Tensor2::from_raw(CoSpace(S0), S1, [[30, 40], [50, 60]]);
        assert_eq!(
            t1.contract_tensor2_00(&t2),
            Tensor1::from_raw(S1, [1300, 1600])
        );
    }

    #[test]
    fn test_tensor1_contract_tensor3() {
        let t1 = Tensor1::from_raw(S0, [10, 20]);
        let t2 = Tensor3::from_raw(
            CoSpace(S0),
            S1,
            S2,
            [[[30, 40], [50, 60]], [[70, 80], [90, 100]]],
        );
        assert_eq!(
            t1.contract_tensor3_00(&t2),
            Tensor2::from_raw(S1, S2, [[1700, 2000], [2300, 2600]])
        );
    }

    #[test]
    fn test_tensor2_contract_tensor1() {
        let t1 = Tensor2::from_raw(S0, S1, [[10, 20], [30, 40]]);
        let t2 = Tensor1::from_raw(CoSpace(S1), [50, 60]);
        assert_eq!(
            t1.contract_tensor1_10(&t2),
            Tensor1::from_raw(S0, [1700, 3900])
        );
    }

    #[test]
    fn test_tensor2_contract_tensor2() {
        let t1 = Tensor2::from_raw(S0, S1, [[10, 20], [30, 40]]);
        let t2 = Tensor2::from_raw(CoSpace(S1), S2, [[50, 60], [70, 80]]);
        assert_eq!(
            t1.contract_tensor2_10(&t2),
            Tensor2::from_raw(S0, S2, [[1900, 2200], [4300, 5000]])
        );
    }

    #[test]
    fn test_tensor2_contract_tensor3() {
        let t1 = Tensor2::from_raw(S0, S1, [[10, 20], [30, 40]]);
        let t2 = Tensor3::from_raw(
            CoSpace(S1),
            S2,
            S3,
            [[[50, 60], [70, 80]], [[90, 100], [110, 120]]],
        );
        assert_eq!(
            t1.contract_tensor3_10(&t2),
            Tensor3::from_raw(
                S0,
                S2,
                S3,
                [[[2300, 2600], [2900, 3200]], [[5100, 5800], [6500, 7200]]]
            )
        );
    }

    #[test]
    fn test_tensor3_contract_tensor1() {
        let t1 = Tensor3::from_raw(S0, S1, S2, [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]);
        let t2 = Tensor1::from_raw(CoSpace(S2), [90, 100]);
        assert_eq!(
            t1.contract_tensor1_20(&t2),
            Tensor2::from_raw(S0, S1, [[2900, 6700], [10500, 14300]])
        );
    }

    #[test]
    fn test_tensor3_contract_tensor2() {
        let t1 = Tensor3::from_raw(S0, S1, S2, [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]);
        let t2 = Tensor2::from_raw(CoSpace(S2), S3, [[90, 100], [110, 120]]);
        assert_eq!(
            t1.contract_tensor2_20(&t2),
            Tensor3::from_raw(
                S0,
                S1,
                S3,
                [
                    [[3100, 3400], [7100, 7800]],
                    [[11100, 12200], [15100, 16600]]
                ]
            )
        );
    }

    #[test]
    fn test_tensor2_swap10() {
        let t2 = Tensor2::from_raw(S0, S1, [[10, 20], [30, 40]]);
        assert_eq!(t2.swap10(), Tensor2::from_raw(S1, S0, [[10, 30], [20, 40]]));
    }

    #[test]
    fn test_tensor3_swap021() {
        let t3 = Tensor3::from_raw(S0, S1, S2, [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]);
        assert_eq!(
            t3.swap021(),
            Tensor3::from_raw(S0, S2, S1, [[[10, 30], [20, 40]], [[50, 70], [60, 80]]])
        );
    }

    #[test]
    fn test_tensor3_swap102() {
        let t3 = Tensor3::from_raw(S0, S1, S2, [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]);
        assert_eq!(
            t3.swap102(),
            Tensor3::from_raw(S1, S0, S2, [[[10, 20], [50, 60]], [[30, 40], [70, 80]]])
        );
    }

    #[test]
    fn test_tensor3_swap120() {
        let t3 = Tensor3::from_raw(S0, S1, S2, [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]);
        assert_eq!(
            t3.swap120(),
            Tensor3::from_raw(S1, S2, S0, [[[10, 50], [20, 60]], [[30, 70], [40, 80]]])
        );
    }

    #[test]
    fn test_tensor3_swap201() {
        let t3 = Tensor3::from_raw(S0, S1, S2, [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]);
        assert_eq!(
            t3.swap201(),
            Tensor3::from_raw(S2, S0, S1, [[[10, 30], [50, 70]], [[20, 40], [60, 80]]])
        );
    }

    #[test]
    fn test_tensor3_swap210() {
        let t3 = Tensor3::from_raw(S0, S1, S2, [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]);
        assert_eq!(
            t3.swap210(),
            Tensor3::from_raw(S2, S1, S0, [[[10, 50], [30, 70]], [[20, 60], [40, 80]]])
        );
    }

    #[test]
    fn test_tensor1_add() {
        let t1 = Tensor1::from_raw(S0, [10, 20]);
        let t2 = Tensor1::from_raw(S0, [1, 2]);
        assert_eq!(&t1 + &t2, Tensor1::from_raw(S0, [11, 22]));
    }

    #[test]
    fn test_tensor2_add() {
        let t1 = Tensor2::from_raw(S0, S1, [[10, 20], [30, 40]]);
        let t2 = Tensor2::from_raw(S0, S1, [[1, 2], [3, 4]]);
        assert_eq!(&t1 + &t2, Tensor2::from_raw(S0, S1, [[11, 22], [33, 44]]));
    }

    #[test]
    fn test_tensor3_add() {
        let t1 = Tensor3::from_raw(S0, S1, S2, [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]);
        let t2 = Tensor3::from_raw(S0, S1, S2, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        assert_eq!(
            &t1 + &t2,
            Tensor3::from_raw(S0, S1, S2, [[[11, 22], [33, 44]], [[55, 66], [77, 88]]])
        );
    }

    #[test]
    fn test_tensor1_sub() {
        let t1 = Tensor1::from_raw(S0, [10, 20]);
        let t2 = Tensor1::from_raw(S0, [1, 2]);
        assert_eq!(&t1 - &t2, Tensor1::from_raw(S0, [9, 18]));
    }

    #[test]
    fn test_tensor2_sub() {
        let t1 = Tensor2::from_raw(S0, S1, [[10, 20], [30, 40]]);
        let t2 = Tensor2::from_raw(S0, S1, [[1, 2], [3, 4]]);
        assert_eq!(&t1 - &t2, Tensor2::from_raw(S0, S1, [[9, 18], [27, 36]]));
    }

    #[test]
    fn test_tensor3_sub() {
        let t1 = Tensor3::from_raw(S0, S1, S2, [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]);
        let t2 = Tensor3::from_raw(S0, S1, S2, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        assert_eq!(
            &t1 - &t2,
            Tensor3::from_raw(S0, S1, S2, [[[9, 18], [27, 36]], [[45, 54], [63, 72]]])
        );
    }

    #[test]
    fn test_tensor1_neg() {
        let t1 = Tensor1::from_raw(S0, [10, 20]);
        assert_eq!(-&t1, Tensor1::from_raw(S0, [-10, -20]));
    }

    #[test]
    fn test_tensor2_neg() {
        let t2 = Tensor2::from_raw(S0, S1, [[10, 20], [30, 40]]);
        assert_eq!(-&t2, Tensor2::from_raw(S0, S1, [[-10, -20], [-30, -40]]));
    }

    #[test]
    fn test_tensor3_neg() {
        let t3 = Tensor3::from_raw(S0, S1, S2, [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]);
        assert_eq!(
            -&t3,
            Tensor3::from_raw(
                S0,
                S1,
                S2,
                [[[-10, -20], [-30, -40]], [[-50, -60], [-70, -80]]]
            )
        );
    }

    fn homogenous_vectors_equal(v0: &[f64], v1: &[f64]) -> bool {
        assert_eq!(
            v0.len(),
            v1.len(),
            "homogenous vectors have different lengths"
        );
        let v0_max = v0.iter().max_by_key(|x| x.abs() as i64).unwrap();
        assert_ne!(v0_max, &0., "first homogenous vector is zero vector");
        let v1_max = v1.iter().max_by_key(|x| x.abs() as i64).unwrap();
        assert_ne!(v1_max, &0., "second homogenous vector is zero vector");
        let v0_fixed: Vec<f64> = v0.iter().map(|x| x * v1_max).collect();
        let v1_fixed: Vec<f64> = v1.iter().map(|x| x * v0_max).collect();
        v0_fixed == v1_fixed
    }
}
