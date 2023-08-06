use std::ops::Add;

pub trait Zero {
    fn zero() -> Self;
}

impl Zero for f64 {
    #[inline]
    fn zero() -> Self {
        0.
    }
}

impl Zero for f32 {
    #[inline]
    fn zero() -> Self {
        0.
    }
}

macro_rules! integer_zero {
    ($($type:ident)*) => {
        $(
            impl Zero for $type {
                #[inline]
                fn zero() -> Self {
                    0
                }
            }
        )*
    };
}

integer_zero! { i8 i16 i32 i64 i128 isize u8 u16 u32 u64 usize }

pub trait Descale {
    type Factor;

    fn descaling_factor<'a, I: Iterator<Item = &'a Self>>(iter: I) -> Self::Factor
    where
        Self: 'a;
    fn descale(&self, factor: &Self::Factor) -> Self;
}

impl Descale for f64 {
    type Factor = f64;

    fn descaling_factor<'a, I: Iterator<Item = &'a Self>>(iter: I) -> Self::Factor
    where
        Self: 'a,
    {
        iter.map(|x| x.to_bits() & F64_EXP_MASK)
            .max()
            .map_or(1., |max_exp_bits| {
                f64::from_bits((1f64).to_bits() & !F64_EXP_MASK | max_exp_bits).recip()
            })
    }

    fn descale(&self, factor: &Self::Factor) -> Self {
        *self * factor
    }
}

const F64_MANTISSA_BITS: u32 = f64::MANTISSA_DIGITS - 1;
const F64_SIGN_MASK: u64 = 1u64 << 63;
const F64_EXP_MASK: u64 = ((!1u64) << F64_MANTISSA_BITS) & !F64_SIGN_MASK;

pub trait ScalarAdd<Rhs>: Add<Rhs> {}

impl ScalarAdd<&f64> for &f64 {}
impl ScalarAdd<&f32> for &f32 {}
impl ScalarAdd<&i8> for &i8 {}
impl ScalarAdd<&i16> for &i16 {}
impl ScalarAdd<&i32> for &i32 {}
impl ScalarAdd<&i64> for &i64 {}
impl ScalarAdd<&i128> for &i128 {}
impl ScalarAdd<&isize> for &isize {}
impl ScalarAdd<&u8> for &u8 {}
impl ScalarAdd<&u16> for &u16 {}
impl ScalarAdd<&u32> for &u32 {}
impl ScalarAdd<&u64> for &u64 {}
impl ScalarAdd<&u128> for &u128 {}
impl ScalarAdd<&usize> for &usize {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_descale() {
        assert_eq!(f64::descaling_factor([1234.].iter()), 1. / 1024.);
        assert_eq!(f64::descaling_factor([0.001].iter()), 1024.);
        assert_eq!(
            f64::descaling_factor([0.001, 1234.].iter()),
            f64::descaling_factor([1234.].iter())
        );
        assert_eq!(
            f64::descaling_factor([1234., 0.001].iter()),
            f64::descaling_factor([1234.].iter())
        );
    }
}
