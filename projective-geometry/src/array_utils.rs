use std::array::from_fn;

pub trait ArrayExt<'a, T: 'a, const N: usize>
where
    Self: 'a,
{
    type ArrayZipIterT<'r, RhsT: 'r>;

    fn array_zip_iter<'r, RhsT>(self, rhs: &'r [RhsT; N]) -> Self::ArrayZipIterT<'r, RhsT>;

    fn ref_map<F, U>(self, f: F) -> [U; N]
    where
        F: FnMut(&'a T) -> U;

    fn ref_map_with<'r, RhsT, F, U>(self, rhs: &'r [RhsT; N], f: F) -> [U; N]
    where
        F: FnMut(&'a T, &'r RhsT) -> U;
}

impl<'a, T: 'a, const N: usize> ArrayExt<'a, T, N> for &'a [T; N] {
    type ArrayZipIterT<'r, RhsT: 'r> = impl Iterator<Item = (&'a T, &'r RhsT)>;

    #[inline]
    fn array_zip_iter<'r, RhsT>(self, rhs: &'r [RhsT; N]) -> Self::ArrayZipIterT<'r, RhsT> {
        self.iter().zip(rhs.iter())
    }

    #[inline]
    fn ref_map<F, U>(self, mut f: F) -> [U; N]
    where
        F: FnMut(&'a T) -> U,
    {
        from_fn(|i| f(&self[i]))
    }

    #[inline]
    fn ref_map_with<'r, RhsT, F, U>(self, rhs: &'r [RhsT; N], mut f: F) -> [U; N]
    where
        F: FnMut(&'a T, &'r RhsT) -> U,
    {
        from_fn(|i| f(&self[i], &rhs[i]))
    }
}
