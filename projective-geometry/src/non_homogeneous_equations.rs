use crate::scalar_traits::Zero;
use std::ops::{Div, Mul, Sub};

pub trait NonHomogeneousSolver<T> {
    fn solve(&self, a: &[&[T]], b: &[T]) -> Vec<T>;
}

pub struct NonHomogeneousSolverImpl {}

impl NonHomogeneousSolverImpl {
    pub fn new() -> Self {
        NonHomogeneousSolverImpl {}
    }
}

impl<T: Clone + PartialEq<T> + Zero> NonHomogeneousSolver<T> for NonHomogeneousSolverImpl
where
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: Div<&'b T, Output = T>,
{
    fn solve(&self, a: &[&[T]], b: &[T]) -> Vec<T> {
        let n = b.len();
        assert_eq!(a.len(), n);
        for a_i in a {
            assert_eq!(a_i.len(), n);
        }
        let mut flat_a: Vec<T> = a
            .into_iter()
            .map(|a_i| a_i.iter().cloned())
            .flatten()
            .collect();
        let mut a: Vec<&mut [T]> = flat_a.chunks_exact_mut(n).collect();
        let mut b: Vec<T> = b.iter().cloned().collect();
        assert_eq!(a.len(), n);
        assert_eq!(b.len(), n);
        for j in 0..n {
            let a_j = &mut a[j];
            assert_eq!(a_j.len(), n);
            let a_jj = a_j[j].clone();
            for a_jk in &mut a_j[(j + 1)..] {
                *a_jk = &*a_jk / &a_jj;
            }
            let b_j = &b[j] / &a_jj;
            b[j] = b_j.clone();

            let (rows_before_j, rows_since_j) = a.split_at_mut(j);
            let (a_j, rows_after_j) = rows_since_j.split_first_mut().unwrap();
            for (a_i, b_i) in rows_before_j.iter_mut().zip(b[..j].iter_mut()) {
                assert_eq!(a_i.len(), n);
                let a_ij = a_i[j].clone();
                if a_ij == Zero::zero() {
                    continue;
                }
                for (a_ik, a_jk) in a_i[(j + 1)..].iter_mut().zip(a_j[(j + 1)..].iter()) {
                    *a_ik = &*a_ik - &(&a_ij * a_jk);
                }
                *b_i = &*b_i - &(&a_ij * &b_j);
            }
            for (a_i, b_i) in rows_after_j.iter_mut().zip(b[(j + 1)..].iter_mut()) {
                assert_eq!(a_i.len(), n);
                let a_ij = a_i[j].clone();
                if a_ij == Zero::zero() {
                    continue;
                }
                for (a_ik, a_jk) in a_i[(j + 1)..].iter_mut().zip(a_j[(j + 1)..].iter()) {
                    *a_ik = &*a_ik - &(&a_ij * a_jk);
                }
                *b_i = &*b_i - &(&a_ij * &b_j);
            }
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use crate::array_utils::ArrayExt;
    use crate::non_homogeneous_equations::{NonHomogeneousSolver, NonHomogeneousSolverImpl};
    use crate::test_utils::assert_near;

    #[test]
    fn test_non_homogeneous_solve() {
        let a = [
            [0.366, 0.967, 0.145, 0.445, 0.541],
            [0.921, 0.134, 0.809, 0.983, 0.689],
            [0.478, 0.849, 0.578, 0.675, 0.428],
            [0.021, 0.835, 0.524, 0.943, 0.994],
            [0.486, 0.859, 0.247, 0.964, 0.475],
        ];
        let b = [0.256, 0.937, 0.622, 0.055, 0.214];
        let solver = NonHomogeneousSolverImpl::new();
        let x = solver.solve(&a.ref_map(|a_i| a_i as &[f64]), &b);
        for i in 0..a.len() {
            assert_near(dot_product(&a[i], &*x), b[i], 1e-9);
        }
    }

    #[test]
    fn test_non_homogeneous_solve_with_zeroes() {
        let a = [
            [0.366, 0.967, 0.145, 0.445, 0.541],
            [0.000, 0.134, 0.809, 0.000, 0.689],
            [0.000, 0.849, 0.578, 0.675, 0.428],
            [0.021, 0.000, 0.524, 0.943, 0.994],
            [0.000, 0.000, 0.000, 0.000, 0.475],
        ];
        let b = [0.256, 0.937, 0.622, 0.055, 0.214];
        let solver = NonHomogeneousSolverImpl::new();
        let x = solver.solve(&a.ref_map(|a_i| a_i as &[f64]), &b);
        for i in 0..a.len() {
            assert_near(dot_product(&a[i], &*x), b[i], 1e-9);
        }
    }

    fn dot_product(lhs: &[f64], rhs: &[f64]) -> f64 {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum()
    }
}
