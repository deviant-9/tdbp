use crate::array_utils::ArrayExt;
use crate::non_homogeneous_equations::NonHomogeneousSolver;
use crate::scalar_traits::Zero;
use std::borrow::Cow;
use std::ops::{Add, Div, Mul, Sub};

pub trait FnWithDerivative<InT, OutT, DerT, const IN_N: usize, const OUT_N: usize> {
    fn f(&self, args: &[InT; IN_N]) -> [OutT; OUT_N];
    fn derivatives(&self, args: &[InT; IN_N]) -> [[DerT; IN_N]; OUT_N];
    fn f_and_derivatives(&self, args: &[InT; IN_N]) -> ([OutT; OUT_N], [[DerT; IN_N]; OUT_N]);
}

pub trait Observables<InT, OutT, OutDiffT, DerT> {
    fn append_expected_values_with_errors(&self, result: &mut Vec<(OutT, OutDiffT)>);
    fn append_actual_values_and_derivatives(
        &self,
        vars: &[InT],
        result: &mut Vec<ObservableValueAndDerivatives<OutT, DerT>>,
    );
}

pub struct ObservableValueAndDerivatives<OutT, DerT> {
    pub value: OutT,
    pub derivatives: Vec<ObservableDerivative<DerT>>,
}

pub struct ObservableDerivative<DerT> {
    pub derivative: DerT,
    pub var_i: usize,
}

pub struct EmptyObservables;

impl<InT, OutT, OutDiffT, DerT> Observables<InT, OutT, OutDiffT, DerT> for EmptyObservables {
    #[inline]
    fn append_expected_values_with_errors(&self, _result: &mut Vec<(OutT, OutDiffT)>) {
        /* No observables, do nothing */
    }

    #[inline]
    fn append_actual_values_and_derivatives(
        &self,
        _vars: &[InT],
        _result: &mut Vec<ObservableValueAndDerivatives<OutT, DerT>>,
    ) {
        /* No observables, do nothing */
    }
}

pub struct ObservablesConcat<O1, O2> {
    pub o1: O1,
    pub o2: O2,
}

impl<
        InT,
        OutT,
        OutDiffT,
        DerT,
        O1: Observables<InT, OutT, OutDiffT, DerT>,
        O2: Observables<InT, OutT, OutDiffT, DerT>,
    > Observables<InT, OutT, OutDiffT, DerT> for ObservablesConcat<O1, O2>
{
    #[inline]
    fn append_expected_values_with_errors(&self, result: &mut Vec<(OutT, OutDiffT)>) {
        self.o1.append_expected_values_with_errors(result);
        self.o2.append_expected_values_with_errors(result);
    }

    #[inline]
    fn append_actual_values_and_derivatives(
        &self,
        vars: &[InT],
        result: &mut Vec<ObservableValueAndDerivatives<OutT, DerT>>,
    ) {
        self.o1.append_actual_values_and_derivatives(vars, result);
        self.o2.append_actual_values_and_derivatives(vars, result);
    }
}

pub struct CommonFObservableArray<F, OutT, OutDiffT, const IN_N: usize, const OUT_N: usize> {
    pub f: F,
    pub observables: Vec<CommonFOneObservable<OutT, OutDiffT, IN_N, OUT_N>>,
}

pub struct CommonFOneObservable<OutT, OutDiffT, const IN_N: usize, const OUT_N: usize> {
    pub vars_indices: [usize; IN_N],
    pub expected_values_with_errors: [(OutT, OutDiffT); OUT_N],
}

impl<
        F: FnWithDerivative<InT, OutT, DerT, IN_N, OUT_N>,
        InT: Clone,
        OutT: Clone,
        OutDiffT: Clone,
        DerT: Clone,
        const IN_N: usize,
        const OUT_N: usize,
    > Observables<InT, OutT, OutDiffT, DerT>
    for CommonFObservableArray<F, OutT, OutDiffT, IN_N, OUT_N>
{
    fn append_expected_values_with_errors(&self, result: &mut Vec<(OutT, OutDiffT)>) {
        result.reserve(self.observables.len() * OUT_N);
        for observable in &self.observables {
            result.extend(observable.expected_values_with_errors.iter().cloned());
        }
    }

    fn append_actual_values_and_derivatives(
        &self,
        vars: &[InT],
        result: &mut Vec<ObservableValueAndDerivatives<OutT, DerT>>,
    ) {
        result.reserve(self.observables.len() * OUT_N);
        for observable in &self.observables {
            let (values, derivative_groups) = self.f.f_and_derivatives(
                &observable
                    .vars_indices
                    .ref_map(|&var_i| vars[var_i].clone()),
            );
            result.extend(values.iter().zip(derivative_groups.iter()).map(
                |(value, derivatives)| {
                    ObservableValueAndDerivatives {
                        value: value.clone(),
                        derivatives: derivatives
                            .iter()
                            .zip(observable.vars_indices.iter())
                            .map(|(derivative, &var_i)| ObservableDerivative {
                                derivative: derivative.clone(),
                                var_i,
                            })
                            .collect(),
                    }
                },
            ));
        }
    }
}

pub struct GaussNewtonFitter<Solver, InT, OutT, OutDiffT, NormT, O> {
    solver: Solver,
    observables: O,
    expected_values_with_errors: Vec<(OutT, OutDiffT)>,
    vars: Vec<InT>,
    a: Vec<Vec<NormT>>,
    b: Vec<NormT>,
    norm_e: Vec<NormT>,
    norm_e_sqr_sum: NormT,
}

impl<Solver, InT: Clone, OutT: Clone, OutDiffT, NormT: Clone + Zero, O>
    GaussNewtonFitter<Solver, InT, OutT, OutDiffT, NormT, O>
{
    pub fn new<DerT>(solver: Solver, observables: O, initial_vars: Vec<InT>) -> Self
    where
        Solver: NonHomogeneousSolver<NormT>,
        O: Observables<InT, OutT, OutDiffT, DerT>,
        for<'a, 'b> &'a OutT: Sub<&'b OutT, Output = OutDiffT>,
        for<'a, 'b> &'a OutDiffT: Div<&'b OutDiffT, Output = NormT>,
        for<'a, 'b> &'a DerT: Div<&'b OutDiffT, Output = NormT>,
        for<'a, 'b> &'a NormT: Add<&'b NormT, Output = NormT>,
        for<'a, 'b> &'a NormT: Mul<&'b NormT, Output = NormT>,
    {
        let mut expected_values_with_errors: Vec<(OutT, OutDiffT)> = Vec::new();
        observables.append_expected_values_with_errors(&mut expected_values_with_errors);
        let (a, b, norm_e, norm_e_sqr_sum) = get_a_b_norm_e_norm_e_sqr_sum(
            &observables,
            &initial_vars,
            &expected_values_with_errors,
        );
        Self {
            solver,
            observables,
            expected_values_with_errors,
            vars: initial_vars,
            a,
            b,
            norm_e,
            norm_e_sqr_sum,
        }
    }

    pub fn get_vars(&self) -> Cow<[InT]> {
        Cow::Borrowed(&self.vars)
    }

    pub fn step<DerT>(&mut self) -> bool
    where
        Solver: NonHomogeneousSolver<NormT>,
        O: Observables<InT, OutT, OutDiffT, DerT>,
        NormT: PartialOrd<NormT>,
        for<'a, 'b> &'a OutT: Sub<&'b OutT, Output = OutDiffT>,
        for<'a, 'b> &'a OutDiffT: Div<&'b OutDiffT, Output = NormT>,
        for<'a, 'b> &'a DerT: Div<&'b OutDiffT, Output = NormT>,
        for<'a, 'b> &'a NormT: Add<&'b NormT, Output = NormT>,
        for<'a, 'b> &'a NormT: Mul<&'b NormT, Output = NormT>,
        for<'a, 'b> &'a InT: Add<&'b NormT, Output = InT>,
    {
        let a: Vec<&[NormT]> = self.a.iter().map(|a_i| &a_i[..]).collect();
        let vars_step = self.solver.solve(&a, &self.b);
        let new_vars: Vec<InT> = self
            .vars
            .iter()
            .zip(vars_step.iter())
            .map(|(var, step)| var + step)
            .collect();
        let (new_a, new_b, new_norm_e, new_norm_e_sqr_sum) = get_a_b_norm_e_norm_e_sqr_sum(
            &self.observables,
            &new_vars,
            &self.expected_values_with_errors,
        );
        if new_norm_e_sqr_sum < self.norm_e_sqr_sum {
            self.vars = new_vars;
            self.a = new_a;
            self.b = new_b;
            self.norm_e = new_norm_e;
            self.norm_e_sqr_sum = new_norm_e_sqr_sum;
            true
        } else {
            false
        }
    }
}

fn get_a_b_norm_e_norm_e_sqr_sum<
    InT,
    OutT,
    OutDiffT,
    DerT,
    NormDerT: Clone + Zero,
    NormDerSqrT: Clone + Zero,
    NormT: Zero,
    O,
>(
    observables: &O,
    vars: &[InT],
    expected_values_with_errors: &[(OutT, OutDiffT)],
) -> (Vec<Vec<NormDerSqrT>>, Vec<NormDerT>, Vec<NormT>, NormT)
where
    O: Observables<InT, OutT, OutDiffT, DerT>,
    for<'a, 'b> &'a OutT: Sub<&'b OutT, Output = OutDiffT>,
    for<'a, 'b> &'a OutDiffT: Div<&'b OutDiffT, Output = NormT>,
    for<'a, 'b> &'a DerT: Div<&'b OutDiffT, Output = NormDerT>,
    for<'a, 'b> &'a NormDerT: Add<&'b NormDerT, Output = NormDerT>,
    for<'a, 'b> &'a NormDerT: Mul<&'b NormT, Output = NormDerT>,
    for<'a, 'b> &'a NormDerT: Mul<&'b NormDerT, Output = NormDerSqrT>,
    for<'a, 'b> &'a NormDerSqrT: Add<&'b NormDerSqrT, Output = NormDerSqrT>,
    for<'a, 'b> &'a NormT: Add<&'b NormT, Output = NormT>,
    for<'a, 'b> &'a NormT: Mul<&'b NormT, Output = NormT>,
{
    let mut actual_values_and_derivatives: Vec<ObservableValueAndDerivatives<OutT, DerT>> =
        Vec::new();
    observables.append_actual_values_and_derivatives(vars, &mut actual_values_and_derivatives);
    let norm_derivatives: Vec<Vec<ObservableDerivative<NormDerT>>> = actual_values_and_derivatives
        .iter()
        .zip(expected_values_with_errors.iter())
        .map(|(value_and_der, (_, error))| {
            value_and_der
                .derivatives
                .iter()
                .map(|d| ObservableDerivative {
                    derivative: &d.derivative / error,
                    var_i: d.var_i,
                })
                .collect()
        })
        .collect();
    let norm_e: Vec<NormT> = expected_values_with_errors
        .iter()
        .zip(actual_values_and_derivatives.iter())
        .map(|((expected_value, error), actual)| &(expected_value - &actual.value) / error)
        .collect();
    let mut b: Vec<NormDerT> = vec![Zero::zero(); vars.len()];
    for (norm_e_i, der_i) in norm_e.iter().zip(norm_derivatives.iter()) {
        for der_ij in der_i {
            b[der_ij.var_i] = &b[der_ij.var_i] + &(&der_ij.derivative * norm_e_i);
        }
    }
    let mut a: Vec<Vec<NormDerSqrT>> = vec![vec![Zero::zero(); vars.len()]; vars.len()];
    for der_i in &norm_derivatives {
        for der_ij in der_i {
            for der_ik in der_i {
                let a_jk = &mut a[der_ij.var_i][der_ik.var_i];
                *a_jk = &*a_jk + &(&der_ij.derivative * &der_ik.derivative);
            }
        }
    }
    let norm_e_sqr_sum: NormT = norm_e
        .iter()
        .map(|e_i| e_i * e_i)
        .fold(Zero::zero(), |acc, item| &acc + &item);
    (a, b, norm_e, norm_e_sqr_sum)
}

#[cfg(test)]
mod tests {
    use crate::model_fitting::{
        CommonFObservableArray, CommonFOneObservable, EmptyObservables, FnWithDerivative,
        GaussNewtonFitter, ObservablesConcat,
    };
    use crate::non_homogeneous_equations::NonHomogeneousSolverImpl;
    use crate::test_utils::assert_near;

    struct TestFn;

    impl FnWithDerivative<f64, f64, f64, 2, 3> for TestFn {
        fn f(&self, [r, phi]: &[f64; 2]) -> [f64; 3] {
            [r * phi.cos(), r * phi.sin(), 0.]
        }

        fn derivatives(&self, [r, phi]: &[f64; 2]) -> [[f64; 2]; 3] {
            [
                [phi.cos(), -r * phi.sin()],
                [phi.sin(), r * phi.cos()],
                [0., 0.],
            ]
        }

        fn f_and_derivatives(&self, args: &[f64; 2]) -> ([f64; 3], [[f64; 2]; 3]) {
            (self.f(args), self.derivatives(args))
        }
    }

    #[test]
    fn test_gauss_newton_fitter() {
        let r0 = 1234.56;
        let phi0 = 1.23456;
        let [x0, y0, z0] = TestFn.f(&[r0, phi0]);
        let observables = EmptyObservables;
        let observables = ObservablesConcat {
            o1: observables,
            o2: CommonFObservableArray {
                f: TestFn,
                observables: vec![CommonFOneObservable {
                    vars_indices: [1, 0],
                    expected_values_with_errors: [(x0, 1.), (y0, 2.), (z0 + 0.5, 3.)],
                }],
            },
        };
        let solver = NonHomogeneousSolverImpl::new();
        let mut fitter = GaussNewtonFitter::new(solver, observables, vec![phi0 * 0.9, r0 * 1.1]);
        for _ in 0..10 {
            if !fitter.step() {
                break;
            }
        }
        let vars = fitter.get_vars();
        assert_near(vars[0], phi0, 1e-9);
        assert_near(vars[1], r0, 1e-9);
    }
}
