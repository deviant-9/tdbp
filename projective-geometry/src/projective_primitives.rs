use crate::tensors::{CoSpace, Space, Tensor1, Tensor2};

pub type T = f64;

#[derive(Debug)]
pub struct Point1D<S: Space<2>>(Tensor1<T, S, 2>);
#[derive(Debug)]
pub struct Point2D<S: Space<3>>(Tensor1<T, S, 3>);
#[derive(Debug)]
pub struct Point3D<S: Space<4>>(Tensor1<T, S, 4>);

#[derive(Debug)]
pub struct Line2D<S: Space<3>>(Tensor1<T, CoSpace<3, S>, 3>);
#[derive(Debug)]
pub struct Line3D<S: Space<4>>(Tensor2<T, S, S, 4, 4>);

#[derive(Debug)]
pub struct Plane3D<S: Space<4>>(Tensor1<T, CoSpace<4, S>, 4>);

impl<S: Space<2>> Point1D<S> {
    #[inline]
    pub fn from_contra_tensor(tensor: &Tensor1<T, S, 2>) -> Point1D<S> {
        Point1D(tensor.descale())
    }

    #[inline]
    pub fn from_co_tensor(tensor: &Tensor1<T, CoSpace<2, S>, 2>) -> Point1D<S> {
        Point1D::from_contra_tensor(&tensor.contra_levi_contract_0())
    }

    #[inline]
    pub fn contra_tensor(&self) -> Tensor1<T, S, 2> {
        self.0.clone()
    }

    #[inline]
    pub fn co_tensor(&self) -> Tensor1<T, CoSpace<2, S>, 2> {
        self.0.co_levi_contract_0()
    }

    #[inline]
    pub fn from_normal_coords(coords: &[T; 1], s: S) -> Point1D<S> {
        Point1D::from_contra_tensor(&Tensor1::from_raw(s, [coords[0].clone(), 1.]))
    }

    #[inline]
    pub fn normal_coords(&self) -> [T; 1] {
        let scale = self.0.raw.last().unwrap();
        [&self.0.raw[0] / scale]
    }
}

impl<S: Space<3>> Point2D<S> {
    #[inline]
    pub fn from_contra_tensor(tensor: &Tensor1<T, S, 3>) -> Point2D<S> {
        Point2D(tensor.descale())
    }

    /// tensor must be antisymmetric
    #[inline]
    pub fn from_co_tensor(tensor: &Tensor2<T, CoSpace<3, S>, CoSpace<3, S>, 3, 3>) -> Point2D<S> {
        assert_eq!(tensor.s0.0, tensor.s1.0);
        Point2D::from_contra_tensor(&tensor.contra_levi_contract_01())
    }

    #[inline]
    pub fn contra_tensor(&self) -> Tensor1<T, S, 3> {
        self.0.clone()
    }

    #[inline]
    pub fn co_tensor(&self) -> Tensor2<T, CoSpace<3, S>, CoSpace<3, S>, 3, 3> {
        self.0.co_levi_contract_0()
    }

    #[inline]
    pub fn from_normal_coords(coords: &[T; 2], s: S) -> Self {
        Point2D::from_contra_tensor(&Tensor1::from_raw(
            s,
            [coords[0].clone(), coords[1].clone(), 1.],
        ))
    }

    #[inline]
    pub fn normal_coords(&self) -> [T; 2] {
        let scale = self.0.raw.last().unwrap();
        [&self.0.raw[0] / scale, &self.0.raw[1] / scale]
    }

    #[inline]
    pub fn some_outer_line(&self) -> Line2D<S> {
        Line2D::from_co_tensor(&Tensor1::from_raw(
            CoSpace(self.0.s0.clone()),
            self.0.raw.clone(),
        ))
    }

    #[inline]
    pub fn line_to_point(&self, point: &Point2D<S>) -> Line2D<S> {
        assert_eq!(self.0.s0, point.0.s0);
        Line2D::from_contra_tensor(&self.0.mul_tensor1(&point.0))
    }
}

impl<S: Space<4>> Point3D<S> {
    #[inline]
    pub fn from_contra_tensor(tensor: &Tensor1<T, S, 4>) -> Point3D<S> {
        Point3D(tensor.descale())
    }

    #[inline]
    pub fn contra_tensor(&self) -> Tensor1<T, S, 4> {
        self.0.clone()
    }

    #[inline]
    pub fn from_normal_coords(coords: &[T; 3], s: S) -> Self {
        Point3D::from_contra_tensor(&Tensor1::from_raw(
            s,
            [coords[0].clone(), coords[1].clone(), coords[2].clone(), 1.],
        ))
    }

    #[inline]
    pub fn normal_coords(&self) -> [T; 3] {
        let scale = self.0.raw.last().unwrap();
        [
            &self.0.raw[0] / scale,
            &self.0.raw[1] / scale,
            &self.0.raw[2] / scale,
        ]
    }

    #[inline]
    pub fn some_outer_plane(&self) -> Plane3D<S> {
        Plane3D::from_co_tensor(&Tensor1::from_raw(
            CoSpace(self.0.s0.clone()),
            self.0.raw.clone(),
        ))
    }

    #[inline]
    pub fn line_to_point(&self, point: &Point3D<S>) -> Line3D<S> {
        assert_eq!(self.0.s0, point.0.s0);
        let mul = self.0.mul_tensor1(&point.0);
        Line3D::from_contra_tensor(&(&mul - &mul.swap10()))
    }

    #[inline]
    pub fn plane_to_points(&self, point0: &Point3D<S>, point1: &Point3D<S>) -> Plane3D<S> {
        assert_eq!(self.0.s0, point0.0.s0);
        assert_eq!(self.0.s0, point1.0.s0);
        Plane3D::from_co_tensor(
            &self
                .0
                .mul_tensor1(&point0.0)
                .co_levi_contract_01()
                .contract_tensor1_10(&point1.0),
        )
    }

    #[inline]
    pub fn plane_to_line(&self, line: &Line3D<S>) -> Plane3D<S> {
        assert_eq!(self.0.s0, line.0.s0);
        Plane3D::from_co_tensor(&line.0.co_levi_contract_01().contract_tensor1_10(&self.0))
    }
}

impl<S: Space<3>> Line2D<S> {
    /// tensor must be antisymmetric
    #[inline]
    pub fn from_contra_tensor(tensor: &Tensor2<T, S, S, 3, 3>) -> Line2D<S> {
        assert_eq!(tensor.s0, tensor.s1);
        Line2D::from_co_tensor(&tensor.co_levi_contract_01())
    }

    #[inline]
    pub fn from_co_tensor(tensor: &Tensor1<T, CoSpace<3, S>, 3>) -> Line2D<S> {
        Line2D(tensor.descale())
    }

    #[inline]
    pub fn contra_tensor(&self) -> Tensor2<T, S, S, 3, 3> {
        self.0.contra_levi_contract_0()
    }

    #[inline]
    pub fn co_tensor(&self) -> Tensor1<T, CoSpace<3, S>, 3> {
        self.0.clone()
    }

    #[inline]
    pub fn some_outer_point(&self) -> Point2D<S> {
        Point2D::from_contra_tensor(&Tensor1::from_raw(self.0.s0.0.clone(), self.0.raw.clone()))
    }

    #[inline]
    pub fn cross_with_line(&self, line: &Line2D<S>) -> Point2D<S> {
        assert_eq!(self.0.s0, line.0.s0);
        Point2D::from_contra_tensor(&self.0.mul_tensor1(&line.0).contra_levi_contract_01())
    }
}

impl<S: Space<4>> Line3D<S> {
    /// tensor must be antisymmetric
    #[inline]
    pub fn from_contra_tensor(tensor: &Tensor2<T, S, S, 4, 4>) -> Line3D<S> {
        assert_eq!(tensor.s0, tensor.s1);
        Line3D(tensor.descale())
    }

    /// tensor must be antisymmetric
    #[inline]
    pub fn from_co_tensor(tensor: &Tensor2<T, CoSpace<4, S>, CoSpace<4, S>, 4, 4>) -> Line3D<S> {
        assert_eq!(tensor.s0.0, tensor.s1.0);
        Line3D::from_contra_tensor(&tensor.contra_levi_contract_01())
    }

    #[inline]
    pub fn contra_tensor(&self) -> Tensor2<T, S, S, 4, 4> {
        self.0.clone()
    }

    #[inline]
    pub fn co_tensor(&self) -> Tensor2<T, CoSpace<4, S>, CoSpace<4, S>, 4, 4> {
        self.0.co_levi_contract_01()
    }

    #[inline]
    pub fn plane_to_point(&self, point: &Point3D<S>) -> Plane3D<S> {
        assert_eq!(self.0.s0, point.0.s0);
        Plane3D::from_co_tensor(&self.0.co_levi_contract_01().contract_tensor1_10(&point.0))
    }

    #[inline]
    pub fn cross_with_plane(&self, plane: &Plane3D<S>) -> Point3D<S> {
        assert_eq!(self.0.s0, plane.0.s0.0);
        Point3D::from_contra_tensor(&self.0.contract_tensor1_10(&plane.0))
    }
}

impl<S: Space<4>> Plane3D<S> {
    #[inline]
    pub fn from_co_tensor(tensor: &Tensor1<T, CoSpace<4, S>, 4>) -> Plane3D<S> {
        Plane3D(tensor.descale())
    }

    #[inline]
    pub fn co_tensor(&self) -> Tensor1<T, CoSpace<4, S>, 4> {
        self.0.clone()
    }

    #[inline]
    pub fn some_outer_point(&self) -> Point3D<S> {
        Point3D::from_contra_tensor(&Tensor1::from_raw(self.0.s0.0.clone(), self.0.raw.clone()))
    }

    #[inline]
    pub fn cross_with_line(&self, line: &Line3D<S>) -> Point3D<S> {
        assert_eq!(self.0.s0.0, line.0.s0);
        Point3D::from_contra_tensor(&line.0.contract_tensor1_10(&self.0))
    }

    #[inline]
    pub fn cross_with_plane(&self, plane: &Plane3D<S>) -> Line3D<S> {
        assert_eq!(self.0.s0.0, plane.0.s0.0);
        Line3D::from_co_tensor(&self.0.mul_tensor1(&plane.0))
    }

    #[inline]
    pub fn cross_with_planes(&self, plane0: &Plane3D<S>, plane1: &Plane3D<S>) -> Point3D<S> {
        assert_eq!(self.0.s0.0, plane0.0.s0.0);
        assert_eq!(self.0.s0.0, plane1.0.s0.0);
        Point3D::from_contra_tensor(
            &self
                .0
                .mul_tensor1(&plane0.0)
                .contra_levi_contract_01()
                .contract_tensor1_10(&plane1.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar_traits::Zero;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S1;
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S2;
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct S3;

    impl Space<2> for S1 {}
    impl Space<3> for S2 {}
    impl Space<4> for S3 {}

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
    fn test_point1_normal_coords() {
        let p = Point1D::from_normal_coords(&[61.], S1);
        assert!(homogenous_vectors_equal(&p.contra_tensor().raw, &[61., 1.]));
        assert_eq!(p.normal_coords(), [61.]);
    }

    #[test]
    fn test_point2_normal_coords() {
        let p = Point2D::from_normal_coords(&[61., 50.], S2);
        assert!(homogenous_vectors_equal(
            &p.contra_tensor().raw,
            &[61., 50., 1.]
        ));
        assert_eq!(p.normal_coords(), [61., 50.]);
    }

    #[test]
    fn test_point2_some_outer_line() {
        let point = Point2D::from_normal_coords(&[61., 50.], S2);
        assert!(!line2_has_point(&point.some_outer_line(), &point));
    }

    #[test]
    fn test_point2_line_to_point() {
        let p0 = Point2D::from_normal_coords(&[61., 50.], S2);
        let p1 = Point2D::from_normal_coords(&[18., 57.], S2);
        let l01 = p0.line_to_point(&p1);
        assert!(!is_zero(&l01.co_tensor().raw));
        assert!(line2_has_point(&l01, &p0));
        assert!(line2_has_point(&l01, &p1));
        assert!(!line2_has_point(
            &l01,
            &Point2D::from_normal_coords(&[0., 0.], S2)
        ));

        let l10 = p1.line_to_point(&p0);
        assert!(homogenous_vectors_equal(
            &l01.co_tensor().raw,
            &l10.co_tensor().raw
        ));
    }

    fn line2_has_point<S: Space<3>>(line: &Line2D<S>, point: &Point2D<S>) -> bool {
        line.co_tensor().contract_tensor1_00(&point.contra_tensor()) == 0.
    }

    #[test]
    fn test_point3_normal_coords() {
        let p = Point3D::from_normal_coords(&[61., 50., 73.], S3);
        assert!(homogenous_vectors_equal(
            &p.contra_tensor().raw,
            &[61., 50., 73., 1.]
        ));
        assert_eq!(p.normal_coords(), [61., 50., 73.]);
    }

    #[test]
    fn test_point3_some_outer_plane() {
        let point = Point3D::from_normal_coords(&[61., 50., 73.], S3);
        assert!(!plane3_has_point(&point.some_outer_plane(), &point));
    }

    #[test]
    fn test_point3_line_to_point() {
        let p0 = Point3D::from_normal_coords(&[61., 50., 73.], S3);
        let p1 = Point3D::from_normal_coords(&[18., 57., 48.], S3);
        let l01 = p0.line_to_point(&p1);
        assert!(!is_zero(&l01.contra_tensor().raw.flatten()));
        let contra_tensor = l01.contra_tensor();
        let co_tensor = l01.co_tensor();
        for i0 in 0..4 {
            for i1 in 0..4 {
                assert_eq!(
                    contra_tensor.raw[i0][i1], -contra_tensor.raw[i1][i0],
                    "line tensor is not antisymmetric"
                );
                assert_eq!(
                    co_tensor.raw[i0][i1], -co_tensor.raw[i1][i0],
                    "line tensor is not antisymmetric"
                );
            }
        }
        assert!(line3_has_point(&l01, &p0));
        assert!(line3_has_point(&l01, &p1));
        assert!(!line3_has_point(
            &l01,
            &Point3D::from_normal_coords(&[0., 0., 0.], S3)
        ));

        let l10 = p1.line_to_point(&p0);
        assert_homogenous_eq!(
            l01.contra_tensor().raw.flatten(),
            l10.contra_tensor().raw.flatten()
        );
    }

    #[test]
    fn test_point3_plane_to_points() {
        let p0 = Point3D::from_normal_coords(&[61., 50., 73.], S3);
        let p1 = Point3D::from_normal_coords(&[18., 57., 48.], S3);
        let p2 = Point3D::from_normal_coords(&[14., 39., 71.], S3);
        let plane012 = p0.plane_to_points(&p1, &p2);
        assert!(!is_zero(&plane012.co_tensor().raw));
        assert!(plane3_has_point(&plane012, &p0));
        assert!(plane3_has_point(&plane012, &p1));
        assert!(plane3_has_point(&plane012, &p2));
        assert!(!plane3_has_point(
            &plane012,
            &Point3D::from_normal_coords(&[0., 0., 0.], S3)
        ));

        let plane021 = p0.plane_to_points(&p2, &p1);
        let plane102 = p1.plane_to_points(&p0, &p2);
        let plane120 = p1.plane_to_points(&p2, &p0);
        let plane201 = p2.plane_to_points(&p0, &p1);
        let plane210 = p2.plane_to_points(&p1, &p0);
        let plane012_co_tensor = plane012.co_tensor();
        assert!(homogenous_vectors_equal(
            &plane012_co_tensor.raw,
            &plane021.co_tensor().raw
        ));
        assert!(homogenous_vectors_equal(
            &plane012_co_tensor.raw,
            &plane102.co_tensor().raw
        ));
        assert!(homogenous_vectors_equal(
            &plane012_co_tensor.raw,
            &plane120.co_tensor().raw
        ));
        assert!(homogenous_vectors_equal(
            &plane012_co_tensor.raw,
            &plane201.co_tensor().raw
        ));
        assert!(homogenous_vectors_equal(
            &plane012_co_tensor.raw,
            &plane210.co_tensor().raw
        ));
    }

    #[test]
    fn test_point3_plane_to_line() {
        let p0 = Point3D::from_normal_coords(&[61., 50., 73.], S3);
        let p1 = Point3D::from_normal_coords(&[18., 57., 48.], S3);
        let p2 = Point3D::from_normal_coords(&[14., 39., 71.], S3);
        let l12 = p1.line_to_point(&p2);
        let plane012_to_test = p0.plane_to_line(&l12);
        assert!(!is_zero(&plane012_to_test.co_tensor().raw));
        let plane012 = p0.plane_to_points(&p1, &p2);
        assert!(homogenous_vectors_equal(
            &plane012_to_test.co_tensor().raw,
            &plane012.co_tensor().raw
        ));
    }

    fn line3_has_point<S: Space<4>>(line: &Line3D<S>, point: &Point3D<S>) -> bool {
        is_zero(
            &line
                .co_tensor()
                .contract_tensor1_10(&point.contra_tensor())
                .raw,
        )
    }

    fn plane3_has_point<S: Space<4>>(plane: &Plane3D<S>, point: &Point3D<S>) -> bool {
        plane
            .co_tensor()
            .contract_tensor1_00(&point.contra_tensor())
            == 0.
    }

    #[test]
    fn test_line2_some_outer_point() {
        let p0 = Point2D::from_normal_coords(&[61., 50.], S2);
        let p1 = Point2D::from_normal_coords(&[18., 57.], S2);
        let line = p0.line_to_point(&p1);
        assert!(!line2_has_point(&line, &line.some_outer_point()));
    }

    #[test]
    fn test_line2_cross_with_line() {
        let p0 = Point2D::from_normal_coords(&[61., 50.], S2);
        let p1 = Point2D::from_normal_coords(&[18., 57.], S2);
        let p2 = Point2D::from_normal_coords(&[14., 39.], S2);
        let l01 = p0.line_to_point(&p1);
        let l12 = p1.line_to_point(&p2);
        assert_eq!(
            l01.cross_with_line(&l12).normal_coords(),
            p1.normal_coords()
        );
    }

    #[test]
    fn test_line3_plane_to_point() {
        let p0 = Point3D::from_normal_coords(&[61., 50., 73.], S3);
        let p1 = Point3D::from_normal_coords(&[18., 57., 48.], S3);
        let p2 = Point3D::from_normal_coords(&[14., 39., 71.], S3);
        let l01 = p0.line_to_point(&p1);
        let plane012_to_test = l01.plane_to_point(&p2);
        assert!(!is_zero(&plane012_to_test.co_tensor().raw));
        let plane012 = p0.plane_to_points(&p1, &p2);
        assert!(homogenous_vectors_equal(
            &plane012_to_test.co_tensor().raw,
            &plane012.co_tensor().raw
        ));
    }

    #[test]
    fn test_line3_cross_with_plane() {
        let p0 = Point3D::from_normal_coords(&[61., 50., 73.], S3);
        let p1 = Point3D::from_normal_coords(&[18., 57., 48.], S3);
        let p2 = Point3D::from_normal_coords(&[14., 39., 71.], S3);
        let p3 = Point3D::from_normal_coords(&[38., 69., 87.], S3);
        let l01 = p0.line_to_point(&p1);
        let plane123 = p1.plane_to_points(&p2, &p3);
        assert_eq!(
            l01.cross_with_plane(&plane123).normal_coords(),
            p1.normal_coords()
        );
    }

    #[test]
    fn test_plane3_some_outer_point() {
        let p0 = Point3D::from_normal_coords(&[61., 50., 73.], S3);
        let p1 = Point3D::from_normal_coords(&[18., 57., 48.], S3);
        let p2 = Point3D::from_normal_coords(&[14., 39., 71.], S3);
        let plane = p0.plane_to_points(&p1, &p2);
        assert!(!plane3_has_point(&plane, &plane.some_outer_point()));
    }

    #[test]
    fn test_plane3_cross_with_line() {
        let p0 = Point3D::from_normal_coords(&[61., 50., 73.], S3);
        let p1 = Point3D::from_normal_coords(&[18., 57., 48.], S3);
        let p2 = Point3D::from_normal_coords(&[14., 39., 71.], S3);
        let p3 = Point3D::from_normal_coords(&[38., 69., 87.], S3);
        let plane012 = p0.plane_to_points(&p1, &p2);
        let l23 = p2.line_to_point(&p3);
        assert_eq!(
            plane012.cross_with_line(&l23).normal_coords(),
            p2.normal_coords()
        );
    }

    #[test]
    fn test_plane3_cross_with_plane() {
        let p0 = Point3D::from_normal_coords(&[61., 50., 73.], S3);
        let p1 = Point3D::from_normal_coords(&[18., 57., 48.], S3);
        let p2 = Point3D::from_normal_coords(&[14., 39., 71.], S3);
        let p3 = Point3D::from_normal_coords(&[38., 69., 87.], S3);
        let plane012 = p0.plane_to_points(&p1, &p2);
        let plane123 = p1.plane_to_points(&p2, &p3);
        let line12_to_test = plane012.cross_with_plane(&plane123);
        assert!(!is_zero(line12_to_test.contra_tensor().raw.flatten()));
        let line12 = p1.line_to_point(&p2);
        assert!(homogenous_vectors_equal(
            line12_to_test.contra_tensor().raw.flatten(),
            line12.contra_tensor().raw.flatten()
        ));
    }

    #[test]
    fn test_plane3_cross_with_planes() {
        let p0 = Point3D::from_normal_coords(&[61., 50., 73.], S3);
        let p1 = Point3D::from_normal_coords(&[18., 57., 48.], S3);
        let p2 = Point3D::from_normal_coords(&[14., 39., 71.], S3);
        let p3 = Point3D::from_normal_coords(&[38., 69., 87.], S3);
        let plane012 = p0.plane_to_points(&p1, &p2);
        let plane123 = p1.plane_to_points(&p2, &p3);
        let plane230 = p2.plane_to_points(&p3, &p0);
        let p2_to_test = plane012.cross_with_planes(&plane123, &plane230);
        assert!(!is_zero(&p2_to_test.contra_tensor().raw));
        assert!(homogenous_vectors_equal(
            &p2_to_test.contra_tensor().raw,
            &p2.contra_tensor().raw
        ));
    }

    fn is_zero(v0: &[T]) -> bool {
        v0.iter().all(|x| x == &0.)
    }
}
