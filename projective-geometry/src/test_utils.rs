pub fn assert_near(left: f64, right: f64, precision: f64) {
    assert!(
        (left - right).abs() < precision,
        "{left:?} != {right:?} with precision {precision}"
    );
}

pub fn assert_collinear(left: &[f64], right: &[f64], min_abs: f64, precision: f64) {
    assert_eq!(
        left.len(),
        right.len(),
        "homogeneous vectors have different dimensions"
    );
    assert_ne!(left.len(), 0, "homogeneous vectors have zero dimension");
    let left_abs = dot_product(left, left).sqrt();
    assert!(left_abs >= min_abs, "left vector {left:?} is too small");
    let right_abs = dot_product(right, right).sqrt();
    assert!(right_abs >= min_abs, "right vector {right:?} is too small");
    let cos = dot_product(left, right) / (left_abs * right_abs);
    assert!(
        cos * cos > 1. - precision * precision,
        "{left:?} not collinear with {right:?}"
    );
}

fn dot_product(lhs: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum()
}
