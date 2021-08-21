use druid::{Data, Point, Rect, Vec2};
use std::ops::Mul;

#[derive(Copy, Clone, Data, Debug)]
pub struct ZoomAndTranslation {
    zoom: f64,
    translation: Vec2,
}

impl ZoomAndTranslation {
    pub const IDENTITY: ZoomAndTranslation = ZoomAndTranslation {
        zoom: 1.,
        translation: Vec2::ZERO,
    };

    #[inline]
    pub fn get_zoom(self) -> f64 {
        self.zoom
    }

    #[inline]
    pub fn zoom(center: Point, zoom: f64) -> ZoomAndTranslation {
        ZoomAndTranslation {
            zoom,
            translation: center.to_vec2() - center.to_vec2() * zoom,
        }
    }

    #[inline]
    pub fn translate(translation: Vec2) -> ZoomAndTranslation {
        ZoomAndTranslation {
            zoom: 1.,
            translation,
        }
    }

    #[inline]
    pub fn then_zoom(self, center: Point, zoom: f64) -> ZoomAndTranslation {
        Self::zoom(center, zoom) * self
    }

    #[inline]
    pub fn then_translate(self, translation: Vec2) -> ZoomAndTranslation {
        Self::translate(translation) * self
    }

    #[inline]
    pub fn inverse(self) -> ZoomAndTranslation {
        ZoomAndTranslation {
            zoom: 1. / self.zoom,
            translation: -self.translation / self.zoom,
        }
    }

    #[inline]
    pub fn solve_translation(pos: Point, transformed_pos: Point, zoom: f64) -> ZoomAndTranslation {
        let translation = transformed_pos.to_vec2() - pos.to_vec2() * zoom;
        ZoomAndTranslation { zoom, translation }
    }
}

impl Mul<ZoomAndTranslation> for ZoomAndTranslation {
    type Output = ZoomAndTranslation;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        ZoomAndTranslation {
            zoom: rhs.zoom * self.zoom,
            translation: rhs.translation * self.zoom + self.translation,
        }
    }
}

impl Mul<Point> for ZoomAndTranslation {
    type Output = Point;

    #[inline]
    fn mul(self, rhs: Point) -> Self::Output {
        let components: (f64, f64) = (rhs.to_vec2() * self.zoom + self.translation).into();
        Point::from(components)
    }
}

impl Mul<Rect> for ZoomAndTranslation {
    type Output = Rect;

    fn mul(self, rhs: Rect) -> Self::Output {
        Rect::from_points(
            self * rhs.origin(),
            self * (rhs.origin() + rhs.size().to_vec2()),
        )
    }
}
