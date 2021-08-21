mod geometry;

use druid::piet::InterpolationMode;
use druid::widget::{Controller, Flex, Label, Painter, Scope, ScopePolicy, ScopeTransfer};
use druid::{
    AppLauncher, Application, Code, Data, Env, Event, EventCtx, ImageBuf, Lens, LifeCycle,
    LifeCycleCtx, Modifiers, MouseButton, PaintCtx, PlatformError, Point, RenderContext, Size,
    Widget, WidgetExt, WindowDesc,
};
use geometry::ZoomAndTranslation;
use image::{DynamicImage, ImageFormat, ImageResult};
use piet::Image;
use std::cell::RefCell;
use std::convert::TryInto;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

#[derive(Clone, Data, Lens)]
struct FrameViewerData {
    frames: Rc<[Frame]>,
    first_main_frame_i: usize,
    second_main_frame_i: usize,
}

#[derive(Debug, Clone, Data)]
struct Frame {
    image_path: Rc<Path>,
    #[data(ignore)]
    image: RefCell<Weak<ImageBuf>>,
}

#[derive(Clone, Data, Lens)]
struct FrameViewerPrivateData {
    frames: Rc<[Frame]>,
    frame_i: usize,
    kind: FrameViewerKind,
    base_trans: ZoomAndTranslation,
    #[data(same_fn = "PartialEq::eq")]
    interpolation_mode: InterpolationMode,
    #[data(ignore)]
    held_point: Option<Point>,
    #[data(ignore)]
    cached_image: RefCell<Option<CachedFrameImage>>,
}

impl FrameViewerPrivateData {
    pub fn new(frames: Rc<[Frame]>, frame_i: usize, kind: FrameViewerKind) -> Self {
        FrameViewerPrivateData {
            frames,
            frame_i,
            kind,
            base_trans: ZoomAndTranslation::IDENTITY,
            interpolation_mode: InterpolationMode::Bilinear,
            held_point: Option::None,
            cached_image: RefCell::default(),
        }
    }

    pub fn set_frame_i(&mut self, frame_i: usize) {
        if frame_i != self.frame_i {
            self.base_trans = ZoomAndTranslation::IDENTITY;
        }
        self.frame_i = frame_i;
    }

    pub fn adjust_frame_i(&mut self, offset: isize) {
        let frames_count = self.frames.len();
        let frames_count_isize: isize = frames_count.try_into().unwrap();
        let offset: isize = offset % frames_count_isize;
        let offset: usize = if offset >= 0 {
            offset.try_into().unwrap()
        } else {
            let negative_offset: usize = (-offset).try_into().unwrap();
            frames_count - negative_offset
        };
        self.set_frame_i((self.frame_i + offset) % frames_count)
    }

    pub fn get_image(&self) -> Rc<ImageBuf> {
        let frame_i = self.frame_i;
        let frame = &self.frames[frame_i];
        if let Some(cached_image) = self.cached_image.borrow().as_ref() {
            if cached_image.frame_i == frame_i {
                return Rc::clone(&cached_image.image);
            }
        }
        let mut cached_image = self.cached_image.borrow_mut();

        // Free memory before potentially large new image file loading
        *cached_image = None;

        let optional_image = frame.image.borrow().upgrade();
        let image_rc = optional_image.unwrap_or_else(|| {
            let path = &*frame.image_path;
            println!("opening {:?}", path);
            let raw_image = open_image(path).unwrap();
            println!("opened {:?}", path);
            let image_rc = Rc::new(ImageBuf::from_dynamic_image(raw_image));
            frame.image.replace(Rc::downgrade(&image_rc));
            image_rc
        });
        *cached_image = Some(CachedFrameImage {
            frame_i,
            image: Rc::clone(&image_rc),
        });
        image_rc
    }

    pub fn get_path_string(&self) -> String {
        self.frames[self.frame_i]
            .image_path
            .to_string_lossy()
            .to_string()
    }
}

struct FrameViewerScopeTransfer;

impl ScopeTransfer for FrameViewerScopeTransfer {
    type In = FrameViewerData;
    type State = FrameViewerPrivateData;

    fn read_input(&self, state: &mut Self::State, inner: &Self::In) {
        state.frames = Rc::clone(&inner.frames);
        match state.kind {
            FrameViewerKind::Main1 => state.set_frame_i(inner.first_main_frame_i),
            FrameViewerKind::Main2 => state.set_frame_i(inner.second_main_frame_i),
            FrameViewerKind::Auxiliary => {}
        };
    }

    fn write_back_input(&self, state: &Self::State, inner: &mut Self::In) {
        inner.frames = Rc::clone(&state.frames);
        match state.kind {
            FrameViewerKind::Main1 => inner.first_main_frame_i = state.frame_i,
            FrameViewerKind::Main2 => inner.second_main_frame_i = state.frame_i,
            FrameViewerKind::Auxiliary => {}
        }
    }
}

struct FrameViewerScopePolicy {
    kind: FrameViewerKind,
}

impl FrameViewerScopePolicy {
    pub fn new(kind: FrameViewerKind) -> Self {
        FrameViewerScopePolicy { kind }
    }
}

impl ScopePolicy for FrameViewerScopePolicy {
    type In = FrameViewerData;
    type State = FrameViewerPrivateData;
    type Transfer = FrameViewerScopeTransfer;

    fn create(self, inner: &Self::In) -> (Self::State, Self::Transfer) {
        let frame_i = match self.kind {
            FrameViewerKind::Main1 => inner.first_main_frame_i,
            FrameViewerKind::Main2 => inner.second_main_frame_i,
            FrameViewerKind::Auxiliary => 0,
        };
        let state = FrameViewerPrivateData::new(Rc::clone(&inner.frames), frame_i, self.kind);
        (state, FrameViewerScopeTransfer)
    }
}

#[derive(Clone)]
struct CachedFrameImage {
    frame_i: usize,
    image: Rc<ImageBuf>,
}

#[derive(Copy, Clone, Data, PartialEq)]
enum FrameViewerKind {
    Main1,
    Main2,
    #[allow(dead_code)]
    Auxiliary,
}

struct FrameViewController {}

impl<W: Widget<FrameViewerPrivateData>> Controller<FrameViewerPrivateData, W>
    for FrameViewController
{
    fn event(
        &mut self,
        child: &mut W,
        ctx: &mut EventCtx<'_, '_>,
        event: &Event,
        data: &mut FrameViewerPrivateData,
        env: &Env,
    ) {
        match event {
            Event::WindowConnected => {}
            Event::WindowSize(_) => {}
            Event::MouseDown(mouse_event) => {
                ctx.request_focus();
                match mouse_event.button {
                    MouseButton::None => {}
                    MouseButton::Left => {}
                    MouseButton::Right => {}
                    MouseButton::Middle => {
                        let resize_trans = resize_trans(ctx.size(), data.get_image().size());
                        data.held_point =
                            Some((resize_trans * data.base_trans).inverse() * mouse_event.pos);
                    }
                    MouseButton::X1 => {}
                    MouseButton::X2 => {}
                }
            }
            Event::MouseUp(mouse_event) => match mouse_event.button {
                MouseButton::None => {}
                MouseButton::Left => {}
                MouseButton::Right => {}
                MouseButton::Middle => data.held_point = None,
                MouseButton::X1 => {}
                MouseButton::X2 => {}
            },
            Event::MouseMove(mouse_event) => {
                if let Some(held_point) = data.held_point {
                    if mouse_event.buttons.contains(MouseButton::Middle) {
                        let resize_trans = resize_trans(ctx.size(), data.get_image().size());
                        let un_resized_point = resize_trans.inverse() * mouse_event.pos;
                        data.base_trans = ZoomAndTranslation::solve_translation(
                            held_point,
                            un_resized_point,
                            data.base_trans.get_zoom(),
                        );
                    }
                }
            }
            Event::Wheel(mouse_event) => {
                if mouse_event.wheel_delta.y != 0. {
                    let resize_trans = resize_trans(ctx.size(), data.get_image().size());
                    let un_resized_point = resize_trans.inverse() * mouse_event.pos;
                    let zoom = 2f64.powf(-mouse_event.wheel_delta.y.signum());
                    data.base_trans = data.base_trans.then_zoom(un_resized_point, zoom);
                }
            }
            Event::KeyDown(key_event) => {
                println!("Key event: {:?}", key_event);
                match key_event.code {
                    Code::KeyL => {
                        data.interpolation_mode =
                            if data.interpolation_mode == InterpolationMode::Bilinear {
                                InterpolationMode::NearestNeighbor
                            } else {
                                InterpolationMode::Bilinear
                            }
                    }
                    Code::KeyN if key_event.mods.contains(Modifiers::SHIFT) => {
                        data.adjust_frame_i(10)
                    }
                    Code::KeyN => data.adjust_frame_i(1),
                    Code::KeyP if key_event.mods.contains(Modifiers::SHIFT) => {
                        data.adjust_frame_i(-10)
                    }
                    Code::KeyP => data.adjust_frame_i(-1),
                    Code::KeyZ => data.base_trans = ZoomAndTranslation::IDENTITY,
                    Code::BracketLeft => data.adjust_frame_i(-100),
                    Code::BracketRight => data.adjust_frame_i(100),
                    Code::Comma if key_event.mods.contains(Modifiers::SHIFT) => {
                        data.set_frame_i(0);
                    }
                    Code::Period if key_event.mods.contains(Modifiers::SHIFT) => {
                        data.set_frame_i(data.frames.len() - 1);
                    }
                    _ => {}
                }
            }
            Event::KeyUp(_) => {}
            Event::Paste(_) => {}
            Event::Zoom(_) => {}
            Event::Timer(_) => {}
            Event::AnimFrame(_) => {}
            Event::Command(_) => {}
            Event::Notification(_) => {}
            Event::Internal(_) => {}
        }
        child.event(ctx, event, data, env);
    }

    fn lifecycle(
        &mut self,
        child: &mut W,
        ctx: &mut LifeCycleCtx<'_, '_>,
        event: &LifeCycle,
        data: &FrameViewerPrivateData,
        env: &Env,
    ) {
        match event {
            LifeCycle::WidgetAdded => ctx.register_for_focus(),
            _ => {}
        }
        child.lifecycle(ctx, event, data, env);
    }
}

fn frame_viewer(kind: FrameViewerKind) -> impl Widget<FrameViewerData> {
    let viewer = Painter::new(frame_viewer_paint).controller(FrameViewController {});
    let header = Label::dynamic(|data: &FrameViewerPrivateData, _env| data.get_path_string())
        .on_click(|_ctx, data, _env| {
            Application::global()
                .clipboard()
                .put_string(data.get_path_string())
        });
    let viewer = Flex::column()
        .with_child(header)
        .with_flex_child(viewer, 1.0);
    Scope::new(FrameViewerScopePolicy::new(kind), viewer)
}

fn frame_viewer_paint(paint_ctx: &mut PaintCtx, data: &FrameViewerPrivateData, _env: &Env) {
    let image = data.get_image().to_image(paint_ctx.render_ctx);
    let paint_size = paint_ctx.size();
    let image_size = image.size();
    let image_rect = image_size.to_rect();
    let resize_trans = resize_trans(paint_size, image_size);
    let trans = resize_trans * data.base_trans;
    let image_trans = trans * image_normalizing_trans(image_size);
    paint_ctx.draw_image_area(
        &image,
        image_rect,
        image_trans * image_rect,
        data.interpolation_mode,
    );
}

fn image_normalizing_trans(image_size: Size) -> ZoomAndTranslation {
    ZoomAndTranslation::translate(-image_size.to_vec2() * 0.5)
        .then_zoom(Point::ORIGIN, normal_image_pixel_size(image_size))
}

fn normal_image_size(image_size: Size) -> Size {
    image_size * normal_image_pixel_size(image_size)
}

fn normal_image_pixel_size(image_size: Size) -> f64 {
    2. / f64::max(image_size.width, image_size.height)
}

fn resize_zoom(paint_size: Size, image_size: Size) -> f64 {
    let image_size = normal_image_size(image_size);
    f64::min(
        paint_size.width / image_size.width,
        paint_size.height / image_size.height,
    )
}

fn resize_trans(paint_size: Size, image_size: Size) -> ZoomAndTranslation {
    ZoomAndTranslation::zoom(Point::ORIGIN, resize_zoom(paint_size, image_size))
        .then_translate(paint_size.to_vec2() * 0.5)
}

fn open_image(path: impl AsRef<Path>) -> ImageResult<DynamicImage> {
    let path = path.as_ref();
    image::io::Reader::open(path)
        .unwrap()
        .decode()
        .or_else(|e| {
            // Workaround for JFIF images
            let mut reader = image::io::Reader::open(path).unwrap();
            reader.set_format(ImageFormat::Jpeg);
            reader.decode().or_else(|_| Err(e))
        })
}

fn main() -> Result<(), PlatformError> {
    let frames: Vec<Frame> = std::env::args_os()
        .skip(1)
        .map(|arg| Frame {
            image_path: PathBuf::from(arg).into(),
            image: RefCell::default(),
        })
        .collect();
    let window = WindowDesc::new(build_ui).title("3D By Photos");
    AppLauncher::with_window(window).launch(FrameViewerData {
        frames: frames.into(),
        first_main_frame_i: 0,
        second_main_frame_i: 0,
    })?;
    Ok(())
}

fn build_ui() -> impl Widget<FrameViewerData> {
    Flex::row()
        .with_flex_child(frame_viewer(FrameViewerKind::Main1), 1.0)
        .with_flex_child(frame_viewer(FrameViewerKind::Main2), 1.0)
}
