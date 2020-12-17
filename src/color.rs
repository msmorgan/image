use std::ops::{Index, IndexMut};

use num_traits::{NumCast, ToPrimitive};

use crate::traits::{Pixel, Primitive};

/// An enumeration over supported color types and bit depths
#[derive(Copy, PartialEq, Eq, Debug, Clone, Hash)]
pub enum ColorType {
    /// Pixel is 8-bit luminance
    L8,
    /// Pixel is 8-bit luminance with an alpha channel
    La8,
    /// Pixel contains 8-bit R, G and B channels
    Rgb8,
    /// Pixel is 8-bit RGB with an alpha channel
    Rgba8,

    /// Pixel is 16-bit luminance
    L16,
    /// Pixel is 16-bit luminance with an alpha channel
    La16,
    /// Pixel is 16-bit RGB
    Rgb16,
    /// Pixel is 16-bit RGBA
    Rgba16,

    /// Pixel contains 8-bit B, G and R channels
    Bgr8,
    /// Pixel is 8-bit BGR with an alpha channel
    Bgra8,

    #[doc(hidden)]
    __NonExhaustive(crate::utils::NonExhaustiveMarker),
}

impl ColorType {
    /// Returns the number of bytes contained in a pixel of `ColorType` ```c```
    pub fn bytes_per_pixel(self) -> u8 {
        match self {
            ColorType::L8 => 1,
            ColorType::L16 | ColorType::La8 => 2,
            ColorType::Rgb8 | ColorType::Bgr8 => 3,
            ColorType::Rgba8 | ColorType::Bgra8 | ColorType::La16 => 4,
            ColorType::Rgb16 => 6,
            ColorType::Rgba16 => 8,
            ColorType::__NonExhaustive(marker) => match marker._private {},
        }
    }

    /// Returns if there is an alpha channel.
    pub fn has_alpha(self) -> bool {
        use ColorType::*;
        match self {
            L8 | L16 | Rgb8 | Bgr8 | Rgb16 => false,
            La8 | Rgba8 | Bgra8 | La16 | Rgba16 => true,
            __NonExhaustive(marker) => match marker._private {},
        }
    }

    /// Returns false if the color scheme is grayscale, true otherwise.
    pub fn has_color(self) -> bool {
        use ColorType::*;
        match self {
            L8 | L16 | La8 | La16 => false,
            Rgb8 | Bgr8 | Rgb16 | Rgba8 | Bgra8 | Rgba16 => true,
            __NonExhaustive(marker) => match marker._private {},
        }
    }

    /// Returns the number of bits contained in a pixel of `ColorType` ```c``` (which will always be
    /// a multiple of 8).
    pub fn bits_per_pixel(self) -> u16 {
        <u16 as From<u8>>::from(self.bytes_per_pixel()) * 8
    }

    /// Returns the number of color channels that make up this pixel
    pub fn channel_count(self) -> u8 {
        let e: ExtendedColorType = self.into();
        e.channel_count()
    }
}

/// An enumeration of color types encountered in image formats.
///
/// This is not exhaustive over all existing image formats but should be granular enough to allow
/// round tripping of decoding and encoding as much as possible. The variants will be extended as
/// necessary to enable this.
///
/// Another purpose is to advise users of a rough estimate of the accuracy and effort of the
/// decoding from and encoding to such an image format.
#[derive(Copy, PartialEq, Eq, Debug, Clone, Hash)]
pub enum ExtendedColorType {
    /// Pixel is 1-bit luminance
    L1,
    /// Pixel is 1-bit luminance with an alpha channel
    La1,
    /// Pixel contains 1-bit R, G and B channels
    Rgb1,
    /// Pixel is 1-bit RGB with an alpha channel
    Rgba1,
    /// Pixel is 2-bit luminance
    L2,
    /// Pixel is 2-bit luminance with an alpha channel
    La2,
    /// Pixel contains 2-bit R, G and B channels
    Rgb2,
    /// Pixel is 2-bit RGB with an alpha channel
    Rgba2,
    /// Pixel is 4-bit luminance
    L4,
    /// Pixel is 4-bit luminance with an alpha channel
    La4,
    /// Pixel contains 4-bit R, G and B channels
    Rgb4,
    /// Pixel is 4-bit RGB with an alpha channel
    Rgba4,
    /// Pixel is 8-bit luminance
    L8,
    /// Pixel is 8-bit luminance with an alpha channel
    La8,
    /// Pixel contains 8-bit R, G and B channels
    Rgb8,
    /// Pixel is 8-bit RGB with an alpha channel
    Rgba8,
    /// Pixel is 16-bit luminance
    L16,
    /// Pixel is 16-bit luminance with an alpha channel
    La16,
    /// Pixel contains 16-bit R, G and B channels
    Rgb16,
    /// Pixel is 16-bit RGB with an alpha channel
    Rgba16,
    /// Pixel contains 8-bit B, G and R channels
    Bgr8,
    /// Pixel is 8-bit BGR with an alpha channel
    Bgra8,

    /// Pixel is of unknown color type with the specified bits per pixel. This can apply to pixels
    /// which are associated with an external palette. In that case, the pixel value is an index
    /// into the palette.
    Unknown(u8),

    #[doc(hidden)]
    __NonExhaustive(crate::utils::NonExhaustiveMarker),
}

impl ExtendedColorType {
    /// Get the number of channels for colors of this type.
    ///
    /// Note that the `Unknown` variant returns a value of `1` since pixels can only be treated as
    /// an opaque datum by the library.
    pub fn channel_count(self) -> u8 {
        match self {
            ExtendedColorType::L1 |
            ExtendedColorType::L2 |
            ExtendedColorType::L4 |
            ExtendedColorType::L8 |
            ExtendedColorType::L16 |
            ExtendedColorType::Unknown(_) => 1,
            ExtendedColorType::La1 |
            ExtendedColorType::La2 |
            ExtendedColorType::La4 |
            ExtendedColorType::La8 |
            ExtendedColorType::La16 => 2,
            ExtendedColorType::Rgb1 |
            ExtendedColorType::Rgb2 |
            ExtendedColorType::Rgb4 |
            ExtendedColorType::Rgb8 |
            ExtendedColorType::Rgb16 |
            ExtendedColorType::Bgr8 => 3,
            ExtendedColorType::Rgba1 |
            ExtendedColorType::Rgba2 |
            ExtendedColorType::Rgba4 |
            ExtendedColorType::Rgba8 |
            ExtendedColorType::Rgba16 |
            ExtendedColorType::Bgra8 => 4,
            ExtendedColorType::__NonExhaustive(marker) => match marker._private {},
        }
    }
}
impl From<ColorType> for ExtendedColorType {
    fn from(c: ColorType) -> Self {
        match c {
            ColorType::L8 => ExtendedColorType::L8,
            ColorType::La8 => ExtendedColorType::La8,
            ColorType::Rgb8 => ExtendedColorType::Rgb8,
            ColorType::Rgba8 => ExtendedColorType::Rgba8,
            ColorType::L16 => ExtendedColorType::L16,
            ColorType::La16 => ExtendedColorType::La16,
            ColorType::Rgb16 => ExtendedColorType::Rgb16,
            ColorType::Rgba16 => ExtendedColorType::Rgba16,
            ColorType::Bgr8 => ExtendedColorType::Bgr8,
            ColorType::Bgra8 => ExtendedColorType::Bgra8,
            ColorType::__NonExhaustive(marker) => match marker._private {},
        }
    }
}

macro_rules! define_colors {
    {$(
        $ident:ident,
        $channels: expr,
        $alphas: expr,
        $interpretation: expr,
        $color_type_u8: expr,
        $color_type_u16: expr,
        #[$doc:meta];
    )*} => {

$( // START Structure definitions

#[$doc]
#[derive(PartialEq, Eq, Clone, Debug, Copy, Hash)]
#[repr(C)]
#[allow(missing_docs)]
pub struct $ident<T: Primitive> (pub [T; $channels]);

impl<T: Primitive + 'static> Pixel for $ident<T> {
    type Subpixel = T;

    const CHANNEL_COUNT: u8 = $channels;

    const COLOR_MODEL: &'static str = $interpretation;

    const COLOR_TYPE: ColorType =
        [$color_type_u8, $color_type_u16][(std::mem::size_of::<T>() > 1) as usize];

    #[inline(always)]
    fn channels(&self) -> &[T] {
        &self.0
    }
    #[inline(always)]
    fn channels_mut(&mut self) -> &mut [T] {
        &mut self.0
    }

    fn channels4(&self) -> (T, T, T, T) {
        const CHANNELS: usize = $channels;
        let mut channels = [T::max_value(); 4];
        channels[0..CHANNELS].copy_from_slice(&self.0);
        (channels[0], channels[1], channels[2], channels[3])
    }

    fn from_channels(a: T, b: T, c: T, d: T,) -> $ident<T> {
        const CHANNELS: usize = $channels;
        *<$ident<T> as Pixel>::from_slice(&[a, b, c, d][..CHANNELS])
    }

    fn from_slice(slice: &[T]) -> &$ident<T> {
        assert_eq!(slice.len(), $channels);
        unsafe { &*(slice.as_ptr() as *const $ident<T>) }
    }
    fn from_slice_mut(slice: &mut [T]) -> &mut $ident<T> {
        assert_eq!(slice.len(), $channels);
        unsafe { &mut *(slice.as_mut_ptr() as *mut $ident<T>) }
    }

    fn to_rgb(&self) -> Rgb<T> {
        Rgb::from(*self)
    }

    fn to_bgr(&self) -> Bgr<T> {
        Bgr::from(*self)
    }

    fn to_rgba(&self) -> Rgba<T> {
        Rgba::from(*self)
    }

    fn to_bgra(&self) -> Bgra<T> {
        Bgra::from(*self)
    }

    fn to_luma(&self) -> Luma<T> {
        let mut pix = Luma([Zero::zero()]);
        FromColor::from_color(&mut pix, self);
        pix
    }

    fn to_luma_alpha(&self) -> LumaA<T> {
        let mut pix = LumaA([Zero::zero(), Zero::zero()]);
        FromColor::from_color(&mut pix, self);
        pix
    }

    fn map<F>(& self, f: F) -> $ident<T> where F: FnMut(T) -> T {
        let mut this = (*self).clone();
        this.apply(f);
        this
    }

    fn apply<F>(&mut self, mut f: F) where F: FnMut(T) -> T {
        for v in &mut self.0 {
            *v = f(*v)
        }
    }

    fn map_with_alpha<F, G>(&self, f: F, g: G) -> $ident<T> where F: FnMut(T) -> T, G: FnMut(T) -> T {
        let mut this = (*self).clone();
        this.apply_with_alpha(f, g);
        this
    }

    fn apply_with_alpha<F, G>(&mut self, mut f: F, mut g: G) where F: FnMut(T) -> T, G: FnMut(T) -> T {
        const ALPHA: usize = $channels - $alphas;
        for v in self.0[..ALPHA].iter_mut() {
            *v = f(*v)
        }
        // The branch of this match is `const`. This way ensures that no subexpression fails the
        // `const_err` lint (the expression `self.0[ALPHA]` would).
        if let Some(v) = self.0.get_mut(ALPHA) {
            *v = g(*v)
        }
    }

    fn map2<F>(&self, other: &Self, f: F) -> $ident<T> where F: FnMut(T, T) -> T {
        let mut this = (*self).clone();
        this.apply2(other, f);
        this
    }

    fn apply2<F>(&mut self, other: &$ident<T>, mut f: F) where F: FnMut(T, T) -> T {
        for (a, &b) in self.0.iter_mut().zip(other.0.iter()) {
            *a = f(*a, b)
        }
    }

    fn invert(&mut self) {
        Invert::invert(self)
    }

    fn blend(&mut self, other: &$ident<T>) {
        Blend::blend(self, other)
    }
}

impl<T: Primitive> Index<usize> for $ident<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, _index: usize) -> &T {
        &self.0[_index]
    }
}

impl<T: Primitive> IndexMut<usize> for $ident<T> {
    #[inline(always)]
    fn index_mut(&mut self, _index: usize) -> &mut T {
        &mut self.0[_index]
    }
}

impl<T: Primitive + 'static> From<[T; $channels]> for $ident<T> {
    fn from(c: [T; $channels]) -> Self {
        Self(c)
    }
}

)* // END Structure definitions

    }
}

define_colors! {
    Rgb, 3, 0, "RGB", ColorType::Rgb8, ColorType::Rgb16, #[doc = "RGB colors"];
    Bgr, 3, 0, "BGR", ColorType::Bgr8, ColorType::Bgr8, #[doc = "BGR colors"];
    Luma, 1, 0, "Y", ColorType::L8, ColorType::L16, #[doc = "Grayscale colors"];
    Rgba, 4, 1, "RGBA", ColorType::Rgba8, ColorType::Rgba16, #[doc = "RGB colors + alpha channel"];
    Bgra, 4, 1, "BGRA", ColorType::Bgra8, ColorType::Bgra8, #[doc = "BGR colors + alpha channel"];
    LumaA, 2, 1, "YA", ColorType::La8, ColorType::La16, #[doc = "Grayscale colors + alpha channel"];
}

/// Coefficients to transform from sRGB to a CIE Y (luminance) value.
const SRGB_LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

#[inline]
fn to_luma<T: Primitive>(r: T, g: T, b: T) -> T {
    let luma = SRGB_LUMA[0] * r.to_f32().unwrap()
        + SRGB_LUMA[1] * g.to_f32().unwrap()
        + SRGB_LUMA[2] * b.to_f32().unwrap();
    NumCast::from(luma).unwrap()
}

#[inline]
fn full_alpha<T: Primitive>() -> T {
    T::max_value()
}

#[inline]
fn downcast_channel(c16: u16) -> u8 {
    NumCast::from(c16.to_u64().unwrap() >> 8).unwrap()
}

#[inline]
fn upcast_channel(c8: u8) -> u16 {
    NumCast::from(c8.to_u64().unwrap() << 8).unwrap()
}

macro_rules! impl_from {
    // Stage 0: Parse the DSL into nested token trees.
    (
        $( ($in_el:ident, $out_el:ident): $convert:ident; )*
        $(
            | $in_ty:ident([ $( $cmp:ident ),* ]) |
            $(
                => $out_ty:ident( $body:expr )
            )+
            ;
        )*
    ) => {
        impl_from!(
            @__split_elements,
            (
                ( (T), (T), () ),
                $(
                    ( ($in_el), ($out_el), ($convert) ),
                )*
            ),
            (
                $(
                    (
                        ( ($in_ty), ( [ $( $cmp ),* ] ) ),
                        (
                            $(
                                ( ($out_ty), ($body) ),
                            )+
                        ),
                    ),
                )*
            ),
        );
    };

    // Stage 1: Call recursively for each group of element conversions.
    (
        @__split_elements,
        ( $( $el_convert:tt, )* ),
        $defs:tt,
    ) => {
        $(
            impl_from!(
                @__split_inputs,
                $el_convert,
                $defs,
            );
        )*
    };

    // Stage 2: Call recursively for each input struct type.
    (
        @__split_inputs,
        $el_convert:tt,
        (
            $(
                (
                    $in_cmp:tt,
                    $outputs:tt,
                ),
            )*
        ),
    ) => {
        $(
            impl_from!(
                @__add_same_ty_conversion,
                $el_convert,
                $in_cmp,
                $outputs,
            );
        )*
    };

    // Stage 3: If a conversion function exists (i.e., the input and output elements are not both
    // `T`), call recursively to also emit `impl From<InTy<InEl>> for InTy<OutEl>`.
    //
    // This extra step is required because defining `impl<T> From<InTy<T>> for InTy<T>` would
    // conflict with the blanket implementation `impl<T> From<T> for T`, defined in `core`.
    (
        @__add_same_ty_conversion,
        ( $in_el:tt, $out_el:tt, ( $( $convert:ident )? ) ),
        ( $in_ty:tt, $components:tt ),
        $outputs:tt,
    ) => {
        $(
            impl_from!(
                @__final,
                ( $in_el, $out_el, ( $convert ) ),
                ( $in_ty, $components ),
                ( $in_ty, $components ),
            );
        )?

        impl_from!(
            @__split_outputs,
            ( $in_el, $out_el, ( $( $convert )? ) ),
            ( $in_ty, $components ),
            $outputs,
        );
    };

    // Stage 4: Call recursively for each output definition.
    (
        @__split_outputs,
        $el_convert:tt,
        $in_cmp:tt,
        ( $( $out_body:tt, )* ),
    ) => {
        $(
            impl_from!(
                @__final,
                $el_convert,
                $in_cmp,
                $out_body,
            );
        )*
    };

    // Final stage: Output impl where elements are both `T`.
    (
        @__final,
        ( (T), (T), () ),
        ( ($in_ty:ident), ( [ $( $cmp:ident ),* ] ) ),
        ( ($out_ty:ident), ($body:expr) ),
    ) => {
        impl<T: Primitive> From<$in_ty<T>> for $out_ty<T> {
            #[inline]
            #[allow(unused_variables)]
            fn from(x: $in_ty<T>) -> Self {
                let $in_ty([ $( $cmp ),* ]) = x;
                $out_ty($body)
            }
        }
    };

    // Final stage: Output impl where elements are different, and there is a conversion function.
    (
        @__final,
        ( ($in_el:ident), ($out_el:ident), ($convert:ident) ),
        ( ($in_ty:ident), ( [ $( $cmp:ident ),* ] ) ),
        ( ($out_ty:ident), ($body:expr) ),
    ) => {
        impl From<$in_ty<$in_el>> for $out_ty<$out_el> {
            #[inline]
            #[allow(unused_variables)]
            fn from(x: $in_ty<$in_el>) -> Self {
                let $in_ty([ $( $cmp ),* ]) = x;
                let [ $( $cmp ),* ] = [ $( $convert($cmp) ),* ];
                $out_ty($body)
            }
        }
    };
}

impl_from! {
    (u8, u16): upcast_channel;
    (u16, u8): downcast_channel;

    |Rgb([r, g, b])|
        => Bgr([b, g, r])
        => Luma([to_luma(r, g, b)])
        => Rgba([r, g, b, full_alpha()])
        => Bgra([b, g, r, full_alpha()])
        => LumaA([to_luma(r, g, b), full_alpha()]);

    |Bgr([b, g, r])|
        => Rgb([r, g, b])
        => Luma([to_luma(r, g, b)])
        => Rgba([r, g, b, full_alpha()])
        => Bgra([b, g, r, full_alpha()])
        => LumaA([to_luma(r, g, b), full_alpha()]);

    |Luma([y])|
        => Rgb([y, y, y])
        => Bgr([y, y, y])
        => Rgba([y, y, y, full_alpha()])
        => Bgra([y, y, y, full_alpha()])
        => LumaA([y, full_alpha()]);

    |Rgba([r, g, b, a])|
        => Rgb([r, g, b])
        => Bgr([b, g, r])
        => Luma([to_luma(r, g, b)])
        => Bgra([b, g, r, a])
        => LumaA([to_luma(r, g, b), a]);

    |Bgra([b, g, r, a])|
        => Rgb([r, g, b])
        => Bgr([b, g, r])
        => Luma([to_luma(r, g, b)])
        => Rgba([r, g, b, a])
        => LumaA([to_luma(r, g, b), a]);

    |LumaA([y, a])|
        => Rgb([y, y, y])
        => Bgr([y, y, y])
        => Luma([y])
        => Rgba([y, y, y, a])
        => Bgra([y, y, y, a]);
}

#[deprecated(note = "Use `From` instead")]
/// Provides color conversions for the different pixel types.
pub(crate) trait FromColor<Other> {
    /// Changes `self` to represent `Other` in the color space of `Self`
    fn from_color(&mut self, _: &Other);
}

/// Copy-based conversions to target pixel types using `FromColor`.
// FIXME: this trait should be removed and replaced with real color space models
// rather than assuming sRGB.
#[deprecated(note = "Use `Into` instead")]
pub(crate) trait IntoColor<Other> {
    /// Constructs a pixel of the target type and converts this pixel into it.
    fn into_color(&self) -> Other;
}

#[allow(deprecated)]
impl<In: Pixel, Out: Pixel + From<In>> FromColor<In> for Out {
    fn from_color(&mut self, other: &In) {
        *self = (*other).into();
    }
}

#[allow(deprecated)]
impl<In: Pixel + Into<Out>, Out: Pixel> IntoColor<Out> for In {
    fn into_color(&self) -> Out {
        self.clone().into()
    }
}


/// Blends a color inter another one
pub(crate) trait Blend {
    /// Blends a color in-place.
    fn blend(&mut self, other: &Self);
}

impl<T: Primitive> Blend for LumaA<T> {
    fn blend(&mut self, other: &LumaA<T>) {
        let max_t = T::max_value();
        let max_t = max_t.to_f32().unwrap();
        let (bg_luma, bg_a) = (self.0[0], self.0[1]);
        let (fg_luma, fg_a) = (other.0[0], other.0[1]);

        let (bg_luma, bg_a) = (
            bg_luma.to_f32().unwrap() / max_t,
            bg_a.to_f32().unwrap() / max_t,
        );
        let (fg_luma, fg_a) = (
            fg_luma.to_f32().unwrap() / max_t,
            fg_a.to_f32().unwrap() / max_t,
        );

        let alpha_final = bg_a + fg_a - bg_a * fg_a;
        if alpha_final == 0.0 {
            return;
        };
        let bg_luma_a = bg_luma * bg_a;
        let fg_luma_a = fg_luma * fg_a;

        let out_luma_a = fg_luma_a + bg_luma_a * (1.0 - fg_a);
        let out_luma = out_luma_a / alpha_final;

        *self = LumaA([
            NumCast::from(max_t * out_luma).unwrap(),
            NumCast::from(max_t * alpha_final).unwrap(),
        ])
    }
}

impl<T: Primitive> Blend for Luma<T> {
    fn blend(&mut self, other: &Luma<T>) {
        *self = *other
    }
}

impl<T: Primitive> Blend for Rgba<T> {
    fn blend(&mut self, other: &Rgba<T>) {
        // http://stackoverflow.com/questions/7438263/alpha-compositing-algorithm-blend-modes#answer-11163848

        // First, as we don't know what type our pixel is, we have to convert to floats between 0.0 and 1.0
        let max_t = T::max_value();
        let max_t = max_t.to_f32().unwrap();
        let (bg_r, bg_g, bg_b, bg_a) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let (fg_r, fg_g, fg_b, fg_a) = (other.0[0], other.0[1], other.0[2], other.0[3]);
        let (bg_r, bg_g, bg_b, bg_a) = (
            bg_r.to_f32().unwrap() / max_t,
            bg_g.to_f32().unwrap() / max_t,
            bg_b.to_f32().unwrap() / max_t,
            bg_a.to_f32().unwrap() / max_t,
        );
        let (fg_r, fg_g, fg_b, fg_a) = (
            fg_r.to_f32().unwrap() / max_t,
            fg_g.to_f32().unwrap() / max_t,
            fg_b.to_f32().unwrap() / max_t,
            fg_a.to_f32().unwrap() / max_t,
        );

        // Work out what the final alpha level will be
        let alpha_final = bg_a + fg_a - bg_a * fg_a;
        if alpha_final == 0.0 {
            return;
        };

        // We premultiply our channels by their alpha, as this makes it easier to calculate
        let (bg_r_a, bg_g_a, bg_b_a) = (bg_r * bg_a, bg_g * bg_a, bg_b * bg_a);
        let (fg_r_a, fg_g_a, fg_b_a) = (fg_r * fg_a, fg_g * fg_a, fg_b * fg_a);

        // Standard formula for src-over alpha compositing
        let (out_r_a, out_g_a, out_b_a) = (
            fg_r_a + bg_r_a * (1.0 - fg_a),
            fg_g_a + bg_g_a * (1.0 - fg_a),
            fg_b_a + bg_b_a * (1.0 - fg_a),
        );

        // Unmultiply the channels by our resultant alpha channel
        let (out_r, out_g, out_b) = (
            out_r_a / alpha_final,
            out_g_a / alpha_final,
            out_b_a / alpha_final,
        );

        // Cast back to our initial type on return
        *self = Rgba([
            NumCast::from(max_t * out_r).unwrap(),
            NumCast::from(max_t * out_g).unwrap(),
            NumCast::from(max_t * out_b).unwrap(),
            NumCast::from(max_t * alpha_final).unwrap(),
        ])
    }
}



impl<T: Primitive> Blend for Bgra<T> {
    fn blend(&mut self, other: &Bgra<T>) {
        // http://stackoverflow.com/questions/7438263/alpha-compositing-algorithm-blend-modes#answer-11163848

        // First, as we don't know what type our pixel is, we have to convert to floats between 0.0 and 1.0
        let max_t = T::max_value();
        let max_t = max_t.to_f32().unwrap();
        let (bg_r, bg_g, bg_b, bg_a) = (self.0[2], self.0[1], self.0[0], self.0[3]);
        let (fg_r, fg_g, fg_b, fg_a) = (other.0[2], other.0[1], other.0[0], other.0[3]);
        let (bg_r, bg_g, bg_b, bg_a) = (
            bg_r.to_f32().unwrap() / max_t,
            bg_g.to_f32().unwrap() / max_t,
            bg_b.to_f32().unwrap() / max_t,
            bg_a.to_f32().unwrap() / max_t,
        );
        let (fg_r, fg_g, fg_b, fg_a) = (
            fg_r.to_f32().unwrap() / max_t,
            fg_g.to_f32().unwrap() / max_t,
            fg_b.to_f32().unwrap() / max_t,
            fg_a.to_f32().unwrap() / max_t,
        );

        // Work out what the final alpha level will be
        let alpha_final = bg_a + fg_a - bg_a * fg_a;
        if alpha_final == 0.0 {
            return;
        };

        // We premultiply our channels by their alpha, as this makes it easier to calculate
        let (bg_r_a, bg_g_a, bg_b_a) = (bg_r * bg_a, bg_g * bg_a, bg_b * bg_a);
        let (fg_r_a, fg_g_a, fg_b_a) = (fg_r * fg_a, fg_g * fg_a, fg_b * fg_a);

        // Standard formula for src-over alpha compositing
        let (out_r_a, out_g_a, out_b_a) = (
            fg_r_a + bg_r_a * (1.0 - fg_a),
            fg_g_a + bg_g_a * (1.0 - fg_a),
            fg_b_a + bg_b_a * (1.0 - fg_a),
        );

        // Unmultiply the channels by our resultant alpha channel
        let (out_r, out_g, out_b) = (
            out_r_a / alpha_final,
            out_g_a / alpha_final,
            out_b_a / alpha_final,
        );

        // Cast back to our initial type on return
        *self = Bgra([
            NumCast::from(max_t * out_b).unwrap(),
            NumCast::from(max_t * out_g).unwrap(),
            NumCast::from(max_t * out_r).unwrap(),
            NumCast::from(max_t * alpha_final).unwrap(),
        ])
    }
}

impl<T: Primitive> Blend for Rgb<T> {
    fn blend(&mut self, other: &Rgb<T>) {
        *self = *other
    }
}

impl<T: Primitive> Blend for Bgr<T> {
    fn blend(&mut self, other: &Bgr<T>) {
        *self = *other
    }
}


/// Invert a color
pub(crate) trait Invert {
    /// Inverts a color in-place.
    fn invert(&mut self);
}

impl<T: Primitive> Invert for LumaA<T> {
    fn invert(&mut self) {
        let l = self.0;
        let max = T::max_value();

        *self = LumaA([max - l[0], l[1]])
    }
}

impl<T: Primitive> Invert for Luma<T> {
    fn invert(&mut self) {
        let l = self.0;

        let max = T::max_value();
        let l1 = max - l[0];

        *self = Luma([l1])
    }
}

impl<T: Primitive> Invert for Rgba<T> {
    fn invert(&mut self) {
        let rgba = self.0;

        let max = T::max_value();

        *self = Rgba([max - rgba[0], max - rgba[1], max - rgba[2], rgba[3]])
    }
}


impl<T: Primitive> Invert for Bgra<T> {
    fn invert(&mut self) {
        let bgra = self.0;

        let max = T::max_value();

        *self = Bgra([max - bgra[2], max - bgra[1], max - bgra[0], bgra[3]])
    }
}


impl<T: Primitive> Invert for Rgb<T> {
    fn invert(&mut self) {
        let rgb = self.0;

        let max = T::max_value();

        let r1 = max - rgb[0];
        let g1 = max - rgb[1];
        let b1 = max - rgb[2];

        *self = Rgb([r1, g1, b1])
    }
}

impl<T: Primitive> Invert for Bgr<T> {
    fn invert(&mut self) {
        let bgr = self.0;

        let max = T::max_value();

        let r1 = max - bgr[2];
        let g1 = max - bgr[1];
        let b1 = max - bgr[0];

        *self = Bgr([b1, g1, r1])
    }
}

#[cfg(test)]
mod tests {
    use super::{Bgr, Bgra, Luma, LumaA, Pixel, Rgb, Rgba};

    #[test]
    fn test_apply_with_alpha_rgba() {
        let mut rgba = Rgba([0, 0, 0, 0]);
        rgba.apply_with_alpha(|s| s, |_| 0xFF);
        assert_eq!(rgba, Rgba([0, 0, 0, 0xFF]));
    }

    #[test]
    fn test_apply_with_alpha_bgra() {
        let mut bgra = Bgra([0, 0, 0, 0]);
        bgra.apply_with_alpha(|s| s, |_| 0xFF);
        assert_eq!(bgra, Bgra([0, 0, 0, 0xFF]));
    }

    #[test]
    fn test_apply_with_alpha_rgb() {
        let mut rgb = Rgb([0, 0, 0]);
        rgb.apply_with_alpha(|s| s, |_| panic!("bug"));
        assert_eq!(rgb, Rgb([0, 0, 0]));
    }

    #[test]
    fn test_apply_with_alpha_bgr() {
        let mut bgr = Bgr([0, 0, 0]);
        bgr.apply_with_alpha(|s| s, |_| panic!("bug"));
        assert_eq!(bgr, Bgr([0, 0, 0]));
    }


    #[test]
    fn test_map_with_alpha_rgba() {
        let rgba = Rgba([0, 0, 0, 0]).map_with_alpha(|s| s, |_| 0xFF);
        assert_eq!(rgba, Rgba([0, 0, 0, 0xFF]));
    }

    #[test]
    fn test_map_with_alpha_rgb() {
        let rgb = Rgb([0, 0, 0]).map_with_alpha(|s| s, |_| panic!("bug"));
        assert_eq!(rgb, Rgb([0, 0, 0]));
    }

    #[test]
    fn test_map_with_alpha_bgr() {
        let bgr = Bgr([0, 0, 0]).map_with_alpha(|s| s, |_| panic!("bug"));
        assert_eq!(bgr, Bgr([0, 0, 0]));
    }


    #[test]
    fn test_map_with_alpha_bgra() {
        let bgra = Bgra([0, 0, 0, 0]).map_with_alpha(|s| s, |_| 0xFF);
        assert_eq!(bgra, Bgra([0, 0, 0, 0xFF]));
    }

    #[test]
    fn test_blend_luma_alpha() {
        let ref mut a = LumaA([255 as u8, 255]);
        let b = LumaA([255 as u8, 255]);
        a.blend(&b);
        assert_eq!(a.0[0], 255);
        assert_eq!(a.0[1], 255);

        let ref mut a = LumaA([255 as u8, 0]);
        let b = LumaA([255 as u8, 255]);
        a.blend(&b);
        assert_eq!(a.0[0], 255);
        assert_eq!(a.0[1], 255);

        let ref mut a = LumaA([255 as u8, 255]);
        let b = LumaA([255 as u8, 0]);
        a.blend(&b);
        assert_eq!(a.0[0], 255);
        assert_eq!(a.0[1], 255);

        let ref mut a = LumaA([255 as u8, 0]);
        let b = LumaA([255 as u8, 0]);
        a.blend(&b);
        assert_eq!(a.0[0], 255);
        assert_eq!(a.0[1], 0);
    }

    #[test]
    fn test_blend_rgba() {
        let ref mut a = Rgba([255 as u8, 255, 255, 255]);
        let b = Rgba([255 as u8, 255, 255, 255]);
        a.blend(&b);
        assert_eq!(a.0, [255, 255, 255, 255]);

        let ref mut a = Rgba([255 as u8, 255, 255, 0]);
        let b = Rgba([255 as u8, 255, 255, 255]);
        a.blend(&b);
        assert_eq!(a.0, [255, 255, 255, 255]);

        let ref mut a = Rgba([255 as u8, 255, 255, 255]);
        let b = Rgba([255 as u8, 255, 255, 0]);
        a.blend(&b);
        assert_eq!(a.0, [255, 255, 255, 255]);

        let ref mut a = Rgba([255 as u8, 255, 255, 0]);
        let b = Rgba([255 as u8, 255, 255, 0]);
        a.blend(&b);
        assert_eq!(a.0, [255, 255, 255, 0]);
    }

    #[test]
    fn test_apply_without_alpha_rgba() {
        let mut rgba = Rgba([0, 0, 0, 0]);
        rgba.apply_without_alpha(|s| s + 1);
        assert_eq!(rgba, Rgba([1, 1, 1, 0]));
    }

    #[test]
    fn test_apply_without_alpha_bgra() {
        let mut bgra = Bgra([0, 0, 0, 0]);
        bgra.apply_without_alpha(|s| s + 1);
        assert_eq!(bgra, Bgra([1, 1, 1, 0]));
    }

    #[test]
    fn test_apply_without_alpha_rgb() {
        let mut rgb = Rgb([0, 0, 0]);
        rgb.apply_without_alpha(|s| s + 1);
        assert_eq!(rgb, Rgb([1, 1, 1]));
    }

    #[test]
    fn test_apply_without_alpha_bgr() {
        let mut bgr = Bgr([0, 0, 0]);
        bgr.apply_without_alpha(|s| s + 1);
        assert_eq!(bgr, Bgr([1, 1, 1]));
    }

    #[test]
    fn test_map_without_alpha_rgba() {
        let rgba = Rgba([0, 0, 0, 0]).map_without_alpha(|s| s + 1);
        assert_eq!(rgba, Rgba([1, 1, 1, 0]));
    }

    #[test]
    fn test_map_without_alpha_rgb() {
        let rgb = Rgb([0, 0, 0]).map_without_alpha(|s| s + 1);
        assert_eq!(rgb, Rgb([1, 1, 1]));
    }

    #[test]
    fn test_map_without_alpha_bgr() {
        let bgr = Bgr([0, 0, 0]).map_without_alpha(|s| s + 1);
        assert_eq!(bgr, Bgr([1, 1, 1]));
    }

    #[test]
    fn test_map_without_alpha_bgra() {
        let bgra = Bgra([0, 0, 0, 0]).map_without_alpha(|s| s + 1);
        assert_eq!(bgra, Bgra([1, 1, 1, 0]));
    }

    macro_rules! test_lossless_conversion {
        ($a:ty, $b:ty, $c:ty) => {
            let a: $a = [<$a as Pixel>::Subpixel::max_value() >> 2; <$a as Pixel>::CHANNEL_COUNT as usize].into();
            let b: $b = a.into_color();
            let c: $c = b.into_color();
            assert_eq!(a.channels(), c.channels());
        };
    }

    #[test]
    #[allow(deprecated)]
    fn test_lossless_conversions() {
        use super::IntoColor;

        test_lossless_conversion!(Bgr<u8>, Rgba<u8>, Bgr<u8>);
        test_lossless_conversion!(Bgra<u8>, Rgba<u8>, Bgra<u8>);
        test_lossless_conversion!(Luma<u8>, Luma<u16>, Luma<u8>);
        test_lossless_conversion!(LumaA<u8>, LumaA<u16>, LumaA<u8>);
        test_lossless_conversion!(Rgb<u8>, Rgb<u16>, Rgb<u8>);
        test_lossless_conversion!(Rgba<u8>, Rgba<u16>, Rgba<u8>);
    }
}
