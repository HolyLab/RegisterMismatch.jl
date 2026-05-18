# RegisterMismatch.jl

```@meta
DocTestSetup = quote
    using RegisterMismatch, RegisterCore, FFTW
end
```

RegisterMismatch computes the **mismatch** between two images as a function of
integer translation, using FFT-based cross-correlation for efficiency.
It serves as the CPU backend for image registration in the
[HolyLab](https://github.com/HolyLab) image-analysis ecosystem.

## Installation

This package is registered in the
[HolyLabRegistry](https://github.com/HolyLab/HolyLabRegistry).
Add the registry once, then install as usual:

```julia
using Pkg
pkg"registry add General https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("RegisterMismatch")
```

## Concept

**Mismatch** quantifies how dissimilar two images look when one is translated
relative to the other.  For a shift `(i, j)`, the mismatch is

```
mismatch(i, j) = Σ (fixed[x, y] − moving[x+i, y+j])²  /  (Σ fixed² + Σ moving²)
```

The numerator and denominator are stored separately in a
[`NumDenom`](https://github.com/HolyLab/RegisterCore.jl) value so that results
from multiple apertures can be summed before dividing.  The output of
[`mismatch`](@ref) is a `MismatchArray` indexed from `−maxshift` to `+maxshift`
in each dimension; the shift where the mismatch is smallest is the best
integer-pixel alignment.

For large images, or when the displacement varies across the field, use
[`mismatch_apertures`](@ref) to compute local mismatches on a grid of
sub-regions (apertures) and then combine or interpolate the resulting shift
field.

### Preprocessing

Real fluorescence images often have a non-zero background that dominates the
mismatch signal.  [`highpass`](@ref) removes low-frequency background before
registration.  If `fixed` and `moving` have different sizes, [`nanpad`](@ref)
pads the smaller image with `NaN`s so both inputs have the same size.

## Basic usage

### Global registration

```jldoctest
julia> fixed = zeros(Float32, 32, 32); fixed[10:22, 10:22] .= 1f0;

julia> moving = circshift(fixed, (2, 3));

julia> shift = register_translate(fixed, moving, (5, 5))
CartesianIndex(2, 3)
```

[`register_translate`](@ref) calls [`mismatch`](@ref) internally and returns
the shift that minimises the normalised mismatch.  You can also inspect the
full mismatch array:

```jldoctest
julia> fixed = zeros(Float32, 32, 32); fixed[10:22, 10:22] .= 1f0;

julia> moving = circshift(fixed, (2, 3));

julia> mm = mismatch(fixed, moving, (5, 5));

julia> size(mm)
(11, 11)
```

`mm` is indexed from `−5` to `+5` in each dimension; `mm[2, 3]` holds the
mismatch at shift `(2, 3)`.

### Aperture-based (local) registration

```jldoctest
julia> fixed = zeros(Float32, 32, 32); fixed[10:22, 10:22] .= 1f0;

julia> moving = circshift(fixed, (2, 3));

julia> mms = mismatch_apertures(fixed, moving, (2, 2), (5, 5); flags=FFTW.ESTIMATE);

julia> size(mms)
(2, 2)

julia> size(mms[1, 1])
(11, 11)
```

`mms` is a 2×2 array of `MismatchArray`s, one per aperture.  Pass the result
to downstream routines (e.g. from
[RegisterDeformation.jl](https://github.com/HolyLab/RegisterDeformation.jl))
to recover a smooth displacement field.

### Preprocessing

Strip low-frequency background before computing mismatch:

```jldoctest
julia> A = Float32[sin(0.1i + 0.2j) + 5f0 for i in 1:20, j in 1:20];

julia> Ahp = highpass(Float32, A, (3, 3));

julia> eltype(Ahp)
Float32

julia> maximum(abs, Ahp) < maximum(abs, A)
true
```

Pad images to the same size so that `mismatch` can accept them:

```jldoctest
julia> A = ones(Float32, 5, 5); B = ones(Float32, 8, 7);

julia> Apad, Bpad = nanpad(A, B);

julia> size(Apad)
(8, 7)

julia> isnan(Apad[6, 1])
true
```

### Reusing FFT plans with `CMStorage`

When calling `mismatch!` many times against the same fixed image (e.g. in a
multi-frame registration loop), pre-allocate the FFT working storage once and
reuse it:

```jldoctest
julia> cms = CMStorage{Float32}(undef, (16, 16), (4, 4));

julia> fixed = rand(Float32, 16, 16);

julia> fillfixed!(cms, fixed);

julia> mm = MismatchArray(Float32, 9, 9);

julia> moving_padded = rand(Float32, size(cms.padded));

julia> mm = mismatch!(mm, cms, moving_padded);

julia> size(mm)
(9, 9)
```

### Post-processing

After computing mismatch, [`correctbias!`](@ref) imputes corrupted entries
(arising from camera-bias inhomogeneity along axes through the origin) from
their neighbours.  [`truncatenoise!`](@ref) zeros out entries whose denominator
is below a noise threshold.

```jldoctest
julia> mm = mismatch([1.0 2.0; 3.0 4.0], [1.0 2.0; 3.0 4.0], (1, 1));

julia> nd = mismatch_zeroshift([1.0 2.0; 3.0 4.0], [1.0 2.0; 3.0 4.0])
NumDenom(0.0,60.0)
```

`nd.num == 0` confirms that identical images have zero mismatch at zero shift.
