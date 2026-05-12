# RegisterMismatch.jl

[![CI](https://github.com/HolyLab/RegisterMismatch.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/RegisterMismatch.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/HolyLab/RegisterMismatch.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/RegisterMismatch.jl)
[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/RegisterMismatch.jl/stable)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/RegisterMismatch.jl/dev)

FFT-based image mismatch computation for translation-based image registration.
This package is the CPU backend in the
[HolyLab](https://github.com/HolyLab) image registration stack; a GPU
counterpart is provided by
[RegisterMismatchCuda.jl](https://github.com/HolyLab/RegisterMismatchCuda.jl).

## Installation

This package is registered in the
[HolyLabRegistry](https://github.com/HolyLab/HolyLabRegistry).
Add the registry once, then install as usual:

```julia
using Pkg
pkg"registry add General https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("RegisterMismatch")
```

## Quick start

```julia
using RegisterMismatch

# Two images, moving shifted by (2, 3) relative to fixed
fixed  = zeros(Float32, 32, 32); fixed[10:22, 10:22] .= 1f0
moving = circshift(fixed, (2, 3))

# Find the optimal integer-pixel shift
shift = register_translate(fixed, moving, (5, 5))
# CartesianIndex(2, 3)
```

For non-uniform or large displacements, use `mismatch_apertures` to obtain a
grid of local mismatches and recover a smooth displacement field.

See the [documentation](https://HolyLab.github.io/RegisterMismatch.jl/stable)
for a full guide, including preprocessing (`highpass`, `nanpad`),
aperture-based registration, bias correction, and the low-level `CMStorage`
workflow for repeated computations against the same fixed image.
