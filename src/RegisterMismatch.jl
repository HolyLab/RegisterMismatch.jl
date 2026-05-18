module RegisterMismatch

import Base: copy, eltype, ndims

using FFTW: FFTW
using ImageCore: ImageCore, sdims
using MappedArrays: MappedArrays, of_eltype
using PaddedViews: PaddedViews, PaddedView
using Printf: Printf, @printf
using RFFT: RFFT, RCpair, plan_irfft!, plan_rfft!
using Reexport: Reexport, @reexport
using RegisterCore: RegisterCore, MismatchArray, highpass, highpass!, maxshift
import RegisterMismatchCommon: mismatch0, mismatch, mismatch_apertures
@reexport using RegisterMismatchCommon: DimsLike, RegisterMismatchCommon, WidthLike,
                                        FFTPROD, set_FFTPROD,
                                        allocate_mmarrays, aperture_grid, aperture_range,
                                        assertsamesize, checksamesize,
                                        checksize_maxshift, checksizemaxshift,
                                        correctbias!, correctbias,
                                        default_aperture_width,
                                        each_aperture_center, each_point,
                                        mismatch_zeroshift,
                                        nanpad, padranges, padsize,
                                        register_translate, shiftrange, tovec,
                                        truncatenoise!, truncatenoise

export
    CMStorage,
    fillfixed!,
    highpass,
    highpass!,
    inner_threading,
    mismatch0,
    mismatch,
    mismatch!,
    mismatch_apertures,
    mismatch_apertures!,
    set_inner_threading!

const INNER_THREADING = Ref{Bool}(true)

"""
    set_inner_threading!(state::Bool)

Enable (`true`) or disable (`false`) threading in inner mismatch loops. Enabled by default.
"""
set_inner_threading!(state::Bool) = (INNER_THREADING[] = state)

Base.@deprecate allow_inner_threading!(state) set_inner_threading!(state)

"""
    inner_threading() -> Bool

Return whether threading is currently enabled in inner mismatch loops.
"""
inner_threading() = INNER_THREADING[]

macro maybe_threads(ex)
    return :(
        if inner_threading()
            $(esc(:($Threads.@threads $ex)))
        else
            $(esc(ex))
        end
    )
end

"""
The major types and functions exported are:

- `mismatch` and `mismatch!`: compute the mismatch between two images
- `mismatch_apertures` and `mismatch_apertures!`: compute apertured mismatch between two images
- `mismatch_zeroshift` (alias: `mismatch0`): simple direct mismatch calculation with no shift
- `nanpad`: pad the smaller image with NaNs
- `highpass`: highpass filter an image
- `correctbias!`: replace corrupted mismatch data (due to camera bias inhomogeneity) with imputed data
- `truncatenoise!`: threshold mismatch computation to prevent problems from roundoff
- `aperture_grid`: create a regular grid of apertures
- `allocate_mmarrays`: create storage for output of `mismatch_apertures!`
- `CMStorage`: a type that facilitates re-use of intermediate storage during registration computations
"""
RegisterMismatch

FFTW.set_num_threads(min(Sys.CPU_THREADS, 8))
set_FFTPROD([2, 3])  # default: FFT sizes are products of 2^a * 3^b

mutable struct NanCorrFFTs{T <: AbstractFloat, N, RCType <: RCpair{T, N}}
    const I0::RCType
    const I1::RCType
    const I2::RCType
end

copy(x::NanCorrFFTs) = NanCorrFFTs(copy(x.I0), copy(x.I1), copy(x.I2))

"""
    CMStorage{T}(undef, aperture_width, maxshift; flags=FFTW.ESTIMATE, timelimit=Inf, display=true) -> CMStorage{T}

Pre-allocate FFT working storage for mismatch computations over apertures of size `aperture_width`
with translations up to `maxshift`. The element type `T` (e.g., `Float32` or `Float64`) must be
specified explicitly.

`flags` is an FFTW planning flag: `FFTW.ESTIMATE` (default, instant) or `FFTW.MEASURE` /
`FFTW.PATIENT` (slower to plan but faster per call — worthwhile when the same aperture size is
reused many times). `timelimit` caps planning time in seconds. Set `display=false` to suppress
the planning progress message printed when `flags != FFTW.ESTIMATE`.

The typical low-level workflow is:
1. Construct `CMStorage` once.
2. Call [`fillfixed!`](@ref) to load the fixed image.
3. Call [`mismatch!`](@ref) for each moving image.

# Examples
```jldoctest
julia> cms = CMStorage{Float32}(undef, (10, 10), (3, 3));

julia> eltype(cms)
Float32

julia> ndims(cms)
2
```
"""
mutable struct CMStorage{T <: AbstractFloat, N, RCType <: RCpair{T, N}, FFT <: Function, IFFT <: Function}
    const aperture_width::Vector{Float64}
    const maxshift::Vector{Int}
    const getindices::Vector{UnitRange{Int}}   # indices for pulling padded data, in source-coordinates
    const padded::Array{T, N}
    const fixed::NanCorrFFTs{T, N, RCType}
    const moving::NanCorrFFTs{T, N, RCType}
    const buf1::RCType
    const buf2::RCType
    # the next two store the result of calling plan_fft! and plan_ifft!
    const fftfunc!::FFT
    const ifftfunc!::IFFT
    const shiftindices::Vector{Vector{Int}} # indices for performing fftshift & snipping from -maxshift:maxshift
end

function CMStorage{T, N}(::UndefInitializer, aperture_width::NTuple{N, <:Real}, maxshift::Dims{N}; flags = FFTW.ESTIMATE, timelimit = Inf, display = true) where {T, N}
    blocksize = map(x -> ceil(Int, x), aperture_width)
    padsz = padsize(blocksize, maxshift)
    padded = Array{T}(undef, padsz)
    getindices = padranges(blocksize, maxshift)
    maxshiftv = [maxshift...]
    region = findall(maxshiftv .> 0)
    fixed = NanCorrFFTs(RCpair{T}(undef, padsz, region), RCpair{T}(undef, padsz, region), RCpair{T}(undef, padsz, region))
    moving = NanCorrFFTs(RCpair{T}(undef, padsz, region), RCpair{T}(undef, padsz, region), RCpair{T}(undef, padsz, region))
    buf1 = RCpair{T}(undef, padsz, region)
    buf2 = RCpair{T}(undef, padsz, region)
    tcalib = 0
    if display && flags != FFTW.ESTIMATE
        print("Planning FFTs (maximum $timelimit seconds)...")
        flush(stdout)
        tcalib = time()
    end
    fftfunc = plan_rfft!(fixed.I0, flags = flags, timelimit = timelimit / 2)
    ifftfunc = plan_irfft!(fixed.I0, flags = flags, timelimit = timelimit / 2)
    if display && flags != FFTW.ESTIMATE
        dt = time() - tcalib
        @printf("done (%.2f seconds)\n", dt)
    end
    shiftindices = Vector{Int}[ [size(padded, i) .+ ((-maxshift[i] + 1):0); 1:(maxshift[i] + 1)] for i in 1:length(maxshift) ]
    return CMStorage{T, N, typeof(buf1), typeof(fftfunc), typeof(ifftfunc)}(Float64[aperture_width...], maxshiftv, getindices, padded, fixed, moving, buf1, buf2, fftfunc, ifftfunc, shiftindices)
end

function CMStorage{T}(::UndefInitializer, aperture_width::WidthLike, maxshift::DimsLike; kwargs...) where {T <: Real}
    N = length(aperture_width)
    length(maxshift) == N || error("aperture_width and maxshift must have the same length, got $N and $(length(maxshift))")
    return CMStorage{T, N}(undef,
                           ntuple(i -> Float64(aperture_width[i]), N),
                           ntuple(i -> Int(maxshift[i]), N);
                           kwargs...)
end

eltype(cms::CMStorage{T, N}) where {T, N} = T
ndims(cms::CMStorage{T, N}) where {T, N} = N
copy(cms::CMStorage) = CMStorage(copy(cms.aperture_width), copy(cms.maxshift), copy(cms.getindices),
                                  copy(cms.padded), copy(cms.fixed), copy(cms.moving),
                                  copy(cms.buf1), copy(cms.buf2),
                                  cms.fftfunc!, cms.ifftfunc!,
                                  deepcopy(cms.shiftindices))

"""
    mm = mismatch([T], fixed, moving, maxshift; normalization=:intensity) -> MismatchArray

Compute the mismatch between `fixed` and `moving` as a function of translations up to size
`maxshift`. Returns a `MismatchArray` indexed from `-maxshift[d]:maxshift[d]` in each dimension.

The optional type parameter `T` sets the element type (default: `Float32` for integer or
fixed-point images, `eltype(fixed)` otherwise). The `normalization` keyword controls the
denominator: `:intensity` (default) normalizes by local image intensity; `:pixels` normalizes
by pixel count.

`fixed` and `moving` must have the same size; pad with `NaN`s as needed (see [`nanpad`](@ref)).

# Examples
```jldoctest
julia> fixed = zeros(Float32, 10, 10); fixed[3:7, 3:7] .= 1;

julia> moving = circshift(fixed, (2, 1));

julia> mm = mismatch(fixed, moving, (3, 3));

julia> size(mm)
(7, 7)
```
"""
function mismatch(::Type{T}, fixed::AbstractArray, moving::AbstractArray, maxshift::DimsLike; normalization = :intensity) where {T <: Real}
    msz = 2 .* maxshift .+ 1
    mm = MismatchArray(T, msz...)
    cms = CMStorage{T}(undef, size(fixed), maxshift)
    fillfixed!(cms, fixed)
    erng = shiftrange.((cms.getindices...,), first.(axes(fixed)) .- 1)  # expanded rng
    mpad = PaddedView(convert(T, NaN), of_eltype(T, moving), erng)
    mismatch!(mm, cms, mpad, normalization = normalization)
    return mm
end

"""
    mismatch!(mm, cms, moving; normalization=:intensity) -> mm

Compute the mismatch as a function of shift, storing the result in `mm`. The `fixed` image must
have been loaded into `cms` via [`fillfixed!`](@ref) before calling this function. `cms` is a
[`CMStorage`](@ref) object.

See also [`mismatch`](@ref) for a higher-level interface that handles setup automatically.
"""
function mismatch!(mm::MismatchArray, cms::CMStorage, moving::AbstractArray; normalization = :intensity)
    # Pad the moving snippet using any available data, including
    # regions that might be in the parent Array but are not present
    # within the boundaries of the SubArray. Use NaN only for pixels
    # truly lacking data.
    checksizemaxshift(mm, cms.maxshift)
    copyto!(cms.padded, CartesianIndices(cms.padded), moving, CartesianIndices(moving))
    fftnan!(cms.moving, cms.padded, cms.fftfunc!)
    # Compute the mismatch
    f0 = complex(cms.fixed.I0)
    f1 = complex(cms.fixed.I1)
    f2 = complex(cms.fixed.I2)
    m0 = complex(cms.moving.I0)
    m1 = complex(cms.moving.I1)
    m2 = complex(cms.moving.I2)
    tnum = complex(cms.buf1)
    tdenom = complex(cms.buf2)
    if normalization == :intensity
        @inbounds @maybe_threads for i in eachindex(tnum)
            c = 2 * conj(f1[i]) * m1[i]
            q = conj(f2[i]) * m0[i] + conj(f0[i]) * m2[i]
            tdenom[i] = q
            tnum[i] = q - c
        end
    elseif normalization == :pixels
        @inbounds @maybe_threads for i in eachindex(tnum)
            f0i, m0i = f0[i], m0[i]
            tdenom[i] = conj(f0i) * m0i
            tnum[i] = conj(f2[i]) * m0i - 2 * conj(f1[i]) * m1[i] + conj(f0i) * m2[i]
        end
    else
        error("normalization $normalization not recognized")
    end
    cms.ifftfunc!(cms.buf1)
    cms.ifftfunc!(cms.buf2)
    copyto!(mm, (view(real(cms.buf1), cms.shiftindices...), view(real(cms.buf2), cms.shiftindices...)))
    return mm
end

"""
    mms = mismatch_apertures([T], fixed, moving, gridsize, maxshift; normalization=:pixels, flags=FFTW.MEASURE, kwargs...)
    mms = mismatch_apertures([T], fixed, moving, aperture_centers, aperture_width, maxshift; kwargs...)

Compute the mismatch between `fixed` and `moving` over a grid of aperture positions. Returns an
array of `MismatchArray`s with the same shape as `aperture_centers`.

The first form divides the image into a `gridsize` regular grid, inferring aperture centers and
widths automatically. The second form accepts explicit `aperture_centers` and `aperture_width`.
In both cases, the mismatch within each aperture is computed for translations up to `maxshift`.

`fixed` and `moving` must have the same size; pad with `NaN`s as needed (see [`nanpad`](@ref)).
The optional type parameter `T` sets the element type of the output (default: `Float32` for
integer or fixed-point images, `eltype(fixed)` otherwise).

The `aperture_centers` can be a vector-of-tuples, vector-of-vectors, or matrix (each point as a
column); for rectangular grids use [`aperture_grid`](@ref).

# Examples
```jldoctest
julia> using FFTW

julia> fixed = ones(Float32, 20, 20); moving = ones(Float32, 20, 20);

julia> mms = mismatch_apertures(fixed, moving, (2, 2), (3, 3); flags=FFTW.ESTIMATE);

julia> size(mms)
(2, 2)

julia> size(mms[1, 1])
(7, 7)
```
"""
function mismatch_apertures(
        ::Type{T},
        fixed::AbstractArray,
        moving::AbstractArray,
        aperture_centers::AbstractArray,
        aperture_width::WidthLike,
        maxshift::DimsLike;
        normalization = :pixels,
        flags = FFTW.MEASURE,
        kwargs...
    ) where {T}
    nd = sdims(fixed)
    (length(aperture_width) == nd && length(maxshift) == nd) || error("Dimensionality mismatch")
    mms = allocate_mmarrays(T, aperture_centers, maxshift)
    cms = CMStorage{T}(undef, aperture_width, maxshift; flags = flags, kwargs...)
    return mismatch_apertures!(mms, fixed, moving, aperture_centers, cms, normalization = normalization)
end

"""
    mms = mismatch_apertures!(mms, fixed, moving, aperture_centers, cms; normalization=:pixels) -> mms

Compute the mismatch between `fixed` and `moving` over apertures at positions `aperture_centers`,
storing results in `mms`. Working storage and parameters are provided by `cms`, a
[`CMStorage`](@ref) object. `mms` must be an array of `MismatchArray`s with length equal to the
number of aperture centers.

See also [`mismatch_apertures`](@ref) for a higher-level interface, and
[`allocate_mmarrays`](@ref) for allocating `mms`.
"""
function mismatch_apertures!(mms, fixed, moving, aperture_centers, cms::CMStorage{T}; normalization = :pixels) where {T}
    N = ndims(cms)
    fillvalue = convert(T, NaN)
    getinds = (cms.getindices...,)::NTuple{ndims(fixed), UnitRange{Int}}
    fixedT, movingT = of_eltype(T, fixed), of_eltype(T, moving)
    for (mm, center) in zip(mms, each_aperture_center(aperture_centers))
        rng = aperture_range(center, cms.aperture_width)
        fsnip = PaddedView(fillvalue, fixedT, rng)
        erng = shiftrange.(getinds, first.(rng) .- 1)  # expanded rng
        msnip = PaddedView(fillvalue, movingT, erng)
        # Perform the calculation
        fillfixed!(cms, fsnip)
        mismatch!(mm, cms, msnip; normalization = normalization)
    end
    return mms
end

# Calculate the components needed to "nancorrelate"
function fftnan!(out::NanCorrFFTs{T}, A::AbstractArray{T}, fftfunc!::Function) where {T <: Real}
    I0 = real(out.I0)
    I1 = real(out.I1)
    I2 = real(out.I2)
    _fftnan!(parent(I0), parent(I1), parent(I2), A)
    fftfunc!(out.I0)
    fftfunc!(out.I1)
    fftfunc!(out.I2)
    return out
end

function _fftnan!(I0, I1, I2, A::AbstractArray{T}) where {T <: Real}
    return @inbounds @maybe_threads for i in CartesianIndices(size(A))
        a = A[i]
        f = !isnan(a)
        I0[i] = f
        af = f ? a : zero(T)
        I1[i] = af
        I2[i] = af * af
    end
end

"""
    fillfixed!(cms::CMStorage, fixed) -> cms

Load the `fixed` image into `cms`, preparing it for mismatch computations. Call this once
before calling [`mismatch!`](@ref) one or more times with different moving images.

This is the setup step performed internally by [`mismatch`](@ref) and
[`mismatch_apertures`](@ref). Use it directly when computing multiple mismatches against the
same fixed image with different moving images.
"""
function fillfixed!(cms::CMStorage{T}, fixed::AbstractArray) where {T}
    fill!(cms.padded, NaN)
    pinds = CartesianIndices(ntuple(d -> (1:size(fixed, d)) .+ cms.maxshift[d], ndims(fixed)))
    copyto!(cms.padded, pinds, fixed, CartesianIndices(fixed))
    return fftnan!(cms.fixed, cms.padded, cms.fftfunc!)
end

end
