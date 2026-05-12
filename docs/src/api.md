# API Reference

```@meta
DocTestSetup = quote
    using RegisterMismatch, RegisterCore, FFTW
end
```

## RegisterMismatch module

```@docs
RegisterMismatch
```

## Core mismatch functions

```@docs
mismatch
mismatch!
mismatch_apertures
mismatch_apertures!
mismatch0
```

## Low-level reusable workflow

```@docs
CMStorage
fillfixed!
```

## Preprocessing

```@docs
highpass
highpass!
nanpad
```

## Post-processing

```@docs
correctbias!
truncatenoise!
```

## Utilities

```@docs
register_translate
aperture_grid
allocate_mmarrays
aperture_range
each_point
set_FFTPROD
```

## Types and helpers (RegisterCore)

```@docs
MismatchArray
NumDenom
separate
ratio
maxshift
mismatcharrays
```

## Indexing helpers (RegisterCore)

```@docs
argmin_mismatch
paddedview
trimmedview
ColonFun
```

## Preprocessing (RegisterCore)

```@docs
PreprocessSNF
```

## Type aliases (RegisterMismatchCommon)

```@docs
DimsLike
WidthLike
```

## Internal helpers

```@docs
assertsamesize
checksize_maxshift
default_aperture_width
padranges
padsize
shiftrange
tovec
RegisterMismatchCommon
RegisterCore
```
