import RegisterMismatch  # we use qualified names to support simultaneous testing with RegisterMismatchCuda, which exports the same API
using ImageCore, ImageFiltering
using ImageCore.OffsetArrays
using CenterIndexedArrays, RegisterCore, RegisterMismatchCommon
using Test, Libdl

havecuda = isdefined(Main, :use_cuda) ? Main.use_cuda : !isempty(find_library(["libcudart", "cudart"], ["/usr/local/cuda"]))
if havecuda
    println("Running both CPU and CUDA versions")
    using CUDAdrv
    import RegisterMismatchCuda
    RMlist = (RegisterMismatch, RegisterMismatchCuda)
    devlist = devices() # dev->capability(dev)[1] >= 2, nmax=1)
else
    RMlist = (RegisterMismatch,)
end
accuracy = 1e-5 # new isapprox accuracy = 1e-6

# Iterators
@testset "Iterators" begin
    agrid = [(1,1)  (1,3)  (1,5)  (1,7);
             (4,1)  (4,3)  (4,5)  (4,7);
             (7,1)  (7,3)  (7,5)  (7,7);
             (10,1) (10,3) (10,5) (10,7)]
    @test RegisterMismatch.aperture_grid((10,7), (4,4)) == agrid
    for (i,pt) in enumerate(RegisterMismatch.each_point(agrid))
        @test pt == agrid[i]
    end
    agrida = reshape(reinterpret(Int, vec(agrid)), (2,4,4))
    for (i,pt) in enumerate(RegisterMismatch.each_point(agrida))
        @test pt == [agrid[i]...]
    end

    @test RegisterMismatch.aperture_range((15,), (7,)) == (12:18,)
end

# Bias correction
@testset "Bias correction" begin
    val = NumDenom(2.0,0.5)
    # 2d
    mma = fill(val, 3, 3)
    for i = 1:3
        mma[2,i] = NumDenom(rand(),rand())
        mma[i,2] = NumDenom(rand(),rand())
    end
    mm = CenterIndexedArray(mma)
    RegisterMismatch.correctbias!(mm)
    for I in eachindex(mm)
        mmI = mm[I]
        @test mmI.num ≈ val.num
        @test mmI.denom ≈ val.denom
    end
    # 3d
    mma = fill(val, 3, 3, 3)
    for i = 1:3, j = 1:3
        mma[2,i,j] = NumDenom(rand(),rand())
        mma[i,2,j] = NumDenom(rand(),rand())
    end
    mm = CenterIndexedArray(mma)
    RegisterMismatch.correctbias!(mm)
    for I in eachindex(mm)
        mmI = mm[I]
        @test mmI.num ≈ val.num
        @test mmI.denom ≈ val.denom
    end
end

@testset "Mismatch" begin
    mdutils = nothing
    devlist = nothing
    @testset for threading in (true, false)
        RegisterMismatch.allow_inner_threading!(threading)
        @testset for imsz in ((7, 10), (6, 5))
            @testset for maxshift in ((4, 3), (3, 2))
                Apad = parent(ImageFiltering.padarray(reshape(1:prod(imsz), imsz[1], imsz[2]), Fill(0, maxshift, maxshift)))
                Bpad = parent(ImageFiltering.padarray(rand(1:20, imsz[1], imsz[2]), Fill(0, maxshift, maxshift)))
                for RM in RMlist
                    # intensity normalization
                    mm = RM.mismatch(Apad, Bpad, maxshift, normalization=:intensity)
                    num, denom = RegisterCore.separate(mm)
                    mmref = CenterIndexedArray{Float64}(undef, (2 .* maxshift .+ 1)...)
                    for j = -maxshift[2]:maxshift[2], i = -maxshift[1]:maxshift[1]
                        Bshift = circshift(Bpad, -[i, j])
                        df = Apad - Bshift
                        mmref[i, j] = sum(df .^ 2)
                    end
                    nrm = sum(Apad .^ 2) + sum(Bpad .^ 2)
                    @test ≈(mmref.data, num.data, atol=accuracy * nrm)
                    @test ≈(fill(nrm, size(denom)), denom.data, atol=accuracy * nrm)
                    # pixel normalization
                    mm = RM.mismatch(Apad, Bpad, maxshift, normalization=:pixels)
                    _, denom = RegisterCore.separate(mm)
                    n = Vector{Int}[size(Apad, i) .- abs.(-maxshift[i]:maxshift[i]) for i = 1:2]
                    @test ≈(denom.data, n[1] .* n[2]', atol=accuracy * maximum(denom))
                end
            end
        end
    end
end

# Test for denom overflow with mismatch0
@testset "Denom overflow" begin
    C16 = N0f16[0.6 0.1; 0.7 0.1]
    D16 = N0f16[0.7 0.1; 0.6 0.1]
    C = Float64.(C16)
    D = Float64.(D16)
    nd = RegisterMismatch.mismatch0(C16, D16; normalization=:intensity)
    @test nd.denom ≈ sum((C.^2).+(D.^2))
    @test nd.num ≈ sum((C.-D).^2)
end

# Compare direct, global, and apertured
@testset "Compare direct, global and apertured" begin
    C = rand(7,9)
    D = rand(7,9)
    mm = RegisterMismatch.mismatch(C, D, (3,3))
    nd = RegisterMismatch.mismatch0(C, D)
    @test mm[0,0].num ≈ nd.num
    @test mm[0,0].denom ≈ nd.denom
    mm = RegisterMismatch.mismatch(C, D, (3,3), normalization=:pixels)
    nd = RegisterMismatch.mismatch0(C, D, normalization=:pixels)
    @test mm[0,0].num ≈ nd.num
    @test mm[0,0].denom ≈ nd.denom

    mms = RegisterMismatch.mismatch_apertures(C, D, (2,2), (3,2), normalization=:intensity)
    nd0 = RegisterMismatch.mismatch0(C, D)
    nd1 = RegisterMismatch.mismatch0(mms)
    @test nd0.num ≈ nd1.num
    @test nd0.denom ≈ nd1.denom
end

# Now do it for block mismatch
# A key property we're testing here is that
#     sum(nums) == num
# where num is equivalent to what would be computed globally (using mismatch)
@testset "Block mismatch" begin
    for imsz in ((15,16), (14,17))
        for maxshift in ((4,3), (3,2))
            for gridsize in ((2,1), (2,3), (2,2), (1,3))
                Apad = parent(ImageFiltering.padarray(reshape(1:prod(imsz), imsz[1], imsz[2]), Fill(0, maxshift, maxshift)))
                Bpad = parent(ImageFiltering.padarray(rand(1:20, imsz[1], imsz[2]), Fill(0, maxshift, maxshift)))
                for RM in RMlist
                    # intensity normalization
                    mms = RM.mismatch_apertures(Float64, Apad, Bpad, gridsize, maxshift, normalization=:intensity, display=false)
                    nums, denoms = RegisterCore.separate(mms)
                    num = sum(nums)
                    denom = sum(denoms)
                    mm = CenterIndexedArray{Float64}(undef, (2 .* maxshift .+ 1)...)
                    for j = -maxshift[2]:maxshift[2], i = -maxshift[1]:maxshift[1]
                        Bshift = circshift(Bpad,-[i,j])
                        df = Apad-Bshift
                        mm[i,j] = sum(df.^2)
                    end
                    nrm = sum(Apad.^2)+sum(Bpad.^2)
                    @test ≈(mm.data, num.data, atol=accuracy*nrm)
                    @test ≈(fill(nrm,size(denom)), denom.data, atol=accuracy*nrm)
                    # pixel normalization
                    mms = RM.mismatch_apertures(Float64, Apad, Bpad, gridsize, maxshift, normalization=:pixels, display=false)
                    _, denoms = RegisterCore.separate(mms)
                    denom = sum(denoms)
                    n = Vector{Int}[size(Apad,i).-abs.(-maxshift[i]:maxshift[i]) for i = 1:2]
                    @test ≈(denom.data, n[1].*n[2]', atol=accuracy*maximum(denom))
                end
            end
        end
    end

    for RM in RMlist
        # Test 3d similarly
        Apad = parent(ImageFiltering.padarray(reshape(1:80*6, 10, 8, 6), Fill(0, (4,3,2))))
        Bpad = parent(ImageFiltering.padarray(rand(1:80*6, 10, 8, 6), Fill(0, (4,3,2))))
        mm = RM.mismatch(Apad, Bpad, (4,3,2))
        num, denom = RegisterCore.separate(mm)
        mmref = CenterIndexedArray{Float64}(undef, 9, 7, 5)
        for k=-2:2, j = -3:3, i = -4:4
            Bshift = circshift(Bpad,-[i,j,k])
            df = Apad-Bshift
            mmref[i,j,k] = sum(df.^2)
        end
        nrm = sum(Apad.^2)+sum(Bpad.^2)
        @test ≈(mmref.data, num.data, atol=accuracy*nrm)
        @test ≈(fill(nrm,size(denom)), denom.data, atol=accuracy*nrm)

        mms = RM.mismatch_apertures(Apad, Bpad, (2,3,2), (4,3,2), normalization=:intensity, display=false)
        nums, denoms = RegisterCore.separate(mms)
        num = sum(nums)
        denom = sum(denoms)
        @test ≈(mmref.data, num.data, atol=accuracy*nrm)
        @test ≈(fill(sum(Apad.^2)+sum(Bpad.^2),size(denom)), denom.data, atol=accuracy*nrm)
    end
end

### Test the utility functions that aren't implicitly tested by the above
# These only need to be tested once, since they come from RegisterMismatchCommon
@testset "Utility functions" begin
    RM = RegisterMismatch
    A = fill(7, 8, 4)
    Ahp = RM.highpass(Float32, A, (1,1))
    @test eltype(Ahp) == Float32
    @test maximum(abs.(Ahp)) < 100*eps(Float32)
    Ahp = RM.highpass(A, (1.2,Inf))
    @test A == Ahp
    A = Float64.(A)
    A[1,1] = NaN
    Ahp = RM.highpass(Float32, A, (1,1))
    @test sum(x->!isfinite(x), Ahp) == 1
    @test isnan(Ahp[1,1])
    Ahp[1,1] = 0
    @test maximum(abs, Ahp) < 3e-6


    num = float([1 2; 3 4])
    denom = [1 1e-6; 2 1]
    nd = RegisterCore._packnd!(similar(num, RegisterCore.NumDenom{Float64}), num, denom)
    RM.truncatenoise!(nd, 1e-5)
    numt, denomt = RegisterCore.separate(nd)
    @test numt == [1 0; 3 4]
    @test denomt == [1 0; 2 1]

    outer = reshape(1:120, 12, 10)
    A = outer[1:10,2:9]
    B = outer[3:12,1:8]
    maxshift = (3,11)
    @test register_translate(A, B, maxshift) == CartesianIndex((-2,1))
end

# Mismatched types
@testset "Mismatched types" begin
   A = rand(Float32, 5, 5)
    B = rand(5, 5)
    mm = RegisterMismatch.mismatch0(A, B)
    @test eltype(mm) == Float64
end

# Mismatched axes in the inputs (issue #7)
fullimg = collect(reshape(1:20, 4, 5))
fixed = fullimg[1:4, 2:5]
mxshift = (2, 2)
mm = RegisterMismatch.mismatch(fixed, fullimg, mxshift)
num, denom = RegisterCore.separate(mm)
mm = CenterIndexedArray{Float64}(undef, (2 .* mxshift .+ 1)...)
for j = -mxshift[2]:mxshift[2], i = -mxshift[1]:mxshift[1]
    mshift = OffsetArray(fullimg, (-i, -j))
    idx = intersect.(axes(fixed), axes(mshift))
    fsnip, msnip = fixed[idx...], mshift[idx...]
    mm[i,j] = sum((fsnip .- msnip).^2) / (sum(fsnip.^2) + sum(msnip.^2))
end
@test num ./ denom ≈ mm
