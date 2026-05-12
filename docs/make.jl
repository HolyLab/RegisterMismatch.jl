using Documenter
using RegisterMismatch
using RegisterCore
using RegisterMismatchCommon
using FFTW

DocMeta.setdocmeta!(RegisterMismatch, :DocTestSetup, :(using RegisterMismatch, RegisterCore, FFTW); recursive = true)
DocMeta.setdocmeta!(RegisterMismatchCommon, :DocTestSetup, :(using RegisterMismatchCommon); recursive = true)
DocMeta.setdocmeta!(RegisterCore, :DocTestSetup, :(using RegisterCore); recursive = true)

makedocs(;
    modules = [RegisterMismatch, RegisterMismatchCommon, RegisterCore],
    sitename = "RegisterMismatch.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://HolyLab.github.io/RegisterMismatch.jl",
        repolink = "https://github.com/HolyLab/RegisterMismatch.jl",
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
    warnonly = false,
    # upstream packages are installed (not dev'd), so Documenter cannot determine
    # their git commit; disable source links to avoid MissingRemoteError
    remotes = nothing,
)

deploydocs(;
    repo = "github.com/HolyLab/RegisterMismatch.jl",
    devbranch = "master",
    push_preview = true,
)
