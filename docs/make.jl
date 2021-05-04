using LevelSets
using Documenter

DocMeta.setdocmeta!(LevelSets, :DocTestSetup, :(using LevelSets); recursive=true)

makedocs(;
    modules=[LevelSets],
    authors="Jason Hicken <hickej2@rpi.edu> and contributors",
    repo="https://github.com/jehicken/LevelSets.jl/blob/{commit}{path}#{line}",
    sitename="LevelSets.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jehicken.github.io/LevelSets.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jehicken/LevelSets.jl",
    devbranch="main",
)
