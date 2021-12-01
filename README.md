# LevelSets

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jehicken.github.io/LevelSets.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jehicken.github.io/LevelSets.jl/dev)
[![Build Status](https://github.com/jehicken/LevelSets.jl/workflows/CI/badge.svg)](https://github.com/jehicken/LevelSets.jl/actions)
[![Coverage](https://codecov.io/gh/jehicken/LevelSets.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jehicken/LevelSets.jl)

![](wreath.png "Level-set example" )

This package implements the level-set formula described in Hicken J. and Kaur S., "_An Explicit Level-Set Formula to Approximate Geometries_," Proceedings of the AIAA 2022 SciTech Forum.  Please see the paper for further details.

The tests in `test/runtestss.jl` offer some examples of how to use the package; however, the following module provides a model application to the ellipsoid.  You will need the `WriteVTK` and `PyPlot` packages to run the functions in this example.

```julia
module Ellipsoid

using WriteVTK
using LinearAlgebra
using LevelSets
using PyPlot

export plotEllipsoid, ellipsoidConverge

"""
    kappa1, kappa2, lambda1, lambda2 = principals(E, F, G, L, M, N)

Compute the principal curvatures, `kappa1` and `kappa2`, and principal 
directions, `lambda1` and `lambda2`, using the first- and second-
fundamental forms.
"""
function principals(E::Float64, F::Float64, G::Float64, L::Float64,
                    M::Float64, N::Float64, )
    denom_fac = 1/(E*G - F^2)
    K = (L*N - M^2)*denom_fac 
    H = (E*N + G*L -2*F*M)*denom_fac/2.0
    kappa1 = H + sqrt(H^2 - K)
    kappa2 = H - sqrt(H^2 - K)
    lambda1 = ones(2)
    lambda2 = ones(2)
    tol = 1e-10
    if abs(N - kappa1*G) > tol
        lambda1[2] = - (M - kappa1*F)/(N - kappa1*G)
    elseif abs(M - kappa1*F) > tol
        lambda1[2] = - (L - kappa1*E)/(M - kappa1*F)
    else 
        lambda1[2] = 1.0
        lambda1[1] = 0.0
    end 
    if abs(N - kappa2*G) > tol
        lambda2[2] = - (M - kappa2*F)/(N - kappa2*G)
    elseif abs(M - kappa2*F) > tol
        lambda2[2] = - (L - kappa2*E)/(M - kappa2*F)
    else 
        lambda2[2] = 1.0
        lambda2[1] = 0.0
    end
    return kappa1, kappa2, lambda1, lambda2
end

"""
    plotEllipsoid(numu, numv, rhofactor[, quad=false, a=2.0, b=0.5, c=1.0])

Create a LSF for the ellipsoid (with axes lengths `a`, `b`, and `c`), and save a VTK file of it for visualization. 
"""
function plotEllipsoid(numu::Int, numv::Int, rhofactor::Float64; 
                       quad::Bool=false, a::Float64=2.0, b::Float64=0.5, 
                       c::Float64=1.0)
    
    # sample the ellipsoid to get the centers and frame 
    numbasis = numu*numv 
    xc = zeros(3, numbasis)
    nrm = zero(xc)
    tang = zeros(3, 2, numbasis)
    crv = zeros(2, numbasis)
    min_min = 1e100
    max_max = 0.0
    max_ratio = 0.0
    for i = 1:numu 
        u = 2*pi*(i-0.5)/numu
        for j = 1:numv 
            v = pi*(j-0.5)/numv 
            n = numv*(i-1) + j
            xc[:, n] = [a*cos(u)*sin(v); b*sin(u)*sin(v); -c*cos(v)]
            du = [-a*sin(u)*sin(v); b*cos(u)*sin(v); 0.0]
            dv = [a*cos(u)*cos(v); b*sin(u)*cos(v); c*sin(v)]
            du2 = [-a*cos(u)*sin(v); -b*sin(u)*sin(v); 0.0]
            dv2 = [-a*cos(u)*sin(v); -b*sin(u)*sin(v); c*cos(v)]
            dudv = [-a*sin(u)*cos(v); b*cos(u)*cos(v); 0.0]
            nrm[:, n] = cross(du, dv)
            fac = 1/norm(nrm[:,n])
            # compute the terms for the first and second fundamental forms 
            E = dot(du, du)
            F = dot(du, dv) 
            G = dot(dv, dv) 
            L = dot(du2, nrm[:,n])*fac
            M = dot(dudv, nrm[:,n])*fac 
            N = dot(dv2, nrm[:,n])*fac 
            # compute the principal curvatures/directions
            kappa1, kappa2, lambda1, lambda2 = principals(E, F, G, L, M, N)
            tang[:,1,n] = lambda1[1]*du + lambda1[2]*dv 
            tang[:,2,n] = lambda2[1]*du + lambda2[2]*dv
            println("E, F, G, L, M, N = ",E,", ",F,", ",G,", ",L,", ",M,", ",N)
            println("tang1 dot tang2 = ", dot(tang[:,1,n], tang[:,2,n]))
            @assert( abs(dot(tang[:,1,n], tang[:,2,n])) < 1e-9,
            "tangents are not orthogonal!" )
            if quad 
                crv[1,n] = -kappa1 
                crv[2,n] = -kappa2
                global max_ratio = max(max_ratio, abs(kappa1/kappa2), abs(kappa2/kappa1))
                global min_min = min(min_min, abs(kappa1), abs(kappa2))
                global max_max = max(max_max, abs(kappa1), abs(kappa2))
            else 
                crv[1,n] = 0.0
                crv[2,n] = 0.0
            end
        end 
    end 
    println("miniumum prin. curvature = ", min_min)
    println("maximum prin. curvature = ",max_max)
    println("max curvature ratio = ",max_ratio)
    
    # construct the LevelSet 
    rho = rhofactor*convert(Float64, numu)
    levset = LevelSet{3,Float64}(xc, nrm, tang, crv, rho)
    
    # write the centers to a file 
    centers = Vector{MeshCell{VTKCellType, Vector{Int64}}}(undef, levset.numbasis)
    for i = 1:levset.numbasis 
        centers[i] = MeshCell(VTKCellTypes.VTK_VERTEX, [i])
    end
    normal = view(levset.frame, :, 1, :)
    saved_files = vtk_grid("centers.vtu", levset.xcenter, centers) do vtk
        vtk["normal"] = normal
    end
    
    # Next, we make a fine mesh in (u,v) space for plotting in Paraview 
    numu = 1200
    numv = 1200
    numvert = numu*numv
    function getvertex(i, j)
        return numv*(j-1) + i
    end
    
    # get the points making up the vtk triangles
    points = zeros(3, numvert)
    for i = 1:numu 
        u = 2*pi*(i-1)/(numu-1)
        for j = 1:numv 
            v = pi*(j-1)/(numv-1)
            n = getvertex(i, j)
            x0 = [a*cos(u)*sin(v); b*sin(u)*sin(v); c*cos(v)]
            snappoint!(view(points,:,n), x0, levset, tol=1e-8, max_newton=50, 
            silent_fail=true)
        end
    end
    
    # define the triangles 
    numcell = 2*(numu-1)*(numv-1)
    cells = Vector{MeshCell{VTKCellType, Vector{Int64}}}(undef, numcell)
    for i = 1:numu-1
        for j = 1:numv-1
            n = (numv-1)*(j-1) + i
            connect = [getvertex(i,j), getvertex(i+1,j), getvertex(i,j+1)]
            cells[n] = MeshCell(VTKCellTypes.VTK_TRIANGLE, connect)
            n += (numu-1)*(numv-1)
            connect = [getvertex(i+1,j+1), getvertex(i,j+1), getvertex(i+1,j)]
            cells[n] = MeshCell(VTKCellTypes.VTK_TRIANGLE, connect)
        end
    end
    
    # save the vtk unstructured grid file
    saved_files = vtk_grid("ellipse.vtu", points, cells) do vtk
        # add datasets here...nothing to add in this case
    end
    
end

"""
    ellipsoidConverge([a=2.0, b=0.5, c=1.0])

Run the ellipsoid convergence study from the paper.
"""
function ellipsoidConverge(;a::Float64=2.0, b::Float64=0.5, c::Float64=1.0)

    # set the points used to measure error 
    numu, numv = 1200, 1200
    numpts = numu*numv
    x = zeros(3, numpts)
    ptr = 1
    for i = 1:numu 
        u = 2*pi*(i-0.5)/numu
        for j = 1:numv 
            v = pi*(j-0.5)/numv            
            x[:,ptr] = [a*cos(u)*sin(v); b*sin(u)*sin(v); c*cos(v)]
            ptr += 1
        end
    end
    res = zeros(size(x,2))

    numu = [10, 20, 40, 80, 160]
    rhofactor = [1, 10]
    l2err = zeros(size(numu,1), size(rhofactor,1))
    l2err_ho = zeros(size(numu,1), size(rhofactor,1))
    for (k, nu) in enumerate(numu)
        # set basis centers and normals
        numbasis = nu*nu 
        xc = zeros(3, numbasis)
        nrm = zeros(3, numbasis)
        tang = zeros(3, 2, numbasis)
        crv = zeros(2, numbasis)
        ptr = 1
        for i = 1:nu
            u = 2*pi*(i-0.5)/nu
            for j = 1:nu
                v = pi*(j-0.5)/nu 
                xc[:, ptr] = [a*cos(u)*sin(v); b*sin(u)*sin(v); -c*cos(v)]
                du = [-a*sin(u)*sin(v); b*cos(u)*sin(v); 0.0]
                dv = [a*cos(u)*cos(v); b*sin(u)*cos(v); c*sin(v)]
                du2 = [-a*cos(u)*sin(v); -b*sin(u)*sin(v); 0.0]
                dv2 = [-a*cos(u)*sin(v); -b*sin(u)*sin(v); c*cos(v)]
                dudv = [-a*sin(u)*cos(v); b*cos(u)*cos(v); 0.0]
                nrm[:,ptr] = cross(du, dv)
                fac = 1/norm(nrm[:,ptr])
                # compute the terms for the first and second fundamental forms 
                E = dot(du, du)
                F = dot(du, dv)
                G = dot(dv, dv)
                L = dot(du2, nrm[:,ptr])*fac
                M = dot(dudv, nrm[:,ptr])*fac 
                N = dot(dv2, nrm[:,ptr])*fac 
                # compute principal curvatures/directions
                kappa1, kappa2, lambda1, lambda2 = principals(E, F, G, L, M, N)
                tang[:,1,ptr] = lambda1[1]*du + lambda1[2]*dv 
                tang[:,2,ptr] = lambda2[1]*du + lambda2[2]*dv
                crv[1,ptr] = -kappa1 
                crv[2,ptr] = -kappa2
                ptr += 1
            end 
        end

        for (j, r) in enumerate(rhofactor)
            rho = r*convert(Float64, nu)

            # linear approximation 
            levset = LevelSet{3,Float64}(xc, nrm, tang, zeros(size(crv)), rho)
            LevelSets.residual!(res, x, levset)
            l2err[k,j] = sqrt(dot(res,res)/numpts)
            println("Linear: L2 integral norm = ",l2err[k,j])

            # quadratic approximation 
            levset = LevelSet{3,Float64}(xc, nrm, tang, crv, rho)
            LevelSets.residual!(res, x, levset)
            l2err_ho[k,j] = sqrt(dot(res,res)/numpts)
            println("Quadratic: L2 integral norm = ",l2err_ho[k,j])
        end
    end

    PyPlot.figure(figsize=[6,2.5])

    linear_style = ["-ro", "--ro", ":ro"]
    quad_style = ["-ks", "--ks", ":ks"]
    for (j, r) in enumerate(rhofactor)
        label = L"linear: $\rho/\sqrt{n_{\Gamma}} = " * string(r) * L"$"
        PyPlot.plot(numu, l2err[:,j], linear_style[j], linewidth=1, 
                    markersize=4, label=label)
    end
    for (j, r) in enumerate(rhofactor)
        label = L"quadratic: $\rho/\sqrt{n_{\Gamma}} = " * string(r) * L"$"
        PyPlot.plot(numu, l2err_ho[:,j], quad_style[j], linewidth=1, 
                    markersize=4,label=label)
    end
    PyPlot.gca().axis([5, 200, 1e-7, 0.1])
    PyPlot.gca().set_position([0.12, 0.17, 0.5, 0.8])
    PyPlot.gca().grid(which="major", linestyle=":")
    PyPlot.xticks(fontsize=8)
    PyPlot.yticks(fontsize=8)
    PyPlot.xscale("log")
    PyPlot.yscale("log")
    PyPlot.xlabel(L"$\sqrt{n_{\Gamma}}$", labelpad=4, fontsize=10)
    PyPlot.ylabel("RMS Error", labelpad=12, fontsize=10)
    PyPlot.gca().legend(loc=(1.1, 0.0), handlelength=4, labelspacing=1,
                        fontsize=8)

    PyPlot.savefig("ellipsoid-converge.png", dpi=300, transparent=true)
end

end # module Ellipsoid 
```