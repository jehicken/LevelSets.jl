"Implements smooth-max inspired level-set functions."
module LevelSets

using LinearAlgebra
using StaticArrays
using NearestNeighbors
using DualNumbers

export LevelSet
export evallevelset, snappoint!, findclosest!
export difflevelset!, hessianlevelset!

import Base.eps
eps(::Type{Complex{T}}) where T <: AbstractFloat = eps(T)
eps(::Complex{T}) where T <: AbstractFloat = eps(T)

"Data required by smooth-max level-set function."
mutable struct LevelSet{Dim,T<:Number}
    "Number of basis functions that define the level set."
    numbasis::Int
    "Centers for the basis functions."
    xcenter::Array{T,2}
    "Normal and tangent(s) at the centers."
    frame::Array{T,3}
    "Curvature at each point"
    kappa::Array{T,2}
    "Penalty parameter"
    rho::T
    "Parameter that smooths distance near zero"
    delta::T
    "tree used to accelerate the evaluation of the levelset"
    tree::KDTree{SVector{Dim,Float64}, Euclidean, Float64}

    function LevelSet{Dim,T}(xcenter::AbstractArray{T,2},
                             normal::AbstractArray{T,2},
                             tangents::AbstractArray{T,3},
                             kappa::AbstractArray{T,2},
                             rho::T) where {Dim,T<:Number}
        @assert(size(xcenter) == size(normal),
                "size of xcenter and normal do not match")
        @assert(size(xcenter,2) == size(tangents,3) && 
                size(xcenter,1) == size(tangents,1),
                "size of xcenter and tangents do not match")
        @assert(size(tangents,2) == Dim-1,
                "number of tangents is inconsistent with dimension")
        dim = size(xcenter,1)
        numbasis = size(xcenter,2) 
        xc = deepcopy(xcenter)
        kap = deepcopy(kappa)
        frm = zeros(T, Dim, Dim, numbasis)
        @inbounds for i = 1:numbasis 
            frm[:,1,i] = normal[:,i]/norm(normal[:,i])
            for j = 1:Dim-1
                frm[:,j+1,i] = tangents[:,j,i]
                for k = 1:j 
                    frm[:,j+1,i] -= dot(frm[:,j+1,i], frm[:,k,i])*frm[:,k,i]
                end
                frm[:,j+1,i] /= norm(frm[:,j+1,i])
            end
        end
        delta = 1e-10
        tree = KDTree(realpart.(xc))
        new(numbasis, xc, frm, kap, rho, delta, tree)
    end 
end

"""
    expfac = expdist(x, xc, rho, delta, min_dist)

Computes the exponential of the scaled distance from `x` to `xc`.  The distance 
is regularized by adding `delta` to the dot product of `x - xc`; this helps 
with derivatives when `norm(x - xc)` is small.  The scalar `-rho` scales the 
difference between this regularized distance and some reference `min_dist`.
"""
function expdist(x::AbstractArray{T,1}, xc::AbstractArray{T,1},
                 rho::T, delta::T, min_dist::T)::T where {T<:Number}
    dist = sqrt(dot(x - xc, x - xc) + delta)
    return exp(-rho*(dist - min_dist))
end

"""
    diffexpdist!(x_bar, x, xc, rho, delta, min_dist, exp_bar)

Computes the derivative of `expdist` with respect to `x` and returns the result
in the array `x_bar`.  The derivative is weighted by `exp_bar`.
"""
function diffexpdist!(x_bar::AbstractArray{T,1}, x::AbstractArray{T,1}, 
                      xc::AbstractArray{T,1}, rho::T, delta::T, min_dist::T, exp_bar::T) where {T<:Number}
    # forward sweep
    dist = sqrt(dot(x - xc, x - xc) + delta)
    # return exp(-rho*(dist - min_dist)) 
    dist_bar = -exp_bar*rho*exp(-rho*(dist - min_dist))
    # dist = sqrt(dot(x - xc, x - xc) + delta)
    x_bar[:] += (dist_bar/dist)*(x - xc)
end

"""
    hessexpdist!(hess, x, xc, rho, delta, min_dist)

Computes the Hessian of `expdist` with respect to `x` and returns the result
in the array `hess`.  
"""
function hessexpdist!(hess::AbstractArray{T,2}, x::AbstractArray{T,1}, 
                      xc::AbstractArray{T,1}, rho::T, delta::T,
                      min_dist::T) where {T<:Number}
    dx = x - xc # allocation!!!
    dist2 = dot(dx, dx) + delta
    dist = sqrt(dist2)
    expfac = exp(-rho*(dist - min_dist))
    hess[:,:] = (rho*expfac/dist2)*( (one(T)/dist + rho)*dx*dx' - dist*I )
end

"""
    ls_local = locallevelset(x, xc, frame, kappa)

Return a quadratic approximation to the level set based on given `xc`, `frame` 
coordinate axes, and curvatures `kappa`.
"""
function locallevelset(x::AbstractArray{T,1}, xc::AbstractArray{T,1},
                       frame::AbstractArray{T,2},
                       kappa::AbstractArray{T,1})::T where {T<:Number}
    perp = dot(frame[:,1], x - xc)
    for j = 1:size(x,1)-1
        perp += 0.5*kappa[j]*dot(frame[:,j+1], x - xc)^2
    end
    return perp
end

"""
    difflocallevelset!(xc_bar, frame_bar, kappa_bar, x, xc, frame, kappa, perp_bar)

Computes the dervatives of `locallevelset` with respect to `xc`, `frame`, and 
`kappa`.  The derivatives are returned in the `_bar` variables and weighted by 
`perp_bar`.
"""
function difflocallevelset!(xc_bar::AbstractArray{T,1},
                            frame_bar::AbstractArray{T,2},
                            kappa_bar::AbstractArray{T,1},
                            x::AbstractArray{T,1}, xc::AbstractArray{T,1},
                            frame::AbstractArray{T,2},
                            kappa::AbstractArray{T,1},
                            perp_bar::T) where {T<:Number}
    for j = 1:size(x,1)-1
        # perp += 0.5*kappa[j]*dot(frame[:,j+1], x - xc)^2
        dotprod = dot(frame[:,j+1], x - xc)
        frame_bar[:,j+1] += perp_bar*kappa[j]*dotprod*(x - xc)
        kappa_bar[j] += perp_bar*0.5*dotprod^2
        xc_bar[:] -= perp_bar*kappa[j]*dotprod*frame[:,j+1]
    end
    # perp = dot(frame[:,1], x - xc)
    xc_bar[:] -= perp_bar*frame[:,1]
    frame_bar[:,1] += perp_bar*(x - xc)
end

"""
    difflocallevelset!(x_bar, x, xc, frame, kappa, perp_bar)

Computes the dervatives of `locallevelset` with respect to `x`.  The 
derivatives are returned in `x_bar` and are weighted by `perp_bar`
"""
function difflocallevelset!(x_bar::AbstractArray{T,1},
                            x::AbstractArray{T,1}, xc::AbstractArray{T,1},
                            frame::AbstractArray{T,2},
                            kappa::AbstractArray{T,1},
                            perp_bar::T) where {T<:Number}
    for j = 1:size(x,1)-1
        # perp += 0.5*kappa[j]*dot(frame[:,j+1], x - xc)^2
        x_bar[:] += perp_bar*kappa[j]*dot(frame[:,j+1], x - xc)*frame[:,j+1]
    end
    # perp = dot(frame[1], x - xc)
    x_bar[:] += perp_bar*frame[:,1]
end

"""
    hesslocallevelset!(hess, x, xc, frame, kappa)

Computes the Hessian of `locallevelset` with repsect to `x`.  Note that `xc` and
`x` are provided but not actually used; quadatic level-set is constant.  We'll
keep this interface in case higher-order local level-sets are used in the 
future.
"""
function hesslocallevelset!(hess::AbstractArray{T,2},
                            x::AbstractArray{T,1}, xc::AbstractArray{T,1},
                            frame::AbstractArray{T,2},
                            kappa::AbstractArray{T,1}) where {T<:Number}
    fill!(hess, zero(T))
    for j = 1:size(x,1)-1
        # x_bar[:] += perp_bar*kappa[j]*dot(frame[:,j+1], x - xc)*frame[:,j+1]
        hess[:,:] += kappa[j]*frame[:,j+1]*frame[:,j+1]'
    end
end

function getnearest(x::AbstractArray{T,1}, levset::LevelSet{Dim,T}
                    )::Tuple{Vector{Int64}, T} where {Dim,T<:Number}
    k = min(10, levset.numbasis)
    nearest, dists = knn(levset.tree, realpart.(x), k)
    return nearest, minimum(dists)
end 

"""
    ls = evallevelset(x, levset)

Return the level-set value at `x` defined by `levset`.
"""
function evallevelset(x::AbstractArray{T,1},
                      levset::LevelSet{Dim,T})::T where {Dim,T<:Number}
    @assert(size(x,1) == Dim, "x inconsistent with levset")
    numer = zero(T)
    denom = zero(T)
    nearest, min_dist = getnearest(x, levset)
    #range = -log(eps(T))/levset.rho
    #nearest = inrange(levset.tree, x, range)
    #min_dist = 1e100
    #@inbounds for i in nearest
    #    xc = view(levset.xcenter, :, i)
    #    dist = sqrt(dot(x - xc, x - xc) + levset.delta)
    #    min_dist = min(min_dist, dist)
    #end 
    #@inbounds for i = 1:levset.numbasis 
    #    xc = view(levset.xcenter, :, i)
    #    dist = sqrt(dot(x - xc, x - xc) + levset.delta)
    #    min_dist = min(min_dist, dist)
    #end
    @inbounds for i = 1:levset.numbasis # for i in nearest 
        xc = view(levset.xcenter, :, i)
        frm = view(levset.frame, :, :, i)
        crv = view(levset.kappa, :, i)
        perp = locallevelset(x, xc, frm, crv)
        expfac = expdist(x, xc, levset.rho, levset.delta, min_dist)
        numer += perp*expfac
        denom += expfac
    end
    ls = numer/denom
    return ls
end

"""
    snappoint!(x, x0, levset[, tol=1e-12, max_newton=20])

Find a point on the level set that is "close" to `x0`.  The solution, `x`, 
satisfies `|phi(x)| < tol`.  However, `x` may not be the point on the level set 
that is closest to `x0`.
"""
function snappoint!(x::AbstractArray{T,1}, x0::AbstractArray{T,1},
                    levset::LevelSet{Dim,T}; tol::Float64=1e-12,
                    max_newton::Int=20, silent_fail::Bool=false
                    ) where {Dim,T<:Number}
    @assert(size(x,1) == Dim, "x inconsistent with levset")
    @assert(size(x0,1) == Dim, "x inconsistent with levset")
    x[:] = x0
    x_bar = zero(x)
    phi = evallevelset(x, levset)
    for n = 1:max_newton 
        #println("iter: ",n,": phi = ",phi)
        if abs(phi) < tol 
            return 
        end
        difflevelset!(x_bar, x, levset)
        x[:] -= phi*x_bar/norm(x_bar)
        phi = evallevelset(x, levset)
    end
    println("Newton failed to converge in snappoint!")
    println("x0 = ",x0,": x = ",x,": phi = ",phi)
    if silent_fail return end
    println("x0 = ",x0,": x = ",x,": phi = ",phi)
    error("Newton failed to converge in snappoint!")
end 

function findclosest!(x::AbstractArray{T,1}, x0::AbstractArray{T,1},
                      levset::LevelSet{Dim,T}; tol::Float64=1e-12,
                      max_newton::Int=10) where {Dim,T<:Number}
    nearest, min_dist = nn(levset.tree, x0)
    x[:] = levset.xcenter[:, nearest]
    phi = evallevelset(x, levset)
    grad_phi = zeros(Dim)
    for k = 1:max_newton 
        if abs(phi) < tol 
            return true 
        end 
        difflevelset!(grad_phi, x, levset)
        norm2_grad_phi = norm(grad_phi)^2
        x[:] = x0 + grad_phi*(dot(grad_phi, x - x0) - phi)/norm2_grad_phi 
        phi = evallevelset(x, levset)
    end 
    # if we get here, convergence failed 
    return false
end

# function findclosest!(x::AbstractArray{T,1}, x0::AbstractArray{T,1},
#                       levset::LevelSet{Dim,T}; tol::Float64=1e-12,
#                        max_newton::Int=10) where {Dim,T<:Number}
#     @assert(size(x,1) == Dim, "x inconsistent with levset")
#     @assert(size(x0,1) == Dim, "x inconsistent with levset")
#     # use the closest center as the initial guess
#     min_dist = 1e100
#     min_idx = -1
#     @inbounds for i = 1:levset.numbasis 
#         xc = view(levset.xcenter, :, i)
#         dist = sqrt(dot(x0 - xc, x0 - xc) + levset.delta)
#         if dist < min_dist 
#             min_dist = dist 
#             min_idx = i 
#         end
#     end
#     x[:] = levset.xcenter[:,min_idx]
#     step_max = sqrt(dot(x-x0, x-x0) - dot(levset.frame[:,1,min_idx], x - x0)^2)
#     step_max *= 0.25
#     println("x init = ",x[:],": step_max = ",step_max)
#     x_bar = zero(x)
#     A = zeros(T, (Dim+1, Dim+1))
#     for i = 1:Dim
#         A[i,i] = one(T)
#     end
#     b = zeros(T, (Dim+1))
#     # Solve the constrained minimization problem
#     hess = view(A, 1:Dim, 1:Dim)
#     xc = view(levset.xcenter, :, min_idx)
#     frm = view(levset.frame, :, :, min_idx)
#     crv = view(levset.kappa, :, min_idx)
#     lambda = zero(T)
#     max_line_search = 10
#     for n = 1:max_newton
#         # check for convergence
#         fill!(x_bar, zero(T))
#         difflevelset!(x_bar, x, levset)
#         phi = evallevelset(x, levset)
#         if n == 1
#             lambda = dot(x_bar, x0 - x)/dot(x_bar, x_bar)
#         end
#         b[1:Dim] = x0 - x - lambda*x_bar
#         b[end] = -phi
#         println("iter: ",n,": norm(dLdx) = ",norm(b[1:Dim]),": phi = ",-b[end])
#         if norm(b[1:Dim]) < tol && abs(phi) < tol 
#             return
#         end
#         fill!(A, zero(T))
#         hessianlevelset!(hess, x, levset)
#         hess[:,:] *= lambda
#         #A[1:Dim,1:Dim] += I
#         for i = 1:Dim
#             hess[i,i] += one(T)
#         end 
#         A[1:Dim,end] = x_bar
#         A[end,1:Dim] = x_bar
#         sol = A\b            
#         if norm(sol) > step_max
#             sol *= step_max/norm(sol)
#         end
#         x[:] += sol[1:Dim]
#         lambda += sol[end]
#     end
#     error("Newton failed to converge in findclosest")
# end

"""
    ls, bound = boundlevelset(x, dx, levset)

Return level-set value and bound defined by `levset`, over the element centered
at `x` and with half side lengths `dx`.
"""
function boundlevelset(x::AbstractArray{T,1}, dx::AbstractArray{T,1},
                       levset::LevelSet{Dim,T}) where {Dim,T<:Number}
    @assert(size(x,1) == Dim, "x inconsistent with levset")
    @assert(size(dx,1) == Dim, "dx inconsistent with levset")
    numer = zero(T)
    denom = zero(T)
    nearest, min_dist = getnearest(x, levset)
    # evaluate the basis functions at the point x
    work = zeros(T, levset.numbasis) # should be in levset to avoid allocations 
    @inbounds for i = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        frm = view(levset.frame, :, :, i)
        crv = view(levset.kappa, :, i)
        perp = locallevelset(x, xc, frm, crv)
        expfac = expdist(x, xc, levset.rho, levset.delta, min_dist)
        work[i] = expfac 
        numer += perp*expfac
        denom += expfac
    end
    ls = numer/denom
    L = norm(dx)
    factor = exp(2.0*levset.rho*L)
    converge_bound = 0.0
    tight_bound = 0.0
    @inbounds for i = 1:levset.numbasis 
        xc = view(levset.xcenter, :, i)
        frm = view(levset.frame, :, :, i)
        crv = view(levset.kappa, :, i)
        # finish computing psi_i at x 
        work[i] /= denom 

        # get various bounds on psi_i 
        psi_upper = min(1.0, work[i]*factor)
        psi_lower = work[i]/factor
        dpsi = max(psi_upper - work[i], work[i] - psi_lower)

        # get local distance function value and bound on difference 
        perp = locallevelset(x, xc, frm, crv)
        dperp = L 
        for j = 1:Dim-1 
            dperp += abs(crv[j])*(L*abs(dot(frm[:,j+1], x - xc)) + 0.5*L^2)
        end

        # Add to the convergent bound 
        converge_bound += abs(perp)*dpsi + psi_upper*dperp 

        # Add to the tight bound 
        tight_bound += (abs(perp - ls) + dperp)*psi_upper
    end

    return ls, min(converge_bound, tight_bound)
end

"""
    ls, bound = boundlevelset2(x, dx, levset)

Return level-set value and bound defined by `levset`, over the element centered
at `x` and with half side lengths `dx`.
"""
function boundlevelset2(x::AbstractArray{T,1}, dx::AbstractArray{T,1},
                        levset::LevelSet{Dim,T}) where {Dim,T<:Number}
    @assert(size(x,1) == Dim, "x inconsistent with levset")
    @assert(size(dx,1) == Dim,"dx inconsistent with levset")
    min_dist = 1e100
    @inbounds for i = 1:levset.numbasis 
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        min_dist = min(min_dist, dist)
    end
    # evaluate the basis functions at the point x
    work = zeros(T, levset.numbasis) # should be in levset to avoid allocations 
    numer = zero(T)
    denom = zero(T)
    @inbounds for i = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        perp = dot(levset.normal[:,i], x - xc)
        perp += 0.5*levset.kappa[1,i]*(x - xc)'*(I - levset.normal[:,i]*levset.normal[:,i]')*(x - xc)
        expfac = exp(-levset.rho*(dist - min_dist))
        work[i] = expfac
        numer += perp*expfac
        denom += expfac
    end 
    ls = numer/denom # level-set value at center of element 
    # get vertices of element
    vtx = zeros(T, levset.dim, 2^levset.dim) 
    vtx[:,:] .= x
    for idx = 0:2^levset.dim-1
        for d = 1:levset.dim
            if idx & (1 << (d-1)) != 0
                vtx[d,idx+1] += dx[d]
            else
                vtx[d,idx+1] -= dx[d]
            end
        end
    end
    # find min/max distances to each basis, and compute bound contributions 
    lower = 0.0 #1e100 
    upper = -1e100
    bound = 0.0 
    L = norm(dx)
    factor = exp(2.0*levset.rho*L)
    @inbounds for i = 1:levset.numbasis
        # finish computing psi_i at x 
        work[i] /= denom 
        xc = view(levset.xcenter, :, i)
        # which vertex is closest, which furthest 
        d_min = 1e100
        d_max = -1e100
        # WARNING: the following is only valid for linear (kappa = 0)
        for idx = 1:size(vtx,2)
            perp = dot(levset.normal[:,i], vtx[:,idx] - xc)
            d_min = min(d_min, perp) 
            d_max = max(d_max, perp)
        end
        #bound += max(d_max - ls, -d_min + ls)*min(work[i]*factor, 1)
        #lower = min(lower, d_min) 
        upper = max(upper, d_max)
        if d_min < 0.0 
            lower += d_min*min(work[i]*factor,1)
        else 
            lower += d_min*work[i]/factor
        end 
        perp = locallevelset(x, xc, frm, crv)
        dpsi = max( min(1.0, work[i]*fac) - work[i], work[i] - work[i]/fac)
        bound += abs(perp)*dpsi + work[i]*L 
        # if d_max < 0.0 
        #     upper += d_max*work[i]/factor
        # else 
        #     upper += d_max*min(work[i]*factor,1)
        # end
    end
    return ls, bound, bound # lower, upper 
end 

"""
    difflevelset!(xcenter_bar, frame_bar, kappa_bar, rho_bar, x, levset)

Compute the derivatives of the level set, defined by `levset`, at point `x`
with respect to the `LevelSet` parameters.

Uses the reverse-mode of algorithmic differentiation (i.e. back propagation).
"""
function difflevelset!(xcenter_bar::AbstractArray{T,2},
                       frame_bar::AbstractArray{T,3},
                       kappa_bar::AbstractArray{T,2},
                       rho_bar::AbstractArray{T,1},
                       x::AbstractArray{T,1},
                       levset::LevelSet{Dim,T}) where {Dim,T<:Number}
    @assert(size(xcenter_bar) == size(levset.xcenter),
            "xcenter_bar and levset.xcenter have inconsistent dimensions")
    @assert(size(frame_bar) == size(levset.frame),
            "frame_bar and levset.frame have inconsistent dimensions")
    @assert(size(kappa_bar) == size(levset.kappa),
            "kappa_bar and levset.kappa have inconsistent dimensions")
    @assert(size(rho_bar,1) == 1,
            "rho_bar should be a single element 1D array")
    @assert(size(x,1) == Dim, "x inconsistent with levset")
    # forward sweep 
    numer = zero(T)
    denom = zero(T)
    nearest, min_dist = getnearest(x, levset)
    #min_dist = 1e100
    #@inbounds for i = 1:levset.numbasis 
    #    xc = view(levset.xcenter, :, i)
    #    dist = sqrt(dot(x - xc, x - xc) + levset.delta)
    #    min_dist = min(min_dist, dist)
    #end
    @inbounds for i in nearest # = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        frm = view(levset.frame, :, :, i)
        crv = view(levset.kappa, :, i)
        perp = locallevelset(x, xc, frm, crv)
        expfac = expdist(x, xc, levset.rho, levset.delta, min_dist)
        numer += perp*expfac
        denom += expfac
    end
    ls = numer/denom
    # return ls
    ls_bar = one(T)
    # ls = numer/denom 
    numer_bar = ls_bar/denom 
    denom_bar = -ls_bar*ls/denom
    @inbounds for i in nearest # = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        xc_bar = view(xcenter_bar, :, i)
        frm = view(levset.frame, :, :, i)
        frm_bar = view(frame_bar, :, :, i)
        crv = view(levset.kappa, :, i)
        crv_bar = view(kappa_bar,: , i)
        perp = locallevelset(x, xc, frm, crv)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        expfac = exp(-levset.rho*(dist - min_dist)) 
        # denom += expfac 
        expfac_bar = denom_bar
        # numer += perp*expfac
        perp_bar = numer_bar*expfac 
        expfac_bar += numer_bar*perp
        # TODO: differentiate expdist w.r.t. xc and rho.
        # expfac = exp(-levset.rho*(dist - min_dist))
        rho_bar[1] -= expfac_bar*expfac*dist 
        dist_bar = -expfac_bar*expfac*levset.rho 
        # perp = locallevelset(x, xc, frm, crv)
        difflocallevelset!(xc_bar, frm_bar, crv_bar, x, xc, frm, crv, perp_bar)
        # dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        xc_bar[:] -= dist_bar.*(x - xc)./dist 
    end
    return nothing
end

"""
    difflevelset!(x_bar, x, levset)

Compute the derivatives of the level set, defined by `levset`, at point `x`
with respect to `x`.  The derivatives are stored in `x_bar`.

Uses the reverse-mode of algorithmic differentiation (i.e. back propagation).
"""
function difflevelset!(x_bar::AbstractArray{T,1}, x::AbstractArray{T,1},
                       levset::LevelSet{Dim,T}) where {Dim,T<:Number}
    @assert(size(x_bar) == size(x),
            "x_bar and x have inconsistent dimensions")
    @assert(size(x,1) == Dim, "x inconsistent with levset")
    # forward sweep 
    numer = zero(T)
    denom = zero(T)
    nearest, min_dist = getnearest(x, levset)
    #min_dist = 1e100
    #@inbounds for i = 1:levset.numbasis 
    #    xc = view(levset.xcenter, :, i)
    #    dist = sqrt(dot(x - xc, x - xc) + levset.delta)
    #    min_dist = min(min_dist, dist)
    #end
    @inbounds for i = 1:levset.numbasis #in nearest # = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        frm = view(levset.frame, :, :, i)
        crv = view(levset.kappa, :, i)
        perp = locallevelset(x, xc, frm, crv)
        expfac = expdist(x, xc, levset.rho, levset.delta, min_dist)
        numer += perp*expfac
        denom += expfac
    end
    ls = numer/denom
    # start reverse sweep
    fill!(x_bar, zero(T))
    # return ls
    ls_bar = one(T)
    # ls = numer/denom 
    numer_bar = ls_bar/denom 
    denom_bar = -ls_bar*ls/denom
    @inbounds for i = 1:levset.numbasis # in nearest # = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        frm = view(levset.frame, :, :, i)
        crv = view(levset.kappa, :, i)
        perp = locallevelset(x, xc, frm, crv)
        expfac = expdist(x, xc, levset.rho, levset.delta, min_dist)
        # denom += expfac
        expfac_bar = denom_bar
        # numer += perp*expfac 
        perp_bar = numer_bar*expfac 
        expfac_bar += numer_bar*perp
        # expfac = expdist(x, xc, levset.rho, levset.delta, min_dist)
        diffexpdist!(x_bar, x, xc, levset.rho, levset.delta, min_dist,
                     expfac_bar)
        # perp = locallevelset(x, xc, frm, crv)
        difflocallevelset!(x_bar, x, xc, frm, crv, perp_bar)        
    end
    return nothing
end

"""
    hessianlevelset!(hess, x, levset)

Compute the Hessian of the level set, defined by `levset`, at point `x`
with respect to `x`.  The derivatives are stored in `hess`.
"""
function hessianlevelset!(hess::AbstractArray{T,2}, x::AbstractArray{T,1},
                          levset::LevelSet{Dim,T}) where {Dim,T<:Number}
    @assert(size(hess,1) == size(hess,2) == size(x,1),
        "hess and x have inconsistent dimensions")
    @assert(size(x,1) == Dim, "x inconsistent with levset")
    min_dist = 1e100
    @inbounds for i = 1:levset.numbasis 
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        min_dist = min(min_dist, dist)
    end
    numer = zero(T)
    denom = zero(T)
    @inbounds for i = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        frm = view(levset.frame, :, :, i)
        crv = view(levset.kappa, :, i)
        perp = locallevelset(x, xc, frm, crv)
        expfac = expdist(x, xc, levset.rho, levset.delta, min_dist)
        numer += perp*expfac
        denom += expfac
    end
    ls = numer/denom
    # start reverse sweep; here we compute both the first derivative, plus some 
    # local contributions to the second derivative 
    x_bar = zero(x) # <--- holds the first derivative of the whole level set
    deriv_denom = zero(x) # derivative of denom with respect to x
    deriv_local = zero(x) # work vector
    deriv_expfac = zero(x) # work vector
    hess_denom = zeros(T, Dim, Dim) # Hessian of denom with respect to x
    hess_local = zero(hess_denom)
    hess_expfac = zero(hess_denom)
    # return ls
    ls_bar = one(T)
    # ls = numer/denom 
    numer_bar = ls_bar/denom 
    denom_bar = -ls_bar*ls/denom
    fill!(hess, zero(T))
    @inbounds for i = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        frm = view(levset.frame, :, :, i)
        crv = view(levset.kappa, :, i)
        perp = locallevelset(x, xc, frm, crv)
        expfac = expdist(x, xc, levset.rho, levset.delta, min_dist)
        # denom += expfac
        expfac_bar = denom_bar
        # numer += perp*expfac 
        perp_bar = numer_bar*expfac 
        expfac_bar += numer_bar*perp
        # expfac = expdist(x, xc, levset.rho, levset.delta, min_dist)
        diffexpdist!(x_bar, x, xc, levset.rho, levset.delta, min_dist,
                     expfac_bar)
        # perp = locallevelset(x, xc, frm, crv)
        difflocallevelset!(x_bar, x, xc, frm, crv, perp_bar)
        
        # the following computes local contributions to second derivative
        fill!(deriv_local, zero(T))
        fill!(deriv_expfac, zero(T))
        difflocallevelset!(deriv_local, x, xc, frm, crv, one(T))
        hesslocallevelset!(hess_local, x, xc, frm, crv)
        diffexpdist!(deriv_expfac, x, xc, levset.rho, levset.delta, min_dist,
                     one(T))
        hessexpdist!(hess_expfac, x, xc, levset.rho, levset.delta, min_dist)
        hess[:,:] += (hess_local*expfac + deriv_local*deriv_expfac' + 
                      deriv_expfac*deriv_local' + perp*hess_expfac)

        # the following adds contributions to the derivatives of denom 
        deriv_denom += deriv_expfac
        hess_denom += hess_expfac 
    end
    hess[:,:] -= deriv_denom*x_bar'
    hess[:,:] -= x_bar*deriv_denom'
    hess[:,:] -= hess_denom*ls 
    hess[:,:] /= denom
end

"""
    residual!(res, x, levset)

Evaluate the level-set defined by `levset` at points `x`, and store in `res`
"""
function residual!(res::AbstractArray{T,1}, x::AbstractArray{T,2},
                   levset::LevelSet{Dim,T}) where {Dim,T<:Number}
    @assert(size(res,1) == size(x,2), "res and x have inconsistent dimensions")
    @assert(size(x,1) == Dim, "x inconsisent with levset")
    @inbounds for j = 1:size(x,2)
        res[j] = evallevelset(view(x,:,j), levset)
    end
    return nothing
end

end # module LevelSets
