"Implements smooth-max inspired level-set functions."
module LevelSets

using LinearAlgebra

export LevelSet, evallevelset, fitlevelset!

"Data required by smooth-max level-set function."
mutable struct LevelSet{T<:Number}
    "Number of spatial dimensions the level-set is embedded in."
    dim::Int 
    "Number of basis functions that define the level set."
    numbasis::Int
    "Centers for the basis functions."
    xcenter::Array{T,2}
    "Outward pointing normal at the centers."
    normal::Array{T,2}
    "Penalty parameter"
    rho::T
    "Parameter that smooths distance near zero"
    delta::T

    function LevelSet{T}(xcenter::AbstractArray{T,2},
                         normal::AbstractArray{T,2},
                         rho::T) where {T<:Number}
        @assert(size(xcenter) == size(normal),
                "size of xcenter and normal do not match")
        dim = size(xcenter,1)
        numbasis = size(xcenter,2) 
        xc = deepcopy(xcenter) 
        # normalize the vectors in normal
        nrm = deepcopy(normal)
        @inbounds for i = 1:numbasis
            nrm[:,i] /= norm(normal[:,i])
        end
        delta = 1e-10
        new(dim, numbasis, xc, nrm, rho, delta)
    end 
end

"""
    ls = evallevelset(x, levset)

Return the level-set value at `x` defined by `levset`.
"""
function evallevelset(x::AbstractArray{T,1},
                      levset::LevelSet{T})::T where {T<:Number}
    @assert(size(x,1) == size(levset.xcenter,1), "x inconsistent with levset")
    numer = zero(T)
    denom = zero(T)
    @inbounds for i = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        perp = dot(levset.normal[:,i], x - xc)
        # second order correction; need radius of curvature for this to work 
        #perp += (x - xc)'*(I - levset.normal[:,i]*levset.normal[:,i]')*(x - xc)/(2*0.05)
        expfac = exp(-levset.rho*dist)
        numer += perp*expfac
        denom += expfac
    end
    ls = numer/denom
    return ls
end

"""
    difflevelset!(xcenter_bar, normal_bar, rho_bar, x, levset)

Compute the derivatives of the level set, defined by `levset`, at point `x`
with respect to the `LevelSet` parameters.

Uses the reverse-mode of algorithmic differentiation (i.e. back propagation).
"""
function difflevelset!(xcenter_bar::AbstractArray{T,2},
                       normal_bar::AbstractArray{T,2},
                       rho_bar::AbstractArray{T,1},
                       x::AbstractArray{T,1},
                       levset::LevelSet{T}) where {T<:Number}
    @assert(size(xcenter_bar) == size(levset.xcenter),
            "xcenter_bar and levset.xcenter have inconsistent dimensions")
    @assert(size(normal_bar) == size(levset.normal),
            "normal_bar and levset.normal have inconsistent dimensions")
    @assert(size(rho_bar,1) == 1,
            "rho_bar should be a single element 1D array")
    @assert(size(x,1) == size(levset.xcenter,1), "x inconsistent with levset")
    # forward sweep 
    numer = zero(T)
    denom = zero(T)
    @inbounds for i = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        perp = dot(levset.normal[:,i], x - xc)
        expfac = exp(-levset.rho*dist)
        numer += perp*expfac
        denom += expfac
    end
    ls = numer/denom
    # return ls
    ls_bar = one(T)
    # ls = numer/denom 
    numer_bar = ls_bar/denom 
    denom_bar = -ls_bar*ls/denom
    @inbounds for i = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        perp = dot(levset.normal[:,i], x - xc)
        expfac = exp(-levset.rho*dist)
        # denom += expfac
        expfac_bar = denom_bar
        # numer += perp*expfac 
        perp_bar = numer_bar*expfac 
        expfac_bar += numer_bar*perp
        # expfac = exp(-rho*dist)
        rho_bar[1] -= expfac_bar*expfac*dist 
        dist_bar = -expfac_bar*expfac*levset.rho
        # perp = dot(levset.normal[:,i], x - xc)
        normal_bar[:,i] += perp_bar.*(x - xc)
        xc_bar = -perp_bar.*levset.normal[:,i] 
        # dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        xc_bar -= dist_bar.*(x - xc)./dist 
        #xc = view(levset.xcenter, :, i)
        xcenter_bar[:,i] += xc_bar
    end
    return nothing
end

"""
    residual!(res, x, levset)

Evaluate the level-set defined by `levset` at points `x`, and store in `res`
"""
function residual!(res::AbstractArray{T,1}, x::AbstractArray{T,2},
                   levset::LevelSet{T}) where {T<:Number}
    @assert(size(res,1) == size(x,2), "res and x have inconsistent dimensions")
    @assert(size(x,1) == size(levset.xcenter,1), "x inconsisent with levset")
    @inbounds for j = 1:size(x,2)
        res[j] = evallevelset(view(x,:,j), levset)
    end
    return nothing
end

"""
    xc_idx, nrm_idx, len_idx = parameterindices(levset)

Create indices for `xcenter`, `normal`, and `lenscale` parameters in `levset`.
"""
function parameterindices(levset::LevelSet{T}) where {T<:Number}
    xcenter_indices = zeros(UInt32, size(levset.xcenter))
    normal_indices = zeros(UInt32, size(levset.normal))
    lenscale_indices = zeros(UInt32, size(levset.lenscale))
    ptr::UInt32 = 1
    for i = 1:levset.numbasis 
        for d = 1:levset.dim
            xcenter_indices[d,i] = ptr
            ptr += 1
        end
        for d = 1:levset.dim 
            normal_indices[d,i] = ptr 
            ptr += 1
        end 
        lenscale_indices[i] = ptr 
        ptr += 1
    end 
    @assert(ptr-1 == (2*levset.dim + 1)*levset.numbasis)
    return xcenter_indices, normal_indices, lenscale_indices 
end 

"""
    jacobian!(jac, x, levset)

Evaluate the Jacobian of `residual!` at `x` with respect to `levset` parameters.
"""
function jacobian!(jac::AbstractArray{T,2}, x::AbstractArray{T,2},
                   levset::LevelSet{T}) where {T<:Number}
    @assert(size(jac,1) == size(x,2), "jac and x have inconsistent dimensions")
    @assert(size(jac,2) == levset.dim*levset.numbasis + 1, 
        "jac columns are inconsistent with levset")
    @assert(size(x,1) == size(levset.xcenter,1), "x inconsisent with levset")
    fill!(jac, zero(T))
    # work arrays for reverse-mode AD 
    xcenter_bar = zeros(T, size(levset.xcenter))
    normal_bar = zeros(T, size(levset.normal))
    rho_bar = zeros(T, 1)
    # indices for parameters 
    #xcenter_indices, normal_indices, lenscale_indices = parameterindices    (levset)
    xcenter_indices = 1:levset.dim*levset.numbasis 
    rho_index = levset.dim*levset.numbasis + 1
    @inbounds for j = 1:size(x,2)
        fill!(xcenter_bar, zero(T))
        fill!(normal_bar, zero(T))
        fill!(rho_bar, zero(T))
        difflevelset!(xcenter_bar, normal_bar, rho_bar, view(x,:,j), levset)
        for val in isnan.(xcenter_bar)
            if val == true
                println("found nan in xcenter_bar")
            end
        end 
        for (k, index) in enumerate(xcenter_indices)
            jac[j,index] += xcenter_bar[k] # note: xcenter_bar is a 2D array
        end 
        #for (k, index) in enumerate(normal_indices) 
        #    jac[j,index] += normal_bar[k] # note: normal_bar is a 2D array
        #end 
        #for (k, index) in enumerate(lenscale_indices)
        #    jac[j,index] += lenscale_bar[k] # note: lenscale_bar is a 2D array
        #end
        jac[j,rho_index] += rho_bar[1]
    end
    return nothing
end 

function fitlevelset!(levset::LevelSet{Float64}, x::AbstractArray{Float64,2};
                      tol_rel::Float64=1e-10, tol_abs::Float64=1e-12)
    @assert(size(x,1) == size(levset.xcenter,1), "x inconsisent with levset")
    numpts = size(x, 2)
    # indices for parameters 
    #xcenter_indices, normal_indices, lenscale_indices = parameterindices(levset)

    # adds du to appropriate parameters in levset 
    function updateparameters(alpha::Float64, du::AbstractArray{Float64,1})
        for i = 1:levset.dim*levset.numbasis 
            levset.xcenter[i] += alpha*du[i]
        end 
        levset.rho += alpha*du[end]
    end

    # begin the Gauss-Newton iterations 
    max_iter = 1000
    max_line_iter = 10
    grad_norm0 = 0.0
    res = zeros(numpts)
    jac = zeros(numpts, levset.dim*levset.numbasis + 1)
    for n = 1:max_iter 
        # get the residual and Jacobian, and check for convergence
        residual!(res, x, levset)
        println("norm(res) = ",norm(res))
        jacobian!(jac, x, levset)
        #println("jac = ",jac)
        grad = jac'*res
        grad_norm = norm(grad)
        if n == 1 grad_norm0 = grad_norm end
        println("Gauss-Newton iteration ",n) 
        println("\tresidual L2 norm = ",norm(res),": Lmax norm = ",
                norm(res, Inf))
        println("\tgradient norm = ",grad_norm, ": 1st-order opt rel = ",
                grad_norm/grad_norm0)
        if (grad_norm < tol_rel*grad_norm0) || (grad_norm < tol_abs)
            return nothing 
        end

        # get the Gauss-Newton step 
        mu = 1e-4
        Hess = jac'*jac + mu*I 
        du = -Hess\grad

        # sufficient decrease line search
        phi = 0.5*dot(res, res)
        dphi = dot(grad, du)
        alpha = 1.0
        c1 = 1e-4
        for l = 1:max_line_iter
            updateparameters(alpha, du)
            residual!(res, x, levset)
            phi_new = 0.5*dot(res, res)
            println("\tline-search iter = ",l,": phi_new/phi = ", phi_new/phi)
            if phi_new < phi + c1*alpha*dphi 
                break
            elseif l == max_line_iter
                error("Line search failed after ",max_line_iter," iterations")
            end
            updateparameters(-alpha, du)
            alpha *= 0.1
        end
    end
    # Error if we get here
    error("fitlevelset failed after ",max_iter," Gauss-Newton iterations")
end

end # module LevelSets
