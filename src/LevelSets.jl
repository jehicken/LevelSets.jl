"Implements smooth-max inspired level-set functions."
module LevelSets

using LinearAlgebra

export LevelSet, evallevelset, fitlevelset!

"Data required by smooth-max level-set function."
struct LevelSet{T<:Number}
    "Number of spatial dimensions the level-set is embedded in."
    dim::Int 
    "Number of basis functions that define the level set."
    numbasis::Int
    "Centers for the basis functions."
    xcenter::Array{T,2}
    "Outward pointing normal at the centers."
    normal::Array{T,2}
    "Length scale for individual basis functions."
    lenscale::Array{T,1}

    function LevelSet{T}(xcenter::AbstractArray{T,2},
                         normal::AbstractArray{T,2},
                         lenscale::AbstractArray{T,1}) where {T<:Number}
        @assert(size(xcenter) == size(normal),
                "size of xcenter and normal do not match")
        dim = size(xcenter,1)
        numbasis = size(xcenter,2) 
        @assert(numbasis == size(lenscale,1),
                "size of lenscale does not match numbasis")
        xc = deepcopy(xcenter) 
        len = deepcopy(lenscale)
        # normalize the vectors in normal
        nrm = deepcopy(normal)
        @inbounds for i = 1:numbasis
            nrm[:,i] /= norm(normal[:,i])
        end
        new(dim, numbasis, xc, nrm, len)
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
    #@inbounds for i = 1:levset.numbasis
    #    xc = view(levset.xcenter, :, i)
    #    dist = norm(x - xc)
    #    perp = dot(levset.normal[:,i], x - xc)
    #    expfac = exp(-levset.lenscale[i]*dist)
    #    numer += perp*expfac
    #    denom += expfac
    #end

    @inbounds for i = 1:levset.numbasis 
        xc = view(levset.xcenter, :, i)
        nrm = view(levset.normal, :, i)
        dx = x - xc
        perp = dot(nrm, dx)
        dist = sqrt(perp*perp + dx'*(I - nrm*nrm')*dx)
        expfac = exp(-levset.lenscale[i]*dist)
        numer += perp*expfac
        denom += expfac
    end 

    ls = numer/denom
    return ls
end

"""
    difflevelset!(xcenter_bar, normal_bar, lenscale_bar, x, levset)

Compute the derivatives of the level set, defined by `levset`, at point `x`
with respect to the `LevelSet` parameters.

Uses the reverse-mode of algorithmic differentiation (i.e. back propagation).
"""
function difflevelset!(xcenter_bar::AbstractArray{T,2},
                       normal_bar::AbstractArray{T,2},
                       lenscale_bar::AbstractArray{T,1},
                       x::AbstractArray{T,1},
                       levset::LevelSet{T}) where {T<:Number}
    @assert(size(xcenter_bar) == size(levset.xcenter),
            "xcenter_bar and levset.xcenter have inconsistent dimensions")
    @assert(size(normal_bar) == size(levset.normal),
            "normal_bar and levset.normal have inconsistent dimensions")
    @assert(size(lenscale_bar) == size(levset.lenscale),
            "lenscale_bar and levset.lenscale have inconsistent dimensions")
    @assert(size(x,1) == size(levset.xcenter,1), "x inconsistent with levset")
    # forward sweep 
    numer = zero(T)
    denom = zero(T)
    @inbounds for i = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        dist = norm(x - xc)
        perp = dot(levset.normal[:,i], x - xc)
        expfac = exp(-levset.lenscale[i]*dist)
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
        dist = norm(x - xc)
        perp = dot(levset.normal[:,i], x - xc)
        expfac = exp(-levset.lenscale[i]*dist)
        # denom += expfac
        expfac_bar = denom_bar
        # numer += perp*expfac 
        perp_bar = numer_bar*expfac 
        expfac_bar += numer_bar*perp
        # expfac = exp(-levset.lenscale[i]*dist)
        lenscale_bar[i] -= expfac_bar*expfac*dist 
        dist_bar = -expfac_bar*expfac*levset.lenscale[i]
        # perp = dot(levset.normal[:,i], x - xc)
        normal_bar[:,i] += perp_bar.*(x - xc)
        xc_bar = -perp_bar.*levset.normal[:,i]
        # dist = norm(x - xc)
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
    @assert(size(jac,2) == (2*levset.dim + 1)*levset.numbasis, 
        "jac columns are inconsistent with levset")
    @assert(size(x,1) == size(levset.xcenter,1), "x inconsisent with levset")
    fill!(jac, zero(T))
    # work arrays for reverse-mode AD 
    xcenter_bar = zeros(T, size(levset.xcenter))
    normal_bar = zeros(T, size(levset.normal))
    lenscale_bar = zeros(T, size(levset.lenscale))
    # indices for parameters 
    xcenter_indices, normal_indices, lenscale_indices = parameterindices(levset)
    @inbounds for j = 1:size(x,2)
        fill!(xcenter_bar, zero(T))
        fill!(normal_bar, zero(T))
        fill!(lenscale_bar, zero(T))
        difflevelset!(xcenter_bar, normal_bar, lenscale_bar, view(x,:,j),      
                      levset)
        for (k, index) in enumerate(xcenter_indices)
            jac[j,index] += xcenter_bar[k] # note: xcenter_bar is a 2D array
        end 
        for (k, index) in enumerate(normal_indices) 
            jac[j,index] += normal_bar[k] # note: normal_bar is a 2D array
        end 
        for (k, index) in enumerate(lenscale_indices)
            jac[j,index] += lenscale_bar[k] # note: lenscale_bar is a 2D array
        end
    end
    return nothing
end 

function fitlevelset!(levset::LevelSet{Float64}, x::AbstractArray{Float64,2};
                      xcenter_dof::Bool=true, normal_dof::Bool=false,
                      lenscale_dof::Bool=true, tol_rel::Float64=1e-10, tol_abs::Float64=1e-12)
    @assert(size(x,1) == size(levset.xcenter,1), "x inconsisent with levset")
    numpts = size(x, 2)
    # indices for parameters 
    xcenter_indices, normal_indices, lenscale_indices = parameterindices(levset)

    # adds du to appropriate parameters in levset 
    function updateparameters(alpha::Float64, du::AbstractArray{Float64,1})
        if xcenter_dof
            for (k, index) in enumerate(xcenter_indices)
                levset.xcenter[k] += alpha*du[index]
            end
        end 
        if normal_dof
            for (k, index) in enumerate(normal_indices)
                levset.normal[k] += alpha*du[index]
            end 
        end 
        if lenscale_dof 
            for (k, index) in enumerate(lenscale_indices)
                levset.lenscale[k] += alpha*du[index]
            end 
        end
    end

    # indices defines the parameter active degrees of freedom 
    indices = zeros(UInt32, 0)
    if xcenter_dof append!(indices, xcenter_indices) end 
    if normal_dof append!(indices, normal_indices) end 
    if lenscale_dof append!(indices, lenscale_dof) end
    numvar = size(indices,1)

    # begin the Gauss-Newton iterations 
    max_iter = 1000
    max_line_iter = 10
    grad_norm0 = 0.0
    res = zeros(numpts)
    jac_full = zeros(numpts, (2*levset.dim + 1)*levset.numbasis)
    jac = zeros(numpts, numvar)
    for n = 1:max_iter 
        # get the residual and Jacobian, and check for convergence
        residual!(res, x, levset)
        jacobian!(jac_full, x, levset)
        jac = jac_full[:,indices]
        grad = jac'*res
        grad_norm = norm(grad)
        if n == 1 grad_norm0 = grad_norm end
        println("Gauss-Newton iteration ",n) 
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
