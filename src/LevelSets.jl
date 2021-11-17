"Implements smooth-max inspired level-set functions."
module LevelSets

using LinearAlgebra

export LevelSet, evallevelset, difflevelset!, fitlevelset!
export boundlevelset, boundlevelsetgrad!

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
    "Curvature at each point"
    kappa::Array{T,2}
    "Penalty parameter"
    rho::T
    "Parameter that smooths distance near zero"
    delta::T

    function LevelSet{T}(xcenter::AbstractArray{T,2},
                         normal::AbstractArray{T,2},
                         kappa::AbstractArray{T,2},
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
        kap = deepcopy(kappa)
        delta = 1e-10
        new(dim, numbasis, xc, nrm, kap, rho, delta)
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
    min_dist = 1e100
    @inbounds for i = 1:levset.numbasis 
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        min_dist = min(min_dist, dist)
    end
    @inbounds for i = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        perp = dot(levset.normal[:,i], x - xc)
        # second order correction; need radius of curvature for this to work 
        perp += 0.5*levset.kappa[1,i]*(x - xc)'*(I - levset.normal[:,i]*levset.normal[:,i]')*(x - xc)
        expfac = exp(-levset.rho*(dist - min_dist))
        numer += perp*expfac
        denom += expfac
    end
    ls = numer/denom
    return ls

    if false 
    @inbounds for i = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        perp = dot(levset.normal[:,i], x - xc)
        # second order correction; need radius of curvature for this to work 
        perp += 0.5*levset.kappa[1,i]*(x - xc)'*(I - levset.normal[:,i]*levset.normal[:,i]')*(x - xc)
        expfac = exp(-levset.rho*dist)
        numer += perp*expfac
        denom += expfac
    end

    # TEMP!!! Trying out specialized function for TE 
    #xc = [1.0; 0.0]
    #nrm = [0.9922778767136676; -0.12403473458920845]
    #dist = sqrt(dot(x - xc, x- xc) + levset.delta)
    #perp = dot(nrm, x - xc)
    #expfac = exp(-levset.rho*dist)
    #numer += ((1/tan(0.39/2))*sqrt(dist*dist - perp*perp) + perp)*expfac
    #denom += expfac 

    ls = numer/denom
    return ls
    end 
end

"""
    ls, bound = boundlevelset!(x, dx, levset)

Return level-set value and bound defined by `levset`, over the element centered
at `x` and with half side lengths `dx`.
"""
function boundlevelset(x::AbstractArray{T,1}, dx::AbstractArray{T,1},
                       levset::LevelSet{T}) where {T<:Number}
    @assert(size(x,1) == size(levset.xcenter,1), "x inconsistent with levset")
    @assert(size(dx,1) == size(levset.xcenter,1),"dx inconsistent with levset")
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
        bound += max(d_max - ls, -d_min + ls)*min(work[i]*factor, 1)
        #lower = min(lower, d_min) 
        upper = max(upper, d_max)
        if d_min < 0.0 
            lower += d_min*min(work[i]*factor,1)
        else 
            lower += d_min*work[i]/factor
        end 
        # if d_max < 0.0 
        #     upper += d_max*work[i]/factor
        # else 
        #     upper += d_max*min(work[i]*factor,1)
        # end
    end
    return ls, bound, bound # lower, upper 
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
    difflevelset!(x_bar, x, levset)

Compute the derivatives of the level set, defined by `levset`, at point `x`
with respect to `x`.  The derivatives are stored in `x_bar`.

Uses the reverse-mode of algorithmic differentiation (i.e. back propagation).
"""
function difflevelset!(x_bar::AbstractArray{T,1}, x::AbstractArray{T,1},
                       levset::LevelSet{T}) where {T<:Number}
    @assert(size(x_bar) == size(x),
            "x_bar and x have inconsistent dimensions")
    @assert(size(x,1) == size(levset.xcenter,1), "x inconsistent with levset")
    # forward sweep 
    numer = zero(T)
    denom = zero(T)
    min_dist = 1e100
    @inbounds for i = 1:levset.numbasis 
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        min_dist = min(min_dist, dist)
    end
    @inbounds for i = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        perp = dot(levset.normal[:,i], x - xc)
        #expfac = exp(-levset.rho*dist)
        expfac = exp(-levset.rho*(dist - min_dist))
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
    @inbounds for i = 1:levset.numbasis
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        perp = dot(levset.normal[:,i], x - xc)
        expfac = exp(-levset.rho*(dist - min_dist))
        # denom += expfac
        expfac_bar = denom_bar
        # numer += perp*expfac 
        perp_bar = numer_bar*expfac 
        expfac_bar += numer_bar*perp
        # expfac = exp(-levset.rho*(dist - min_dist))
        dist_bar = -expfac_bar*expfac*levset.rho
        # perp = dot(levset.normal[:,i], x - xc)
        x_bar[:] += perp_bar.*levset.normal[:,i] 
        # dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        x_bar[:] += dist_bar.*(x - xc)./dist
    end
    return nothing
end

"""
    bound = boundlevelsetgrad!(x_bar, x, dx, levset)

Return bound on level-set gradient `x_bar`, defined by `levset`, over the 
element centered at `x` and with half side lengths `dx`.
"""
function boundlevelsetgrad!(x_bar::AbstractArray{T,1}, x::AbstractArray{T,1}, 
                            dx::AbstractArray{T,1},
                            levset::LevelSet{T})::Array{T,1} where {T<:Number}
    @assert(size(x_bar) == size(x),
            "x_bar and x have inconsistent dimensions")
    @assert(size(x,1) == size(levset.xcenter,1), "x inconsistent with levset")
    @assert(size(dx,1) == size(levset.xcenter,1),"dx inconsistent with levset")
    min_dist = 1e100
    j_max = 0
    @inbounds for i = 1:levset.numbasis 
        xc = view(levset.xcenter, :, i)
        dist = sqrt(dot(x - xc, x - xc) + levset.delta)
        if dist < min_dist 
            min_dist = dist 
            j_max = i 
        end
        #min_dist = min(min_dist, dist)
    end
    # evaluate the gradient at x 
    difflevelset!(x_bar, x, levset) 

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

    # evaluate the basis functions at the point x
    work = zeros(T, levset.numbasis) # should be in levset to avoid allocations 
    d_min = 1e100*ones(T, levset.numbasis)
    d_max = -1e100*ones(T, levset.numbasis)
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
        # WARNING: the following is only valid for linear (kappa = 0)
        for idx = 1:size(vtx,2)
            perp = dot(levset.normal[:,i], vtx[:,idx] - xc)
            d_min[i] = min(d_min[i], perp)
            d_max[i] = max(d_max[i], perp)
        end
    end 
    ls = numer/denom # level-set value at center of element
    work ./= denom # finish computing psi_i at x 
    #j_max = argmax(work) #.*(1 .- work))

    L = norm(dx)
    factor = exp(2.0*levset.rho*L)
    bound = zeros(T, levset.dim)
    xc_jmax = view(levset.xcenter, :, j_max)
    @inbounds for i = 1:levset.numbasis

        xc = view(levset.xcenter, :, i)
        for d = 1:levset.dim
            # WARNING: quadratic not accounted for 
            bound[d] += (abs(levset.normal[d,i] - x_bar[d])*
                         min(work[i]*factor,1.0))
        end

        if true 

            if i == j_max
                continue 
            end 

            # find the bound on | d_i - d_j_max | 
            temp = 0.0
            for idx = 1:size(vtx,2)
                diff_dist = abs(dot(levset.normal[:,i], vtx[:,idx] - xc) 
                        - dot(levset.normal[:,j_max], vtx[:,idx] - xc_jmax))
                temp = max(temp, diff_dist)
            end
            psi_max = min(work[i]*factor, 1)
            psi_min = work[i]/factor
            #temp *= 2*levset.rho*min(0.25, psi_max*(1 - psi_min))
            #temp *= 2*levset.rho*min(0.25, max(psi_max*(1 - psi_max),
            #                                   psi_min*(1 - psi_min)))
            if psi_max > 0.5 && psi_min < 0.5 
                temp *= 2*levset.rho*0.25 
            else 
                temp *= 2*levset.rho*max(psi_max*(1 - psi_max),
                                         psi_min*(1 - psi_min))
            end 
            bound[:] .+= temp

        elseif false
            
            # works, but is conservative 
            if i == j_max
                continue
            end

            temp = max(d_max[i] - d_min[j_max], -d_min[i] + d_max[j_max])
            psi_max = min(work[i]*factor, 1)
            psi_min = work[i]/factor
            if psi_max > 0.5 && psi_min < 0.5 
                temp *= 2*levset.rho*0.25 
            else 
                temp *= 2*levset.rho*max(psi_max*(1 - psi_max),
                                         psi_min*(1 - psi_min))
            end 

            #temp *= 2*levset.rho*min(0.25, psi_max*(1 - psi_min))
            #println("x = ",x,": bound i = ",i," = ",temp)
            bound[:] .+= temp 

        elseif false
            # the following works, but is conservative
            bound[:] .+= 2*levset.rho*max(d_max[i] - ls, -d_min[i] + ls)*
                        min(0.25, min(work[i]*factor, 1)*(1.0 - work[i]/factor))

            temp = 2*levset.rho*max(d_max[i] - ls, -d_min[i] + ls)
            psi_max = min(work[i]*factor, 1)
            psi_min = work[i]/factor
            if psi_max > 0.5 && psi_min < 0.5 
                bound[:] .+= temp*0.25
            else 
                bound[:] .+= temp*max(psi_max*(1 - psi_max),
                                      psi_min*(1 - psi_min))
            end 

            #bound[:] .+= max(d_max - ls, -d_min + ls)*min(work[i]*factor, 1)
        end
    end
    return bound 
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
