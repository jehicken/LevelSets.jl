using LevelSets
using Test
using LinearAlgebra
using DualNumbers
using StaticArrays

@testset "LevelSets.jl" begin
    
    @testset "Constructor" begin 
        dim = 2
        numbasis = 5
        xc = randn(dim, numbasis)
        nrm = randn(dim, numbasis)
        tang = randn(dim, dim-1, numbasis)
        crv = randn(dim-1, numbasis)
        rho = rand()    
        levset = LevelSet{2,Float64}(xc, nrm, tang, crv, rho)
        # check for basic consistency
        @test levset.numbasis == numbasis
        @test isapprox(rho, levset.rho)
        for i = 1:numbasis
            @test isapprox(levset.xcenter[:,i], xc[:,i])
            @test isapprox(norm(levset.frame[:,1,i]), 1.0)
            @test isapprox(levset.frame[:,1,i], nrm[:,i]/norm(nrm[:,i]))
            @test isapprox(norm(levset.frame[:,2,i]), 1.0)
            # the random tangent needs to be orthonormalized before testing
            fac = dot(tang[:,1,i], nrm[:,i])/dot(nrm[:,i], nrm[:,i])
            tmp = tang[:,1,i] - fac*nrm[:,i]
            tmp /= norm(tmp)
            @test isapprox(levset.frame[:,2,i], tmp)
            @test isapprox(levset.kappa[:,i], crv[:,i])
        end
        # check that assertions are raised when appropriate
        nrm_bad = randn(dim+1,numbasis)
        @test_throws AssertionError LevelSet{2,Float64}(xc, nrm_bad, tang, crv, 
                                                        rho)
    end
    
    @testset "Test evallevelset" begin 
        dim = 2
        numbasis = 5
        xc = randn(dim, numbasis)
        nrm = randn(dim, numbasis)
        tang = randn(dim, dim-1, numbasis)
        crv = randn(dim-1, numbasis)
        #crv = zeros(dim-1, numbasis)
        rho = rand()
        levset = LevelSet{2,Float64}(xc, nrm, tang, crv, rho)
        x = randn(dim)
        ls = evallevelset(x, levset)
        # the following reproduces the expected calculation
        numer = 0.0
        denom = 0.0
        for i = 1:numbasis
            xi = view(xc, :, i)
            dist = norm(x - xi)
            fac = 1/norm(nrm[:,i])
            perp = dot(nrm[:,i]*fac, x - xi)
            perp += 0.5*crv[1,i]*(x - xi)'*(I - fac*fac*nrm[:,i]*nrm[:,i]')*
                (x - xi)
            expfac = exp(-rho*dist)
            numer += perp*expfac 
            denom += expfac
        end
        @test isapprox(ls, numer/denom)
    end
    
    @testset "Test difflocallevelset!" begin
        #using LevelSets : locallevelset, difflocallevelset!
        locallevelset = LevelSets.locallevelset
        difflocallevelset! = LevelSets.difflocallevelset!
        dim = 3
        x = randn(dim)
        xc = randn(dim)
        frm = randn(dim, dim)
        crv = randn(dim-1)
        perp_bar = randn()
        # get the analytical derivatives
        xc_bar = zero(xc)
        frm_bar = zero(frm)
        crv_bar = zero(crv)
        difflocallevelset!(xc_bar, frm_bar, crv_bar, x, xc, frm, crv, perp_bar)
        # get the derivatives using dual numbers
        x_dual = dual.(x, 0.0)
        xc_dual = dual.(xc, 0.0)
        frm_dual = dual.(frm, 0.0)
        crv_dual = dual.(crv, 0.0)
        for d = 1:dim
            # check derivative w.r.t. xc
            xc_dual[d] += dual.(0.0, 1.0)
            ls_dual = perp_bar*locallevelset(x_dual, xc_dual, frm_dual,
                                             crv_dual)
            @test isapprox(xc_bar[d], epsilon.(ls_dual))
            xc_dual[d] -= dual.(0.0, 1.0)
            # check derivative w.r.t. frame 
            for k = 1:dim
                frm_dual[d,k] += dual.(0.0, 1.0)
                ls_dual = perp_bar*locallevelset(x_dual, xc_dual, frm_dual, 
                                                 crv_dual)
                @test isapprox(frm_bar[d,k], epsilon.(ls_dual))
                frm_dual[d,k] -= dual.(0.0, 1.0)
            end
        end 
        for d = 1:dim-1
            # check derivative w.r.t. kappa
            crv_dual[d] += dual.(0.0, 1.0)
            ls_dual = perp_bar*locallevelset(x_dual, xc_dual, frm_dual, 
                                             crv_dual)
            @test isapprox(crv_bar[d], epsilon.(ls_dual))
            crv_dual[d] -= dual.(0.0, 1.0)
        end
        x_bar = zero(x)
        difflocallevelset!(x_bar, x, xc, frm, crv, perp_bar)
        for d = 1:dim
            # check derivative w.r.t. x 
            x_dual[d] += dual.(0.0, 1.0)
            ls_dual = perp_bar*locallevelset(x_dual, xc_dual, frm_dual, 
                                             crv_dual)
            @test isapprox(x_bar[d], epsilon.(ls_dual))
            x_dual[d] -= dual.(0.0, 1.0)
        end
    end

    @testset "Test hesslocallevelset! (w.r.t. x)" begin 
        difflocallevelset! = LevelSets.difflocallevelset!
        hesslocallevelset! = LevelSets.hesslocallevelset!
        dim = 3
        x = randn(dim)
        xc = randn(dim)
        frm = randn(dim, dim)
        crv = randn(dim-1)
        # get the analytical Hessian 
        hess = zeros(dim, dim)
        hesslocallevelset!(hess, x, xc, frm, crv)
        # get the Hessian using dual numbers
        x_dual = dual.(x, 0.0)
        xc_dual = dual.(xc, 0.0)
        frm_dual = dual.(frm, 0.0)
        crv_dual = dual.(crv, 0.0)
        x_bar_dual = zero(x_dual)
        for d = 1:dim 
            # derivative w.r.t. x[d]
            x_dual[d] += dual.(0.0, 1.0)
            fill!(x_bar_dual, zero(Dual128))
            difflocallevelset!(x_bar_dual, x_dual, xc_dual, frm_dual, crv_dual,
                               dual.(1.0, 0.0))
            for d2 = 1:dim
                @test isapprox(hess[d,d2], epsilon.(x_bar_dual[d2]))
            end 
            x_dual[d] -= dual.(0.0, 1.0)
        end
    end

    @testset "Test diffexpdist! (w.r.t. x)" begin 
        expdist = LevelSets.expdist
        diffexpdist! = LevelSets.diffexpdist!
        dim = 3
        delta = 1e-8
        x = randn(dim)
        xc = randn(dim)
        rho = rand()
        min_dist = rand()
        exp_bar = randn()
        # get the analytical derivative
        x_bar = zero(x)
        diffexpdist!(x_bar, x, xc, rho, delta, min_dist, exp_bar)
        # get the derivatives using dual numbers
        x_dual = dual.(x, 0.0)
        xc_dual = dual.(xc, 0.0)
        rho_dual = dual.(rho, 0.0)
        delta_dual = dual.(delta, 0.0)
        min_dist_dual = dual.(min_dist, 0.0)
        for d = 1:dim 
            # get derivative with respect to x[d]
            x_dual[d] += dual.(0.0, exp_bar)
            expfac_dual = expdist(x_dual, xc_dual, rho_dual, delta_dual,
                                  min_dist_dual)
            @test isapprox(x_bar[d], epsilon.(expfac_dual))
            x_dual[d] -= dual.(0.0, exp_bar) 
        end
    end

    @testset "Test hessexpdist! (w.r.t. x)" begin 
        diffexpdist! = LevelSets.diffexpdist!
        hessexpdist! = LevelSets.hessexpdist!
        dim = 3
        delta = 1e-8
        x = randn(dim)
        xc = randn(dim)
        rho = rand()
        min_dist = rand()
        # get the analytical Hessian
        hess = zeros(dim, dim)
        hessexpdist!(hess, x, xc, rho, delta, min_dist)
        # get the Hessian using dual numbers 
        x_dual = dual.(x, 0.0)
        xc_dual = dual.(xc, 0.0)
        rho_dual = dual.(rho, 0.0)
        delta_dual = dual.(delta, 0.0)
        min_dist_dual = dual.(min_dist, 0.0)
        x_bar_dual = zero(x_dual)
        for d = 1:dim 
            # get derivative with respect to x[d]
            x_dual[d] += dual.(0.0, 1.0)
            fill!(x_bar_dual, zero(Dual128))
            diffexpdist!(x_bar_dual, x_dual, xc_dual, rho_dual, delta_dual, 
                         min_dist_dual, dual.(1.0, 0.0))
            for d2 = 1:dim 
                @test isapprox(hess[d,d2], epsilon.(x_bar_dual[d2]))
            end
            x_dual[d] -= dual.(0.0, 1.0) 
        end
    end

    @testset "Test difflevelset! (w.r.t. LevelSet parameters)" begin
        dim = 3
        numbasis = 10
        xc = randn(dim, numbasis)
        nrm = randn(dim, numbasis)
        tang = randn(dim, dim-1, numbasis)
        crv = randn(dim-1, numbasis)
        rho = rand()
        levset = LevelSet{3,Float64}(xc, nrm, tang, crv, rho)
        x = randn(dim)
        # get the derivatives using reverse mode 
        xc_bar = zero(xc)
        frm_bar = zeros(dim, dim, numbasis)
        crv_bar = zero(crv)
        rho_bar = zeros(1)
        LevelSets.difflevelset!(xc_bar, frm_bar, crv_bar, rho_bar, x, levset)
        # get the derivatives using Dual numbers and compare with *_bar values
        xc_dual = dual.(xc, 0.0)
        nrm_dual = dual.(nrm, 0.0)
        tang_dual = dual.(tang, 0.0)
        crv_dual = dual.(crv, 0.0)
        rho_dual = dual.(rho, 0.0)
        x_dual = dual.(x, 0.0)
        levset_dual = LevelSet{3,Dual128}(xc_dual, nrm_dual, tang_dual, 
                                          crv_dual, rho_dual)
        for i = 1:numbasis
            for d = 1:dim
                # derivative w.r.t. xcenter[d,i]
                levset_dual.xcenter[d,i] += dual.(0.0, 1.0)
                ls_dual = evallevelset(x_dual, levset_dual)
                @test isapprox(xc_bar[d,i], epsilon.(ls_dual))
                levset_dual.xcenter[d,i] -= dual.(0.0, 1.0)
                for k = 1:dim
                    # derivative w.r.t. frame[k,i][d]
                    levset_dual.frame[d,k,i] += dual.(0.0, 1.0)
                    ls_dual = evallevelset(x_dual, levset_dual)
                    @test isapprox(frm_bar[d,k,i], epsilon.(ls_dual))
                    levset_dual.frame[d,k,i] -= dual.(0.0, 1.0)
                end
            end
            for k = 1:dim-1
                # derivative w.r.t. kappa[k,i]
                levset_dual.kappa[k,i] += dual.(0.0, 1.0)
                ls_dual = evallevelset(x_dual, levset_dual)
                @test isapprox(crv_bar[k,i], epsilon.(ls_dual))
                levset_dual.kappa[k,i] -= dual.(0.0, 1.0)
            end
        end
        # derivative w.r.t. rho
        rho_dual += dual.(0.0, 1.0)
        levset_dual = LevelSet{3,Dual128}(xc_dual, nrm_dual, tang_dual,
                               crv_dual, rho_dual)
        ls_dual = evallevelset(x_dual, levset_dual)
        @test isapprox(rho_bar[1], epsilon.(ls_dual))
    end

    @testset "Test difflevelset! (w.r.t. x)" begin
        dim = 3
        numbasis = 10
        xc = randn(dim, numbasis)
        nrm = randn(dim, numbasis)
        tang = randn(dim, dim-1, numbasis)
        crv = randn(dim-1, numbasis)
        rho = rand()
        levset = LevelSet{3,Float64}(xc, nrm, tang, crv, rho)
        x = randn(dim)
        # get the derivatives using reverse mode 
        x_bar = zero(x)
        LevelSets.difflevelset!(x_bar, x, levset)
        # get the derivatives using Dual numbers and compare with *_bar values
        xc_dual = dual.(xc, 0.0)
        nrm_dual = dual.(nrm, 0.0)
        tang_dual = dual.(tang, 0.0)
        crv_dual = dual.(crv, 0.0)
        rho_dual = dual.(rho, 0.0)        
        levset_dual = LevelSet{3,Dual128}(xc_dual, nrm_dual, tang_dual, 
                                          crv_dual, rho_dual)
        x_dual = dual.(x, 0.0)
        for d = 1:dim 
            x_dual[d] += dual.(0.0, 1.0)
            ls_dual = evallevelset(x_dual, levset_dual)
            @test isapprox(x_bar[d], epsilon.(ls_dual))
            x_dual[d] -= dual.(0.0, 1.0)
        end
    end    
    
    @testset "Test hessianlevelset! (w.r.t. x)" begin 
        dim = 3
        numbasis = 10
        xc = randn(dim, numbasis)
        nrm = randn(dim, numbasis)
        tang = randn(dim, dim-1, numbasis)
        crv = randn(dim-1, numbasis)
        rho = rand()
        levset = LevelSet{3,Float64}(xc, nrm, tang, crv, rho)
        x = randn(dim)
        # get the analytical hessian
        hess = zeros(dim, dim)
        LevelSets.hessianlevelset!(hess, x, levset)
        # get the Hessian using Dual numbers
        xc_dual = dual.(xc, 0.0)
        nrm_dual = dual.(nrm, 0.0)
        tang_dual = dual.(tang, 0.0)
        crv_dual = dual.(crv, 0.0)
        rho_dual = dual.(rho, 0.0)        
        levset_dual = LevelSet{3,Dual128}(xc_dual, nrm_dual, tang_dual, 
                                          crv_dual, rho_dual)
        x_dual = dual.(x, 0.0)
        x_bar_dual = zero(x_dual)
        for d = 1:dim 
            x_dual[d] += dual.(0.0, 1.0)
            fill!(x_bar_dual, zero(Dual128))
            difflevelset!(x_bar_dual, x_dual, levset_dual)
            for d2 = 1:dim
                @test isapprox(hess[d,d2], epsilon.(x_bar_dual[d2]))
            end
            x_dual[d] -= dual.(0.0, 1.0)
        end
    end

    @testset "Test snappoint!" begin 
        # Construct a level-set approximation of the circle, and then snap a 
        # point
        snappoint! = LevelSets.snappoint!
        dim = 2
        numbasis = 12
        xc = zeros(dim, numbasis)
        nrm = zeros(dim, numbasis)
        tang = zeros(dim, dim-1, numbasis)
        crv = zeros(dim-1, numbasis)
        rho = 10.0*numbasis
        radius = 1.0
        for i = 1:numbasis 
            theta = 2*pi*(i-1)/numbasis 
            xc[:,i] = [radius*cos(theta); radius*sin(theta)]
            tang[:,1,i] = [-radius*sin(theta); radius*cos(theta)]
            nrm[:,i] = [tang[2,1,i]; -tang[1,1,i]]
            crv[1,i] = radius
        end 
        levset = LevelSet{2,Float64}(xc, nrm, tang, crv, rho)

        x0 = [2.0; 2.0]
        x = zero(x0)
        snappoint!(x, x0, levset, max_newton=20)
        #println("x = ", x)
        @test isapprox(x, [0.707504186768092; 0.7075041867680917])

        # 3d test 
        a, b, c = 2.0, 1.0, 0.5
        dim = 3
        numu = 10
        numv = 10
        numbasis = numu*numv 
        xc = zeros(dim, numbasis)
        nrm = zeros(dim, numbasis)
        tang = zeros(dim, dim-1, numbasis)
        crv = zeros(dim-1, numbasis)
        for i = 1:numu 
            u = 2*pi*(i-0.5)/numu
            for j = 1:numv 
                v = pi*(j-0.5)/numv 
                n = numv*(i-1) + j
                xc[:, n] = [a*cos(u)*sin(v) + a; b*sin(u)*sin(v) + b; 
                            c*cos(v) + c]
                du = [-a*sin(u)*sin(v); b*cos(u)*sin(v); 0.0]
                dv = [a*cos(u)*cos(v); b*sin(u)*cos(v); -c*sin(v)]
                nrm[:, n] = -cross(du, dv)
                tang[:,1,n] = du 
                tang[:,2,n] = dv
            end
        end
        rho = 10.0*numbasis
        levset = LevelSet{3,Float64}(xc, nrm, tang, crv, rho)
        x0 = [3.0; 3.0; 3.0]
        x = zero(x0)
        snappoint!(x, x0, levset, max_newton=20)
        #println("x = ", x)
        @test isapprox(x, [2.640260656582881, 2.0097225431578574,
                           0.6150832161362152])
    end 


    if false
    @testset "Test residual!" begin
        # This is a simple test: for a single "basis" in the LevelSet, any 
        # point `x` such that `x - xc` is perpendicular to `nrm` will be on the 
        # zero level curve. 
        dim = 3
        numbasis = 1
        xc = randn(dim, numbasis)
        nrm = randn(dim, numbasis)
        rho = rand()
        levset = LevelSet{Float64}(xc, nrm, rho)
        numpts = 10
        # get some points on the plane perpendicular to `nrm`
        x = randn(dim, numpts)
        for j = 1:numpts 
            dx = x[:,j] - xc
            x[:,j] -= dot(dx, levset.normal[:,1])*levset.normal[:,1]
        end
        res = zeros(numpts)
        LevelSets.residual!(res, x, levset) 
        @test isapprox(res, zeros(numpts), atol=1e-13)
    end
    
    # @testset "Test parameterindices" begin 
    #     dim = 2
    #     numbasis = 3
    #     xc = randn(dim, numbasis)
    #     nrm = randn(dim, numbasis)
    #     len = randn(numbasis)
    #     levset = LevelSet{Float64}(xc, nrm, len)
    #     xc_idx, nrm_idx, len_idx = LevelSets.parameterindices(levset)        
    #     for i = 1:numbasis 
    #         for d = 1:dim
    #             @test xc_idx[d,i] == (i-1)*(2*dim+1) + d
    #             @test nrm_idx[d,i] == (i-1)*(2*dim+1) + dim + d 
    #         end
    #         @test len_idx[i] == (i-1)*(2*dim+1) + 2*dim + 1
    #     end
    # end
    
    @testset "Test jacobian!" begin 
        dim = 3
        numbasis = 10
        xc = randn(dim, numbasis)
        nrm = randn(dim, numbasis)
        rho = rand()
        levset = LevelSet{Float64}(xc, nrm, rho)
        numpts = 20
        x = randn(dim, numpts)
        # compute the Jacobian using reverse mode 
        jac = zeros(numpts, dim*numbasis + 1)
        LevelSets.jacobian!(jac, x, levset)
        # compuate the entries in the Jacobian using Dual numbers 
        xc_dual = dual.(xc, 0.0)
        nrm_dual = dual.(nrm, 0.0)
        rho_dual = dual.(rho, 0.0)
        x_dual = dual.(x, 0.0)
        levset_dual = LevelSet{Dual128}(xc_dual, nrm_dual, rho_dual)
        res_dual = dual.(zeros(numpts), 0.0)
        for i = 1:numbasis
            for d = 1:dim
                # derivative w.r.t. xcenter[d,i]
                levset_dual.xcenter[d,i] += dual.(0.0, 1.0)
                LevelSets.residual!(res_dual, x_dual, levset_dual)
                @test isapprox(jac[:,(i-1)*dim + d], epsilon.(res_dual))
                levset_dual.xcenter[d,i] -= dual.(0.0, 1.0)
            end
        end
        # derivative w.r.t. rho 
        rho_dual += dual.(0.0, 1.0)
        levset_dual = LevelSet{Dual128}(xc_dual, nrm_dual, rho_dual)
        LevelSets.residual!(res_dual, x_dual, levset_dual)
        @test isapprox(jac[:,dim*numbasis+1], epsilon.(res_dual))
    end
   
    end
end
