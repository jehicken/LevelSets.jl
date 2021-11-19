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
    
    @testset "Test findclosest!" begin 
        # Construct a level-set approximation of the circle, and then find the 
        # closest point
        findclosest! = LevelSets.findclosest!
        dim = 2
        numbasis = 20
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
        findclosest!(x, x0, levset, max_newton=20)
        println("x = ", x)
        @test isapprox(x, [1/sqrt(2); 1/sqrt(2)])


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
