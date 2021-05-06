using LevelSets
using Test
using LinearAlgebra
using DualNumbers

@testset "LevelSets.jl" begin
    
    @testset "Constructor" begin 
        dim = 2
        numbasis = 5
        xc = randn(dim, numbasis)
        nrm = randn(dim, numbasis)
        rho = rand()
        levset = LevelSet{Float64}(xc, nrm, rho)
        # check for basic consistency
        @test levset.dim == dim 
        @test levset.numbasis == numbasis 
        @test isapprox(xc, levset.xcenter)
        @test isapprox(rho, levset.rho)
        for i = 1:numbasis 
            @test isapprox(norm(levset.normal[:,i]), 1.0)
            @test isapprox(levset.normal[:,i], nrm[:,i]/norm(nrm[:,i]))
        end
        # check that assertions are raised when appropriate
        nrm_bad = randn(dim+1,numbasis)
        @test_throws AssertionError LevelSet{Float64}(xc, nrm_bad, rho)
    end
    
    @testset "Test evallevelset" begin 
        dim = 2
        numbasis = 5
        xc = randn(dim, numbasis)
        nrm = randn(dim, numbasis)
        rho = rand()
        levset = LevelSet{Float64}(xc, nrm, rho)
        x = randn(dim)
        ls = evallevelset(x, levset)
        # the following reproduces the expected calculation
        numer = 0.0
        denom = 0.0
        for i = 1:numbasis
            xi = view(xc, :, i)
            dist = norm(x - xi)
            perp = dot(nrm[:,i]/norm(nrm[:,i]), x - xi)
            expfac = exp(-rho*dist)
            numer += perp*expfac 
            denom += expfac
        end
        @test isapprox(ls, numer/denom)
    end
    
    @testset "Test difflevelset!" begin
        dim = 3
        numbasis = 10
        xc = randn(dim, numbasis)
        nrm = randn(dim, numbasis)
        rho = rand()
        levset = LevelSet{Float64}(xc, nrm, rho)
        x = randn(dim)
        # get the derivatives using reverse mode 
        xc_bar = zeros(dim, numbasis)
        nrm_bar = zeros(dim, numbasis)
        rho_bar = zeros(1)
        LevelSets.difflevelset!(xc_bar, nrm_bar, rho_bar, x, levset)
        # get the derivatives using Dual numbers and compare with *_bar values
        xc_dual = dual.(xc, 0.0)
        nrm_dual = dual.(nrm, 0.0)
        rho_dual = dual.(rho, 0.0)
        x_dual = dual.(x, 0.0)
        levset_dual = LevelSet{Dual128}(xc_dual, nrm_dual, rho_dual)
        for i = 1:numbasis
            for d = 1:dim
                # derivative w.r.t. xcenter[d,i]
                levset_dual.xcenter[d,i] += dual.(0.0, 1.0)
                ls_dual = evallevelset(x_dual, levset_dual)
                @test isapprox(xc_bar[d,i], epsilon.(ls_dual))
                levset_dual.xcenter[d,i] -= dual.(0.0, 1.0)
                # derivative w.r.t. normal[d,i]
                levset_dual.normal[d,i] += dual.(0.0, 1.0)
                ls_dual = evallevelset(x_dual, levset_dual)
                @test isapprox(nrm_bar[d,i], epsilon.(ls_dual))
                levset_dual.normal[d,i] -= dual.(0.0, 1.0)
            end
        end
        # derivative w.r.t. rho
        rho_dual += dual.(0.0, 1.0)
        levset_dual = LevelSet{Dual128}(xc_dual, nrm_dual, rho_dual)
        ls_dual = evallevelset(x_dual, levset_dual)
        @test isapprox(rho_bar[1], epsilon.(ls_dual))
    end
    
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
