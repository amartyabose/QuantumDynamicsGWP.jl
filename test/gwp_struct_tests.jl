@testitem "Single GWPR" begin
    using LinearAlgebra
    using Random
    Random.seed!(1234)
    for _ = 1:100
        d = rand(1:100)
        diagelems = rand(d)
        ψ = QuantumDynamicsGWP.GWPR(; q=(2rand(d).-1)*10.0, p=(2rand(d).-1), A=diagm(diagelems) * 1im, γ_excess=0.0+0.0im)
        @test norm(ψ) ≈ 1.0
        @test QuantumDynamicsGWP.xelem(ψ, ψ) ≈ ψ.q
        @test QuantumDynamicsGWP.pelem(ψ, ψ) ≈ ψ.p
        @test diag(QuantumDynamicsGWP.x2elem(ψ, ψ)) ≈ 1.0 ./ (2 .* diagelems) .+ ψ.q.^2
        @test diag(QuantumDynamicsGWP.p2elem(ψ, ψ)) ≈ diagelems ./ 2 .+ ψ.p.^2
    end
end

@testitem "Sum of Multiple Gaussians" begin
    using LinearAlgebra
    using QuantumDynamicsGWP
    for _ = 1:100
        d = rand(1:100)
        diagelems = rand(d)
        ψ1 = QuantumDynamicsGWP.GWPR(; q=(2rand(d).-1)*10.0, p=(2rand(d).-1), A=diagm(diagelems) * 1im, γ_excess=0.0+0.0im)
        diagelems = rand(d)
        ψ2 = QuantumDynamicsGWP.GWPR(; q=(2rand(d).-1)*10.0, p=(2rand(d).-1), A=diagm(diagelems) * 1im, γ_excess=0.0+0.0im)
        diagelems = rand(d)
        ψ3 = QuantumDynamicsGWP.GWPR(; q=(2rand(d).-1)*10.0, p=(2rand(d).-1), A=diagm(diagelems) * 1im, γ_excess=0.0+0.0im)
        init_list = QuantumDynamicsGWP.GWPSum([ψ1, ψ2, ψ3])
        normalize!(init_list)
        @test norm(init_list) ≈ 1.0
    end
end

@testitem "MC Sampling of GWPSum" begin
    using LinearAlgebra
    using Random
    Random.seed!(1234)
    nMC = 10000
    for _ = 1:1
        d = 1 # rand(1:100)
        @show d
        diagelems = rand(d)
        ψ1 = QuantumDynamicsGWP.GWPR(; q=(2rand(d).-1)*10.0, p=(2rand(d).-1), A=diagm(diagelems) * 1im, γ_excess=0.0+0.0im)
        diagelems = rand(d)
        ψ2 = QuantumDynamicsGWP.GWPR(; q=(2rand(d).-1)*10.0, p=(2rand(d).-1), A=diagm(diagelems) * 1im, γ_excess=0.0+0.0im)
        diagelems = rand(d)
        ψ3 = QuantumDynamicsGWP.GWPR(; q=(2rand(d).-1)*10.0, p=(2rand(d).-1), A=diagm(diagelems) * 1im, γ_excess=0.0+0.0im)
        init_list = QuantumDynamicsGWP.GWPSum([ψ1, ψ2, ψ3])
        normalize!(init_list)
        mc_list = QuantumDynamicsGWP.MCsample(init_list, ones(d), ones(d), Matrix(I, d, d) * 1.0im, nMC)
        @test typeof(mc_list) == QuantumDynamicsGWP.GWPSum{QuantumDynamicsGWP.GWPR}
        normalize!(mc_list)
        over = QuantumDynamicsGWP.overlap(mc_list, init_list)
        @show over
        @test real(over) ≥ 0.95
        @test imag(over) ≤ 1e-10
    end
end

@testitem "Clustering of MC Samples" setup=[gwpsetup] begin
    using LinearAlgebra
    using Random
    Random.seed!(1234)
    normalize!(gwpsetup.init_list)
    nMC = 10000
    mc_list = QuantumDynamicsGWP.MCsample(gwpsetup.init_list, 4.0, 4.0, ones(1, 1) * 1.0im, nMC)
    cluster_list = QuantumDynamicsGWP.cluster_reduction_sophisticated(mc_list, 75)
    normalize!(cluster_list)
    over = QuantumDynamicsGWP.overlap(cluster_list, gwpsetup.init_list)
    @test real(over) ≥ 0.95
    @test imag(over) ≤ 1e-10
end