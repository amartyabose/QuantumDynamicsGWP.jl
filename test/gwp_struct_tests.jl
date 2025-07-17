@testitem "Single GWPR" begin
    using LinearAlgebra
    using Random
    Random.seed!(1234)
    for _ = 1:100
        d = rand(1:100)
        diagelems = rand(d)
        ψ = QuantumDynamicsGWP.GWPR(; q=(2rand(d).-1)*10.0, p=(2rand(d).-1), A=diagm(diagelems) * 1im, γ_excess=0.0+0.0im)
        @show d
        @test norm(ψ) ≈ 1.0
        @test QuantumDynamicsGWP.xelem(ψ, ψ) ≈ ψ.q
        @test QuantumDynamicsGWP.pelem(ψ, ψ) ≈ ψ.p
        @test diag(QuantumDynamicsGWP.x2elem(ψ, ψ)) ≈ 1.0 ./ (2 .* diagelems) .+ ψ.q.^2
        @test diag(QuantumDynamicsGWP.p2elem(ψ, ψ)) ≈ diagelems ./ 2 .+ ψ.p.^2
    end
end

@testmodule gwpsetup begin
    using QuantumDynamicsGWP
    init_list = QuantumDynamicsGWP.GWPSum([QuantumDynamicsGWP.GWPR(; q=[1.0], p=[3.0], A=ones(1, 1) * 0.6im), QuantumDynamicsGWP.GWPR(; q=[-3.0], p=[1.0], A=ones(1,1) * 1.0im), QuantumDynamicsGWP.GWPR(; q=[0.0], p=[0.0], A=ones(1,1) * 1.4im)])
end
@testitem "Sum of Multiple Gaussians" setup=[gwpsetup] begin
    using LinearAlgebra
    @test norm(gwpsetup.init_list) ≈ 1.6944080689288
    normalize!(gwpsetup.init_list)
    @test norm(gwpsetup.init_list) ≈ 1.0
end

@testitem "MC Sampling of GWPSum" setup=[gwpsetup] begin
    using LinearAlgebra
    using Random
    Random.seed!(1234)
    normalize!(gwpsetup.init_list)
    nMC = 10000
    mc_list = QuantumDynamicsGWP.MCsample(gwpsetup.init_list, 4.0, 4.0, ones(1, 1) * 1.0im, nMC)
    @test typeof(mc_list) == QuantumDynamicsGWP.GWPSum{QuantumDynamicsGWP.GWPR}
    normalize!(mc_list)
    over = QuantumDynamicsGWP.overlap(mc_list, gwpsetup.init_list)
    @test real(over) ≥ 0.95
    @test imag(over) ≤ 1e-10
end

@testitem "Clustering of MC Samples" setup=[gwpsetup] begin
    using LinearAlgebra
    using Random
    Random.seed!(1234)
    normalize!(gwpsetup.init_list)
    nMC = 10000
    mc_list = QuantumDynamicsGWP.MCsample(gwpsetup.init_list, 4.0, 4.0, ones(1, 1) * 1.0im, nMC)
    cluster_list = QuantumDynamicsGWP.cluster_reduction(mc_list, 75)
    normalize!(cluster_list)
    over = QuantumDynamicsGWP.overlap(cluster_list, gwpsetup.init_list)
    @test real(over) ≥ 0.95
    @test imag(over) ≤ 1e-10
end