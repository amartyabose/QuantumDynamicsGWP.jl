@testitem "Single GWPR" begin
    using LinearAlgebra
    for _ = 1:100
        ψ = QuantumDynamicsGWP.GWPR(; q=(2rand()-1)*10.0, p=rand(), A=rand() * 1im, γ_excess=0.0)
        @test norm(ψ) ≈ 1.0
    end
end

@testitem "Sum of Multiple Gaussians" begin
    using LinearAlgebra
    init_list = QuantumDynamicsGWP.GWPSum([QuantumDynamicsGWP.GWPR(; q=1.0, p=3.0, A=0.6im), QuantumDynamicsGWP.GWPR(; q=-3.0, p=1.0, A=1.0im), QuantumDynamicsGWP.GWPR(; q=0.0, p=0.0, A=1.4im)])
    @test norm(init_list) ≈ 1.6944080689288
    normalize!(init_list)
    @test norm(init_list) ≈ 1.0
end