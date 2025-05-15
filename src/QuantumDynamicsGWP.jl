module QuantumDynamicsGWP

using LinearAlgebra
using Clustering

abstract type GWP end

mutable struct GWPR <: GWP
    q::Vector{<:Real}
    p::Vector{<:Real}
    A::Matrix{<:Complex}
    γ::Complex
end
γ_default(A::AbstractMatrix) = -0.25im * log(det(imag(A))/π^size(A, 1))
function GWPR(; q::AbstractVector, p::AbstractVector, A::AbstractMatrix, γ_excess::Complex=0.0+0.0im)
    @assert length(q) == length(p) "Position and momentum should have same dimensions"
    @assert length(q) == size(A, 2) && size(A, 1) == size(A, 2) "A matrix should have consistent dimensions"
    @assert issymmetric(A) "A should be symmetric"
    GWPR(q, p, A, γ_default(A) + γ_excess)
end
function GWPR_PQS(; q::AbstractVector, p::AbstractVector, P::AbstractMatrix, Q::AbstractMatrix, S=0.0+0.0im)
    @assert length(q) == length(p) "Position and momentum should have same dimensions"
    d = size(P, 1)
    γ = S + 1im * (0.25d * log(π) + 0.5 * log(det(Q)))
    GWPR(q, p, P*inv(Q), γ)
end
(gwp::GWPR)(x::AbstractVector) = exp(1im * (transpose(x-gwp.q) * gwp.A/2 * (x-gwp.q) + transpose(gwp.p) * (x-gwp.q) + gwp.γ))
γ_default(gwp::GWPR) = γ_default(gwp.A)
extra_coeff(gwp::GWPR) = exp(1im * (gwp.γ - γ_default(gwp)))
function overlap(gwpbra::GWPR, gwpket::GWPR)
    q1, p1, A1, γ1 = gwpbra.q, gwpbra.p, gwpbra.A, gwpbra.γ
    q2, p2, A2, γ2 = gwpket.q, gwpket.p, gwpket.A, gwpket.γ

    @assert size(A1)==size(A2) "The two GWPs are not on same dimensional spaces."

    A = A2 - conj(A1)
    b = p2 - p1 - A2 * q2 + conj(A1) * q1
    over = exp(1im * (γ2 - conj(γ1)) - 1im * (transpose(p2) * q2 - transpose(p1) * q1) + 0.5im * (transpose(q2) * A2 * q2 - transpose(q1) * conj(A1) * q1)) * sqrt((2im * π)^size(A, 1) / det(A)) * exp(-0.5im * transpose(b) * (A \ b))
    isnan(over) || isinf(over) ? 0.0 + 0.0im : over
end
function LinearAlgebra.norm(gwp::GWP)
    over = overlap(gwp, gwp)
    @assert imag(over) ≤ 1e-10 "The imaginary part of the overlap is greater than 1e-10"
    sqrt(real(over))
end
function LinearAlgebra.normalize!(gwp::GWP)
    gwp.γ += 1im * log(norm(gwp))
end

mutable struct GWPSum{T<:GWP}
    gwps::Vector{T}
end
(gwpsum::GWPSum)(x) = sum([g(x) for g in gwpsum])
Base.eltype(gwpsum::GWPSum) = eltype(gwpsum.gwps)
Base.length(gwpsum::GWPSum) = length(gwpsum.gwps)
Base.iterate(gwpsum::GWPSum, state=1) = state ≤ length(gwpsum) ? (gwpsum.gwps[state], state+1) : nothing
Base.getindex(gwpsum::GWPSum, i::Int64) = gwpsum.gwps[i]
function overlap(gwpsumbra::GWPSum, gwpsumket::GWPSum)
    ans = 0.0 + 0.0im
    for gwpbra in gwpsumbra, gwpket in gwpsumket
        ans += overlap(gwpbra, gwpket)
    end
    ans
end
function overlap(gwpbra::GWP, gwplistket::GWPSum)
    ans = 0.0 + 0.0im
    for gwpket in gwplistket
        ans += overlap(gwpbra, gwpket)
    end
    ans
end
overlap(gwplistbra::GWPSum, gwpket::GWP) = conj(overlap(gwpket, gwplistbra))
function LinearAlgebra.norm(gwpsum::GWPSum)
    over = overlap(gwpsum, gwpsum)
    @assert imag(over) ≤ 1e-10 "The imaginary part of the overlap is greater than 1e-10"
    sqrt(real(over))
end
function LinearAlgebra.normalize!(gwpsum::GWPSum)
    exc_gamma = 1im * log(norm(gwpsum))
    for gwp in gwpsum
        gwp.γ += exc_gamma
    end
end
function LinearAlgebra.normalize(gwpsum::GWPSum)
    gwpsumcopy = deepcopy(gwpsum)
    normalize!(gwpsumcopy)
    gwpsumcopy
end

function Prob(qtrial, ptrial, Atrial, init_list::GWPSum)
    abs(overlap(GWPR(; q=qtrial, p=ptrial, A=Atrial), init_list))
end

function MCsample(init_gwplist::GWPSum, dq, dp, A, nMC::Int64)
    mc_gwplist = Vector{eltype(init_gwplist)}(undef, nMC)
    mc_multiplicities = zeros(Int64, nMC)
    q, p = init_gwplist[1].q, init_gwplist[1].p
    spacedim = length(q)
    twopid = (2π)^spacedim
    naccept = 1
    gwp = GWPR(; q, p, A)
    Pcurr = Prob(q, p, A, init_gwplist)
    Pprev = Pcurr
    coeff = overlap(gwp, init_gwplist) / Pcurr
    mc_gwplist[naccept] = GWPR(; q, p, A, γ_excess=-1im * log(coeff))
    mc_multiplicities[naccept] = 1
    for _ = 1:nMC-1
        qtmp = q + (2rand(spacedim) .- 1) * dq
        ptmp = p + (2rand(spacedim) .- 1) * dp
        Pcurr = Prob(qtmp, ptmp, A, init_gwplist)
        if Pcurr / Pprev ≥ rand()
            q = qtmp
            p = ptmp
            Pprev = Pcurr
            naccept += 1
            gwp = GWPR(; q=qtmp, p=ptmp, A)
            coeff = overlap(gwp, init_gwplist) / Pcurr
            gwp.γ -= 1im * log(coeff)
            mc_gwplist[naccept] = gwp
            mc_multiplicities[naccept] = 1
        else
            mc_multiplicities[naccept] += 1
        end
    end
    for j = 1:naccept
        mc_gwplist[j].γ -= 1im * log(mc_multiplicities[j] / (nMC * twopid))
    end
    GWPSum(mc_gwplist[1:naccept])
end

function cluster_reduction(mc_sum::GWPSum, nclusters::Int64)
    qppoints = reduce(hcat, [vcat(gwp.q, gwp.p) for gwp in mc_sum])
    weights = [abs(extra_coeff(gwp)) for gwp in mc_sum]
    kmeans_result = kmeans(qppoints, nclusters; weights)
    a = assignments(kmeans_result)
    centers = kmeans_result.centers

    coeff = zeros(ComplexF64, nclusters)
    for (ass, extracoeff) in zip(a, extra_coeff.(mc_sum))
        coeff[ass] += extracoeff
    end
    A = mc_sum[1].A
    gwpsum = Vector{GWP}(undef, nclusters)
    ndims = size(qppoints, 1)÷2
    for j = 1:nclusters
        gwpsum[j] = GWPR(; q=centers[1:ndims, j], p=centers[ndims+1:end, j], A, γ_excess=-1im * log(coeff[j]))
    end

    GWPSum(gwpsum)
end
end
