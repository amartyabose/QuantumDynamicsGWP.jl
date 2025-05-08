module QuantumDynamicsGWP

using LinearAlgebra

abstract type GWP end

mutable struct GWPR <: GWP
    q
    p
    A
    γ
end
GWPR(; q, p, A, γ_excess=0.0) = GWPR(q, p, A, - 0.25im * log(imag(A)/π) + γ_excess)
(gwp::GWPR)(x) = exp(1im * (gwp.A/2 * (x-gwp.q)^2 + gwp.p * (x-gwp.q) + gwp.γ))
γ_default(gwp::GWPR) = -0.25im * log(imag(gwp.A) / π)
extra_coeff(gwp::GWPR) = exp(1im * (gwp.γ - γ_default(gwp)))
function overlap(gwpbra::GWPR, gwpket::GWPR)
    q1, p1, A1, γ1 = gwpbra.q, gwpbra.p, gwpbra.A, gwpbra.γ
    q2, p2, A2, γ2 = gwpket.q, gwpket.p, gwpket.A, gwpket.γ

    A = A2 - conj(A1)
    b = p2 - p1 - A2 * q2 + conj(A1) * q1
    exp(1im * (γ2 - conj(γ1)) - 1im * (p2 * q2 - p1 * q1) + 0.5im * (A2 * q2^2 - conj(A1) * q1^2)) * sqrt(2im * π / A) * exp(-0.5im * b^2 / A)
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

function Prob(qtrial, ptrial, Atrial, init_list::GWPSum)
    abs(overlap(GWPR(; q=qtrial, p=ptrial, A=Atrial), init_list))
end

function MCsample(init_gwplist::GWPSum, dq, dp, A, nMC::Int64)
    mc_gwplist = Vector{eltype(init_gwplist)}(undef, nMC)
    mc_multiplicities = zeros(Int64, nMC)
    q, p = init_gwplist[1].q, init_gwplist[1].p
    naccept = 1
    gwp = GWPR(; q, p, A)
    Pcurr = Prob(q, p, A, init_gwplist)
    Pprev = Pcurr
    coeff = overlap(gwp, init_gwplist) / (Pcurr * nMC * 2π)
    mc_gwplist[naccept] = GWPR(; q, p, A, γ_excess=-1im * log(coeff))
    mc_multiplicities[naccept] = 1
    for _ = 1:nMC-1
        qtmp = q + (2rand() - 1) * dq
        ptmp = p + (2rand() - 1) * dp
        Pcurr = Prob(qtmp, ptmp, A, init_gwplist)
        if Pcurr / Pprev ≥ rand()
            q = qtmp
            p = ptmp
            Pprev = Pcurr
            naccept += 1
            gwp = GWPR(; q=qtmp, p=ptmp, A)
            coeff = overlap(gwp, init_gwplist) / (Pprev * nMC * 2π)
            gwp.γ -= 1im * log(coeff)
            mc_gwplist[naccept] = gwp
            mc_multiplicities[naccept] = 1
        else
            mc_multiplicities[naccept] += 1
        end
    end
    for j = 1:naccept
        mc_gwplist[j].γ -= 1im * log(mc_multiplicities[j])
    end
    GWPSum(mc_gwplist[1:naccept])
end

end
