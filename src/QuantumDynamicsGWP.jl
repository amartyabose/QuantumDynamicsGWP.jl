module QuantumDynamicsGWP

using LinearAlgebra
# using Clustering
using ParallelKMeans
using FLoops

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
    @assert norm(A - transpose(A)) ≤ 1e-12 "A should be symmetric. $(A)"
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
    over = exp(1im * (γ2 - conj(γ1)) - 1im * (transpose(p2) * q2 - transpose(p1) * q1)) * sqrt((2im * π)^size(A, 1) / det(A)) * exp(-0.5im * (transpose(b) * (A \ b)-(transpose(q2) * A2 * q2 - transpose(q1) * conj(A1) * q1)))
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

function xelem(gwpbra::GWPR, gwpket::GWPR)
    q1, p1, A1 = gwpbra.q, gwpbra.p, gwpbra.A
    q2, p2, A2 = gwpket.q, gwpket.p, gwpket.A

    A = A2 .- conj.(A1)
    b = p2 .- p1 .- A2 * q2 .+ conj.(A1) * q1
    pre = -A \ b

    pre .* overlap(gwpbra, gwpket)  # d×1 complex vector
end

function pelem(gwpbra::GWPR, gwpket::GWPR)
    q1, p1, A1 = gwpbra.q, gwpbra.p, gwpbra.A
    q2, p2, A2 = gwpket.q, gwpket.p, gwpket.A

    A = A2 .- conj.(A1)
    b = p2 .- p1 .- A2 * q2 .+ conj.(A1) * q1
    m = -A \ b
    pre = p2 .+ A2 * (m .- q2)

    pre .* overlap(gwpbra, gwpket)  # d×1 complex vector
end

function x2elem(gwpbra::GWPR, gwpket::GWPR)
    q1, p1, A1 = gwpbra.q, gwpbra.p, gwpbra.A
    q2, p2, A2 = gwpket.q, gwpket.p, gwpket.A

    A = A2 - conj.(A1)
    b = p2 - p1 - A2 * q2 + conj.(A1) * q1
    m = -A \ b

    outer = m * transpose(m)
    cov = 1im * inv(A)  # i (A2 - A1*)^{-1}

    S = overlap(gwpbra, gwpket)
    S * (outer + cov)  # d×d complex matrix
end

function p2elem(gwpbra::GWPR, gwpket::GWPR)
    q1, p1, A1 = gwpbra.q, gwpbra.p, gwpbra.A
    q2, p2, A2 = gwpket.q, gwpket.p, gwpket.A

    A = A2 - conj.(A1)
    b = p2 - p1 - A2 * q2 + conj.(A1) * q1
    m = -A \ b  # μ

    v = p2 + A2 * (m - q2)  # ν

    outer = v * transpose(v)  # ν ν^T (transpose without conj)
    width1 = 1im * A2 * inv(A) * A2 
    width2 = -1im * A2

    S = overlap(gwpbra, gwpket)
    S * (outer + width1 + width2)  # d×d complex matrix
end

mutable struct GWPSum{T<:GWP}
    gwps::Vector{T}
end
function (gwpsum::GWPSum)(x)
    ngwps = length(gwpsum)
    @floop for j in 1:ngwps
        @reduce ans = 0.0im + gwpsum[j](x)
    end
    ans
end
Base.eltype(gwpsum::GWPSum) = eltype(gwpsum.gwps)
Base.length(gwpsum::GWPSum) = length(gwpsum.gwps)
Base.firstindex(gwpsum::GWPSum) = 1
Base.lastindex(gwpsum::GWPSum) = length(gwpsum)
Base.iterate(gwpsum::GWPSum, state=1) = state ≤ length(gwpsum) ? (gwpsum.gwps[state], state+1) : nothing
Base.getindex(gwpsum::GWPSum, i::Int64) = gwpsum.gwps[i]
function overlap(gwpsumbra::GWPSum, gwpsumket::GWPSum)
    nbras = length(gwpsumbra)
    nkets = length(gwpsumket)
    @floop for j = 1:nbras, k = 1:nkets
        @reduce ans = 0.0im + overlap(gwpsumbra[j], gwpsumket[k])
    end
    ans
end
function overlap(gwpbra::GWP, gwplistket::GWPSum)
    nkets = length(gwplistket)
    @floop for j = 1:nkets
        @reduce ans = 0.0im + overlap(gwpbra, gwplistket[j])
    end
    ans
end
overlap(gwplistbra::GWPSum, gwpket::GWP) = conj(overlap(gwpket, gwplistbra))
function expval(gwpsum::GWPSum, fn)
    num_gwps = length(gwpsum)
    expval = [zero(real(f(gwpsum[1], gwpsum[1]))) for f in fn]
    @floop for j = 1:num_gwps
        localval = [real(f(gwpsum[j], gwpsum[j])) for f in fn]
        for k = j+1:num_gwps
            @inbounds localval += [2.0 * real(f(gwpsum[j], gwpsum[k])) for f in fn]
        end
        @reduce expval .+= localval
    end
    expval
end
xexpect(gwpsum::GWPSum) = expval(gwpsum, [xelem])[1]
x2expect(gwpsum::GWPSum) = expval(gwpsum, [x2elem])[1]
pexpect(gwpsum::GWPSum) = expval(gwpsum, [pelem])[1]
p2expect(gwpsum::GWPSum) = expval(gwpsum, [p2elem])[1]
LinearAlgebra.norm(gwpsum::GWPSum) = sqrt(expval(gwpsum, [overlap])[1])
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
        j = rand(1:spacedim)
        qtmp = deepcopy(q)
        ptmp = deepcopy(p)
        qtmp[j] += (2rand() - 1) * dq[j]
        ptmp[j] += (2rand() - 1) * dp[j]
        Pcurr = Prob(qtmp, ptmp, A, init_gwplist)
        if Pcurr / Pprev ≥ rand()
            q = deepcopy(qtmp)
            p = deepcopy(ptmp)
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

function groupby_clusters(mc_sum::GWPSum, assignments, nclusters)
    clusterlist = [GWP[] for _ = 1:nclusters]
    for (a, w) in zip(assignments, mc_sum)
        push!(clusterlist[a], w)
    end
    [GWPSum(c) for c in clusterlist]
end

function cluster_reduction_dumb(mc_sum::GWPSum, nclusters::Int64)
    qppoints = reduce(hcat, [vcat(gwp.q, gwp.p) for gwp in mc_sum])
    weights = [abs(extra_coeff(gwp)) for gwp in mc_sum]
    kmeans_result = kmeans(Coreset(), qppoints, nclusters; weights)
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

function cluster_reduction_sophisticated(mc_sum::GWPSum, nclusters::Int64)
    qsize = length(mc_sum[1].q)
    psize = length(mc_sum[1].p)
    qppoints = reduce(hcat, [vcat(gwp.q, gwp.p) for gwp in mc_sum])
    weights = [abs(extra_coeff(gwp)) for gwp in mc_sum]
    kmeans_result = kmeans(Yinyang(), qppoints, nclusters; weights)
    # a = assignments(kmeans_result)
    a = kmeans_result.assignments
    clustersum = groupby_clusters(mc_sum, a, nclusters)
    qmean = zeros(qsize, nclusters)
    pmean = zeros(psize, nclusters)
    qvar = zeros(qsize, nclusters)
    pvar = zeros(psize, nclusters)
    clusternorm = zeros(nclusters)
    A = [zeros(ComplexF64, qsize, qsize) for _=1:nclusters]
    γ_excess = zeros(ComplexF64, nclusters)
    gwplist = Vector{GWPR}(undef, nclusters)
    @inbounds begin
        for (j, cluster) in enumerate(clustersum)
            xbar, x2bar, pbar, p2bar, psipsi = expval(cluster, [xelem, x2elem, pelem, p2elem, overlap])
            clusternorm[j] = sqrt(psipsi)
            qmean[:, j] .= xbar / psipsi
            pmean[:, j] .= pbar / psipsi
            qvar[:, j] .= diag(x2bar) / psipsi .- qmean[:, j].^2
            pvar[:, j] .= diag(p2bar) / psipsi .- pmean[:, j].^2
            imA = 1.0 ./ (2 * qvar[:, j])
            reA = sqrt.(abs.((2 * pvar[:, j] .- imA) .* imA))
            A[j] = diagm(reA + 1im * imA)
            γ_excess[j] = -1im * log(clusternorm[j]) + angle(cluster(qmean[:, j]))
            gwplist[j] = GWPR(; q=qmean[:,j], p=pmean[:,j], A=A[j], γ_excess=γ_excess[j])
        end
    end
    GWPSum(gwplist)
end
end
