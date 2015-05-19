reload("NPGM")
import PyPlot

import NPGM
import NPGM.BSplineFunctions

# splines version
n = 200
p = 50
rho = 0.8
Sigma = zeros(Float64, p, p)
for a=1:p
  for b=1:p
    Sigma[a,b] = rho^abs(a-b)
  end
end
Theta = inv(Sigma)
Theta[find(abs(Theta) .< 1e-4)] = 0.
sqSigma = sqrtm(Sigma)

tGraph = Array(Bool, p, p)
for r=1:p
  for c=1:p
    tGraph[r,c] = Theta[r,c] == zero(Float64) ? false : true
  end
end

nodeBasis = NPGM.BSplineFunctions.BSplineNodeBasis(4, -4., 4., 4)
K = nodeBasis.numBasis
edgeBasis = NPGM.BSplineFunctions.BSplineEdgeBasis(3, (-4., -4.), (4., 4.), (2, 2))
L = edgeBasis.numBasis

numLambda = 50
λarr = 10.^linspace(log10(0.01), log10(0.001), numLambda)

X = randn(n, p) * sqSigma
nodeInd = 2

#################
  min_eigval=1e-10
#   D = NPGM.getNeighborhoodD(X, nodeInd, nodeBasis, edgeBasis)
#   E = NPGM.getNeighborhoodE(X, nodeInd, nodeBasis, edgeBasis)

# ef = eigfact(Symmetric(D'D/n), min_eigval, Inf)
# sv = sort(ef.values, rev=true)

θ, A11, A12, A22, b1, b2, groups = NPGM.prepareLowRankNeighborhoodData(X, nodeInd, nodeBasis, edgeBasis; min_eigval=min_eigval)

A11.values

#   n, p = size(X)
#   groups = Array(UnitRange{Int64}, p-1)
#   for t=1:p-1
#     groups[t] = (t-1)*L+1:t*L
#   end

#   # compute A11, A12, A22
#   i1 = 1:K
#   i2 = K+1:size(D, 2)
#   svdD = svdfact(D)
#   S = svdD.S
#   m = length(S)
#   k = 0
#   @inbounds while k < m
#     k += 1
#     v = S[k].^2. / n
#     if v < min_eigval
#       k -= 1
#       break
#     end
#     S[k] = v
#   end
#   resize!(S, k)
#   U1 = svdD[:Vt][1:k, i1]'
#   U2 = svdD[:Vt][1:k, i2]'
#   tmp = U1*Diagonal(S)
#   ef = eigfact(Symmetric(Diagonal(S)-tmp'*((tmp*(Diagonal(S)\tmp'))\tmp)), min_eigval, Inf)
#   A22 = NPGM.LowRankEigen(U2 * ef.vectors, ef.values)
#   A12 = U1 * (Diagonal(S) * U2')
#   ef = eigfact(Symmetric(U1*(Diagonal(S)*U1')), min_eigval, Inf)
#   A11 = NPGM.LowRankEigen(ef.vectors, ef.values)

#   E1 = sub(E, :, i1)
#   E2 = sub(E, :, i2)
#   b1 = vec( sum(E1, 1) / n )
#   b2 = vec( sum(E2, 1) / n ) - At_mul_B(A12, A11 \ b1)
###############
options = ProximalOPT.ProximalOptions(;ftol=1e-4, xtol=1e-4)
@time eNeigh = NPGM.estimate_neighborhood(X, nodeInd, nodeBasis, edgeBasis, λarr)

eNeigh

bNeighborhood = Array(Bool, p)
θv = slice(eNeigh, 18, :)
NPGM.getNeighborhood(bNeighborhood, θv, nodeInd, p, K, L)

condX = zeros(p)
condX[2] = -2.
x = [linspace(-4., 4., 100);]
fx = similar(x)
NPGM.condDensity(fx, x, vec(eNeigh[18, :]), nodeBasis, edgeBasis, 1, condX, -4., 4.)
PyPlot.plot(x, fx)

