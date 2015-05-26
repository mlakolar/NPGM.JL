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

options = ProximalOPT.ProximalOptions(;ftol=1e-4, xtol=1e-4)
@time eNeigh = NPGM.estimate_neighborhood(X, nodeInd, nodeBasis, edgeBasis, λarr)

bNeighborhood = Array(Bool, p)
θv = slice(eNeigh, 18, :)
NPGM.getNeighborhood(bNeighborhood, θv, nodeInd, p, K, L)

condX = zeros(p)
condX[2] = -2.
x = [linspace(-4., 4., 100);]
fx = similar(x)
NPGM.condDensity(fx, x, vec(eNeigh[18, :]), nodeBasis, edgeBasis, 1, condX, -4., 4.)
PyPlot.plot(x, fx)

