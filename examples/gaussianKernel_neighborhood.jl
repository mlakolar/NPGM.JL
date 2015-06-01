import NPGM
import NPGM.GaussianFunctions
import NPGM.KernelFunctions

kernel = NPGM.KernelFunctions.GaussianKernel(1.)
nodeBasis = NPGM.GaussianFunctions.GaussianNodeBasis()
edgeBasis = NPGM.KernelFunctions.KernelEdgeBasis(kernel)

n = 1000
p = 10
X = zeros(n, p)

nodeBeta = zeros(p, 2)
for i=1:p
  nodeBeta[i, 1] = 0.25
end
edges = zeros(Int64, (9, 2))
for i=1:9
  edges[i, 1] = i
  edges[i, 2] = i+1
end
edgeBeta = zeros(9, 1)
for i=1:9
  edgeBeta[i] = 5
end

function mh_sampler(x::Float64)
    x = x + randn()
end;

function mh_ratio(x::Float64, y::Float64)
    1.
end;

NPGM.graph_sample!(X, n, edges, [1:p;],
              nodeBasis.val, edgeBasis.val, mh_sampler, mh_ratio,
              nodeBeta, edgeBeta; burnin=1000, gap=200)

options = ProximalOPT.ProximalOptions(;ftol=1e-4, xtol=1e-4)

numLambda = 50
λarr = 10.^linspace(log10(0.8), log10(0.01), numLambda)

nodeInd = 1
eNeigh = NPGM.estimate_neighborhood(X, nodeInd, nodeBasis, edgeBasis, λarr)

eNeigh[30, :]

