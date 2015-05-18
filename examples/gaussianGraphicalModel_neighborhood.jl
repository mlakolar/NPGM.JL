#reload("NPGM")
import NPGM
import NPGM.GaussianFunctions
import PyPlot

# parametric version
n = 500
p = 100
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

nodeBasis = NPGM.GaussianFunctions.GaussianNodeBasis(0.)
K = nodeBasis.numBasis
edgeBasis = NPGM.GaussianFunctions.GaussianEdgeBasis(0.)
L = edgeBasis.numBasis

numLambda = 30
λarr = 10.^linspace(log10(1), log10(0.1), numLambda)
ePrecision = zeros(numLambda)
eRecall = zeros(numLambda)

numRep = 10
for rep=1:numRep
  X = randn(n, p) * sqSigma

  eGraphs = Array(Matrix{Bool}, numLambda)
  for i=1:numLambda
    eGraphs[i] = zeros(Bool, (p, p))
  end

  bNeighborhood = Array(Bool, p)
  for nodeInd=1:p
    @printf("Rep ==> %d Node ==> %d\n", rep, nodeInd)
    eNeigh = NPGM.estimate_neighborhood(X, nodeInd, nodeBasis, edgeBasis, λarr)
    for i=1:numLambda
      θv = slice(eNeigh, i, :)
      NPGM.getNeighborhood(bNeighborhood, θv, nodeInd, p, K, L)
      eGraphs[i][nodeInd, :] = bNeighborhood
    end
  end

  for i=1:numLambda
    ePrecision[i] += NPGM.precision(tGraph, eGraphs[i])
    eRecall[i] += NPGM.recall(tGraph, eGraphs[i])
  end
  @show ePrecision
  @show eRecall
end

scale!(ePrecision, 1./numRep)
scale!(eRecall, 1./numRep)

my_dpi = 96
xsize=500*4
ysize=375*5
fig = figure("pyplot_customtime",figsize=(2*xsize/my_dpi, ysize/my_dpi), dpi=my_dpi) # Create a figure and save the handle
ax1 = PyPlot.subplot(121)
PyPlot.plot(λarr, ePrecision, linewidth=4)
PyPlot.title("Precision")
ax1[:set_ylim]([0,1])

ax2 = PyPlot.subplot(122)
PyPlot.plot(λarr, eRecall, linewidth=4)
PyPlot.title("Recall")
ax2[:set_ylim]([0,1])
