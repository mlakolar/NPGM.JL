
#reload("NPGM")
import NPGM
#import NPGM.GaussianFunctions
import NPGM.HeatFunctions
import PyPlot

cd("Dropbox\\GitHub\\mk\\pkg\\NPGM.jl\\examples")
Theta = readdlm("./heat_data/graph_structure", Int64)
# parametric version
p = 25;
tGraph = Array(Bool, p, p)
for r=1:p
  for c=1:p
    tGraph[r,c] = Theta[r,c] == zero(Int64) ? false : true
  end
end

using Gadfly
spy(Theta)
spy(tGraph)

nodeBasis = NPGM.HeatFunctions.HeatNodeBasis(0.)
K = nodeBasis.numBasis
edgeBasis = NPGM.HeatFunctions.HeatEdgeBasis(0.)
L = edgeBasis.numBasis

numLambda = 30
λarr = 10.^linspace(log10(1), log10(0.1), numLambda)
ePrecision = zeros(numLambda)
eRecall = zeros(numLambda)

X = readdlm("./heat_data/1", Float64)

eGraphs = Array(Matrix{Bool}, numLambda)
for i=1:numLambda
  eGraphs[i] = zeros(Bool, (p, p))
end

bNeighborhood = Array(Bool, p)
for nodeInd=1:p
  eNeigh = NPGM.estimate_neighborhood(X, nodeInd, nodeBasis, edgeBasis, λarr)
  for i=1:numLambda
    θv = slice(eNeigh, i, :)
    NPGM.getNeighborhood(bNeighborhood, θv, nodeInd, p, K, L)
    eGraphs[i][nodeInd, :] = bNeighborhood
  end
end

num_edges = Array(Int64, numLambda)
for( i =1:numLambda)
  num_edges[i] = ceil( sum(eGraphs[i])/2 - p/2)
end

PyPlot.plot(λarr, num_edges)
PyPlot.plot([0, 1], [sum(tGraph)/2, sum(tGraph)/2])

for i=1:numLambda
  ePrecision[i] += NPGM.precision(tGraph, eGraphs[i])
  eRecall[i] += NPGM.recall(tGraph, eGraphs[i])
end
@show ePrecision
@show eRecall

#scale!(ePrecision, 1./numRep)
#scale!(eRecall, 1./numRep)

my_dpi = 96
xsize=500*4
ysize=375*5
fig = PyPlot.figure("pyplot_customtime",figsize=(2*xsize/my_dpi, ysize/my_dpi), dpi=my_dpi) # Create a figure and save the handle
ax1 = PyPlot.subplot(121)
PyPlot.plot(λarr, ePrecision, linewidth=4)
PyPlot.title("Precision")
ax1[:set_ylim]([0,1])

ax2 = PyPlot.subplot(122)
PyPlot.plot(λarr, eRecall, linewidth=4)
PyPlot.title("Recall")
ax2[:set_ylim]([0,1])
