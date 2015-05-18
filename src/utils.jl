# takes data as input and prepares everything for optimization
# X data matrix
# node for which we want to f
function prepareNeighborhoodData{T<:FloatingPoint}(
    X::Matrix{T},
    nodeInd::Int64,
    nodeBasis::NodeBasis,
    edgeBasis::EdgeBasis
    )
  K = nodeBasis.numBasis
  L = edgeBasis.numBasis

  D = getNeighborhoodD(X, nodeInd, nodeBasis, edgeBasis)
  E = getNeighborhoodE(X, nodeInd, nodeBasis, edgeBasis)

  n, p = size(X)

  groups = Array(UnitRange{Int64}, p)
  groups[1] = 1:K
  for t=1:p-1
    groups[t+1] = (K + (t-1)*L+1):(K + t*L)
  end

  A = D'D / n
  b = vec( sum(E, 1) / n )

  #
  zeros(K+(p-1)*L), A, b, groups
end

function estimate_neighborhood{T<:FloatingPoint}(
    X::Matrix{T},
    nodeInd::Int64,
    nodeBasis::NodeBasis,
    edgeBasis::EdgeBasis,
    λarr::Vector{T};
    options::ProximalOPT.ProximalOptions=ProximalOPT.ProximalOptions()
    )
  n, p = size(X)
  numLambda = length(λarr)
  θ, A, b, groups = prepareNeighborhoodData(X, nodeInd, nodeBasis, edgeBasis)
  θarr = Array(Float64, (numLambda, length(θ)))
  numGroups = length(groups)
  f = ProximalOPT.QuadraticFunction(A, b)
  for indLambda=1:numLambda
    λ = vcat(0., λarr[indLambda] * ones(numGroups-1))
    g = ProximalOPT.ProxL1L2(λ, groups)
    ProximalOPT.solve!(ProximalOPT.AccProxGradDescent(), θ, f, g; options=options)
    θarr[indLambda, :] = θ
  end
 θarr
end

function getNeighborhood{T<:FloatingPoint}(
    out::Vector{Bool},
    θ::StridedVector{T},
    nodeInd::Int64,
    p::Int64,
    K::Int64,
    L::Int64;
    zero_thr::T = 1e-4
    )
  out[nodeInd] = true
  i = 0
  for b=1:p
    if b == nodeInd
      continue
    end
    i += 1
    θv = sub(θ, K+(i-1)*L+1:K+i*L)
    out[b] = vecnorm(θv) > zero_thr ? true : false
  end
  out
end


#########################################################
#
#   diagnostics
#
#########################################################


function _trapz{T<:FloatingPoint}(Y::Vector{T}, xmin::T, xmax::T)
  ((sum(Y) - (Y[1] + Y[end]) / 2.) * (xmax - xmin) / (length(Y) - 1))
end

function condDensity{T<:FloatingPoint}(
    fx::Vector{T},
    x::Vector{T},
    θ::Vector{T},
    nodeBasis::NodeBasis,
    edgeBasis::EdgeBasis,
    indNode::Int64,
    condX::Vector{T}
    )

  K = nodeBasis.numBasis
  L = edgeBasis.numBasis
  node_val = nodeBasis.val
  edge_val = edgeBasis.val
  p = length(condX)
  N = length(fx)
  @assert length(x) == N
  @inbounds for i=1:N
    ex = x[i]
    tmp = zero(T)
    for k=1:K
      tmp += node_val(ex, k) * θ[k]
    end
    indEdge = 0
    for b=1:p
      if b == indNode
        continue
      end
      indEdge += 1
      for l=1:L
        tmp += edge_val(ex, condX[b], l) * θ[K + (indEdge-1)*L + l]
      end
    end
    fx[i] = exp(tmp)
  end
  scale!(pX, one(T) / _trapz(pX, ed.xmin, ed.xmax))
  x, fx
end


#########################################################
#
#   evaluation
#
#########################################################


function precision(tGraph::Matrix{Bool}, eGraph::Matrix{Bool}; symmetrize=0)
  # symmetrize - 0 AND
  # symmetrize - 1 OR
  p = size(tGraph, 1)
  numEstim = 0
  numEstimAndTrue = 0
  for a=1:p
    for b=a+1:p
      eEdge = symmetrize == 0 ? eGraph[a,b] && eGraph[b,a] : eGraph[a,b] || eGraph[b,a]
      numEstim += eEdge ? 1 : 0
      numEstimAndTrue += eEdge && tGraph[a,b] ? 1 : 0
    end
  end
  numEstim > 0 ? convert(Float64, numEstimAndTrue / numEstim) : 1.
end

function recall(tGraph::Matrix{Bool}, eGraph::Matrix{Bool}; symmetrize=0)
  # symmetrize - 0 AND
  # symmetrize - 1 OR
  p = size(tGraph, 1)
  numTrue = 0
  numEstimAndTrue = 0
  for a=1:p
    for b=a+1:p
      eEdge = symmetrize == 0 ? eGraph[a,b] && eGraph[b,a] : eGraph[a,b] || eGraph[b,a]
      numTrue += tGraph[a,b] ? 1 : 0
      numEstimAndTrue += eEdge && tGraph[a,b] ? 1 : 0
    end
  end
  numTrue > 0 ? convert(Float64, numEstimAndTrue / numTrue) : 1.
end

