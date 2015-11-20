# takes data as input and prepares everything for optimization
# X data matrix
# node for which we want to f
function prepareLowRankNeighborhoodData{T<:FloatingPoint}(
    X::Matrix{T},
    nodeInd::Int64,
    createNodeBasis::Function,
    createEdgeBasis::Function;
    min_eigval::T=1e-10
    )
  tmp_nodeBasis = createNodeBasis(1); tmp_edgeBasis = createEdgeBasis(1,2)
  K = tmp_nodeBasis.numBasis
  L = tmp_edgeBasis.numBasis

  D = getNeighborhoodD(X, nodeInd, createNodeBasis, createEdgeBasis)
  E = getNeighborhoodE(X, nodeInd, createNodeBasis, createEdgeBasis)

  n, p = size(X)
  groups = Array(UnitRange{Int64}, p-1)
  for t=1:p-1
    groups[t] = (t-1)*L+1:t*L
  end

  # A = D'D / n
  # compute A11, A12, A22
  # we find a low rank approximation to A
  i1 = 1:K
  i2 = K+1:size(D, 2)
  A = D'D / n
  ef = eigfact(Symmetric(A[i1, i1]), min_eigval, Inf)
  A11 = LowRankEigen(ef.vectors, ef.values)
  A12 = A[i1, i2]
  ef = eigfact(Symmetric(A[i2, i2] - A12' * (A11 \ A12)), min_eigval, Inf)
  A22 = LowRankEigen(ef.vectors, ef.values)

  E1 = sub(E, :, i1)
  E2 = sub(E, :, i2)
  b1 = vec( sum(E1, 1) / n )
  b2 = vec( sum(E2, 1) / n ) - At_mul_B(A12, A11 \ b1)

  #
  zeros(K+(p-1)*L), A11, A12, A22, b1, b2, groups
end

function prepareLowRankNeighborhoodData1{T<:FloatingPoint}(
    X::Matrix{T},
    nodeInd::Int64,
    createNodeBasis::Function,
    createEdgeBasis::Function;
    min_eigval::T=1e-10
    )
  
  tmp_nodeBasis = createNodeBasis(1); tmp_edgeBasis = createEdgebasis(1,2)
  K = tmp_nodeBasis.numBasis
  L = tmp_edgeBasis.numBasis

  D = getNeighborhoodD(X, nodeInd, createNodeBasis, createEdgeBasis)
  E = getNeighborhoodE(X, nodeInd, createNodeBasis, createEdgeBasis)

  n, p = size(X)
  groups = Array(UnitRange{Int64}, p-1)
  for t=1:p-1
    groups[t] = (t-1)*L+1:t*L
  end

  # A = D'D / n
  # compute A11, A12, A22
  # we find a low rank approximation to A
  i1 = 1:K
  i2 = K+1:size(D, 2)
  D1 = sub(D, :, i1)
  D2 = sub(D, :, i2)
  A = D'D / n
  ef = eigfact(Symmetric(A[i1, i1]), min_eigval, Inf)
  A11 = LowRankEigen(ef.vectors, ef.values)
  A12 = A[i1, i2]
#   ef = eigfact(Symmetric(A[i2, i2] - A12' * (A11 \ A12)), min_eigval, Inf)
  svdf = svdfact(D1)
  A22 = XtX((eye(n) - svdf[:U]*svdf[:U]')*D2)
#   A22 = LowRankEigen(ef.vectors, ef.values)

  E1 = sub(E, :, i1)
  E2 = sub(E, :, i2)
  b1 = vec( sum(E1, 1) / n )
  b2 = vec( sum(E2, 1) / n ) - At_mul_B(A12, A11 \ b1)

  #
  zeros(K+(p-1)*L), A11, A12, A22, b1, b2, groups
end


function estimate_neighborhood{T<:FloatingPoint}(
    X::Matrix{T},
    nodeInd::Int64,
    createNodeBasis::Function,
    createEdgeBasis::Function,
    λarr;#::Vector{T};
    options::ProximalOPT.ProximalOptions=ProximalOPT.ProximalOptions(),
    min_eigval::T=1e-10
    )
  n, p = size(X)
  tmp_nodeBasis = createNodeBasis(1); tmp_edgeBasis = createEdgeBasis(1,2)
  K = tmp_nodeBasis.numBasis
  L = tmp_edgeBasis.numBasis

  numLambda = length(λarr)
  #
  θ, A11, A12, A22, b1, b2, groups = prepareLowRankNeighborhoodData(X, nodeInd, createNodeBasis, createEdgeBasis; min_eigval=min_eigval)
  θ1 = sub(θ, 1:K)
  θ2 = sub(θ, K+1:length(θ))
  #
  θarr = Array(Float64, (numLambda, length(θ)))
  numGroups = length(groups)
  f = ProximalOPT.QuadraticFunction(A22, b2)
  for indLambda=1:numLambda
    g = ProximalOPT.ProxL1L2(λarr[indLambda], groups)
    ProximalOPT.solve!(ProximalOPT.AccProxGradDescent(), θ2, f, g; options=options)
    # compute     θ1 = -(A11 \ (A12*θ2 + b1))
    copy!(θ1, b1)
    gemv!('N', -1., A12, θ2, -1., θ1)
    A_ldiv_B!(A11, θ1)
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
    condX::Vector{T},
    xmin::T,
    xmax::T
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
  scale!(fx, one(T) / _trapz(fx, xmin, xmax))
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

function RandomData(n, p)
	prob = 0.3; rand_prob = rand(p, p); rand_prob= rand_prob/2 + rand_prob'/2
	for i = 1:p
		rand_prob[i,i] = 1
	end
	tGraph = Array(Bool, p, p)
	for i = 1:p
		for j=1:p
			if rand_prob[i,j] <= prob && i!=j
				tGraph[i,j] = true
			else
				tGraph[i,j] = false
			end
		end
	end
	deg = sum(tGraph, 1)
	for i =1:length(deg)
		if deg[i] == 0
			tGraph[i,i+1] = tGraph[i+1,i] = true
		end
	end

	Theta = zeros(Float64, p, p)
	off_val = 0.3
	for i = 1:p
		for j=(i+1):p
			if tGraph[i,j]
				Theta[i,j] = Theta[j,i] = off_val
			end
		end
	end

	eigen_val = eig(Theta)[1]; min_eigen = 1000
	for i =1:p
		if min_eigen > eigen_val[i]
			min_eigen = eigen_val[i]
		end
	end

	for i = 1:p
		Theta[i,i] += abs(min_eigen)+ 0.2
	end

	Sigma = inv(Theta)
	sqSigma = real( sqrtm(Sigma) )
	X = randn(n, p) * sqSigma
	X, tGraph, Theta, Sigma
end
