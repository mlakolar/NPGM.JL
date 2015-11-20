function getNeighborhoodD{T<:FloatingPoint}(
	X::AbstractMatrix{T},
    nodeInd::Int64,
    NodeBasis::Function,   #change it
    EdgeBasis::Function,   #change it
	param::Vector{T}) 
    n, p = size(X); K = n+1; L = 2*n + 1
	getNeighborhoodD!(zeros(size(X, 1), K+(size(X,2)-1)*L),
		              X, nodeInd, NodeBasis, EdgeBasis, param)
end

# This function returns D matrix of size n x (K + (p-1) * L)
# where
#     K is the number of basis per node
#     L is the number of basis per edge
# note that the function is implemented only for pairwise graphical models
function getNeighborhoodD!{T<:FloatingPoint}(
    out::StridedMatrix{T},
    X::StridedMatrix{T},
    nodeInd::Int64, 
	NodeBasis::Function,
	EdgeBasis::Function,
	param::Vector{T})

	n, p = size(X)
  	K = n+1 
  	L = 2*n + 1 
  	nodeBasis = NodeBasis(X[:,nodeInd], param)
  	node_der = nodeBasis.der
  	@assert size(out) == (n, K+(p-1)*L)
  	@inbounds for rowInd=1:n
    # process node
    	for k=1:K
			out[rowInd, k] = node_der(X[rowInd, nodeInd], k)
    	end
    	# process edges
    	indEdge = 0
    	for b=1:p
    		if nodeInd == b
        		continue
      		end
      		indEdge += 1
			edgeBasis = EdgeBasis(X[:,[nodeInd, b]], param)
			edge_der = edgeBasis.der
      # derivative with respect to first argument
      		for l=1:L
        		out[rowInd, K + (indEdge-1)*L + l] = edge_der(X[rowInd, nodeInd], X[rowInd, b], l, 1)
      		end
    	end
  	end
  out
end


# returns matrix E  of size n x K + (p-1)*L
# first K components correspond to nodeInd
# next (p-1)*L elements correpond to edges
function getNeighborhoodE!{T<:FloatingPoint}(
    out::StridedMatrix{T},
    X::StridedMatrix{T},
    nodeInd::Int64,
    NodeBasis::Function,
    EdgeBasis::Function,
	param::Vector{T})
  n, p = size(X)
  K = n + 1 
  L = 2*n + 1 

  nodeBasis = NodeBasis( X[:,nodeInd], param)
  node_der2 = nodeBasis.der2
  @assert size(out) == (n, K+(p-1)*L)
  @inbounds for rowInd=1:n
    # process node
    for k=1:K
      out[rowInd, k] = node_der2(X[rowInd, nodeInd], k)
    end
    # process edges
    numEdges = 0
    for b=1:p
      if nodeInd == b
        continue
      end
      numEdges += 1
	  edgeBasis = EdgeBasis( X[:, [nodeInd, b]], param)
	  edge_der2 = edgeBasis.der2
      for l=1:L
        out[rowInd, K + (numEdges-1)*L + l] = edge_der2(X[rowInd, nodeInd], X[rowInd, b], l, 1)
      end
    end
  end
  out
end

function getNeighborhoodE{T<:FloatingPoint}(
  X::StridedMatrix{T},
  nodeInd::Int64,
  NodeBasis::Function,
  EdgeBasis::Function, param::Vector{T}) 
	n,p = size(X); K = n+1; L = 2*n + 1
	getNeighborhoodE!(zeros(size(X, 1), K+(size(X,2)-1)*L),
                        X,
                        nodeInd,
                        NodeBasis,
                        EdgeBasis, param)
end
