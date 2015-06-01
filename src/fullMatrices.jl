########################################
#
#  Matrix D
#
########################################

# suppose that p(x) ~ exp{ t(x)' beta }
# where  t(x), beta ∈ R^m with X ∈ R^p  then the matrix D
# is of size p x m

# fills matrix of size 2 x (p*K + p * (p-1) / 2 * L)
# with elements of D(x_i)
# where
#     K is the number of basis per node
#     L is the number of basis per edge
# does not allocate memmory
# note that the function is implemented only for pairwise graphical models
function _getCompactDi!{T<:FloatingPoint}(
    out::StridedMatrix{T},
    X::StridedMatrix{T},
    rowInd::Int64,
    nodeBasis::NodeBasis,
    edgeBasis::EdgeBasis
    )
  n, p = size(X)
  K = nodeBasis.numBasis
  L = edgeBasis.numBasis
  node_der = nodeBasis.der
  edge_der = edgeBasis.der
  # process node elements
  @inbounds for a=1:p
    for k=1:K
      _ci = (a-1)*K+k
      out[_ci, 1] = node_der(X[rowInd, a], k)
      out[_ci, 2] = zero(T)
    end
  end
  # process edges
  indEdge = 0
  @inbounds for a=1:p
    for b=a+1:p
      indEdge += 1
      # derivative with respect to first argument
      for l=1:L
        _ci = p*K + (indEdge-1)*L + l
        out[_ci, 1] = edge_der(X[rowInd, a], X[rowInd, b], l, 1)
      end
      # derivative with respect to second argument
      for l=1:L
        _ci = p*K + (indEdge-1)*L + l
        out[_ci, 2] = edge_der(X[rowInd, a], X[rowInd, b], l, 2)
      end
    end
  end
end

# helper function to obtain index of an edge
# (1, 2) -> 1
# (1, p) -> (p-1)
# (2, 3) -> p
# (3, 4) -> (p-1) + (p-2) + 1
# ...
function _getEdgeIndex(a::Int64, b::Int64, p::Int64)
    (a-1) * p - round(Int, (a-1)*a/2) + (b-a)
end
# constructs matrix DD = sum_i D(x_i)'D(x_i)
function getDD(
    X::Array{Float64, 2},
    nodeBasis::NodeBasis,
    edgeBasis::EdgeBasis
    )
  n, p = size(X)
  K = nodeBasis.numBasis
  L = edgeBasis.numBasis
  numColumns = round(Int, p*K + p * (p-1) / 2 * L)
  partial_derivatives = zeros(Float64, numColumns, 2)

  # number of non-zero elements in D'D
  numNZ = p * K * K +
    round(Int, L * L * p * (p-1) / 2) +
    2 * (p-1) * K * L * p +
    L * L * p * (p - 1) * (p - 2)
  V = zeros(Float64, numNZ)
  I = zeros(Int64, numNZ)
  J = zeros(Int64, numNZ)

  for i=1:n
    # obtain partial derivatives to construct DD matrix
    _getCompactDi!(partial_derivatives, X, i, nodeBasis, edgeBasis)

    # diagonal elements
    ci = 1
    for a=1:p
      for k1=1:K
        V[ci] += partial_derivatives[(a-1)*K+k1, 1]^2
        ci += 1
        for k2=k1+1:K
          tmp = partial_derivatives[(a-1)*K+k1, 1] * partial_derivatives[(a-1)*K+k2, 1]
          V[ci  ] += tmp
          V[ci+1] += tmp

          ci += 2
        end
      end
    end
    indEdge = 0
    for a=1:p
      for b=a+1:p
        indEdge += 1
        for l1=1:L
          _ri = p*K + (indEdge-1)*L+l1
          V[ci] += partial_derivatives[_ri, 1]^2 + partial_derivatives[_ri, 2]^2
          ci += 1
          for l2=l1+1:L
            _ri = p*K + (indEdge-1)*L+l1
            _ci = p*K + (indEdge-1)*L+l2
            tmp = partial_derivatives[_ri, 1]*partial_derivatives[_ci, 1] + partial_derivatives[_ri, 2]*partial_derivatives[_ci, 2]
            V[ci    ] += tmp
            V[ci + 1] += tmp

            ci += 2
          end
        end
      end
    end
    # off-diagonal elements
    # first computing the part of D'D corresponding to node x edge
    # iterate over edges and record two non-zero elements per edge
    # also record elements
    indEdge = 0
    for a=1:p
      for b=a+1:p
        indEdge += 1
        for l=1:L
          for k=1:K
            # node a, component k x edge (a,b) component l
            _ri = (a-1)*K + k
            _ci = p*K + (indEdge-1)*L + l
            tmp = partial_derivatives[_ci, 1] * partial_derivatives[_ri, 1]
            V[ci]   += tmp
            V[ci+1] += tmp

            # node b, component k x edge (a,b) component l
            _ri = (b-1)*K + k
            _ci = p*K + (indEdge-1)*L + l
            tmp = partial_derivatives[_ci, 2] * partial_derivatives[_ri, 1]
            V[ci+2] += tmp
            V[ci+3] += tmp

            ci += 4
          end
        end
      end
    end

    indEdge = 0
    for a=1:p
      for b=a+1:p
        indEdge += 1
        for l1=1:L
          for l2=1:L

            # edges that have a in common
            # edge (a,b), component l1 x edge (a, c), component l2
            for c=b+1:p
              indEdge2 = _getEdgeIndex(a,c,p)
              _ri = p*K + (indEdge-1)*L + l1
              _ci = p*K + (indEdge2-1)*L + l2
              tmp = partial_derivatives[_ci, 1] * partial_derivatives[_ri, 1]
              V[ci]   += tmp
              V[ci+1] += tmp
              ci += 2
            end

            # edges that have b in common
            # edge (a,b), component l1 x edge (c, b), component l2
            for c=a+1:b-1
              indEdge2 = _getEdgeIndex(c,b,p)
              _ri = p*K + (indEdge-1)*L + l1
              _ci = p*K + (indEdge2-1)*L + l2
              tmp = partial_derivatives[_ci, 2] * partial_derivatives[_ri, 2]
              V[ci]   += tmp
              V[ci+1] += tmp
              ci += 2
            end
            # edge (a,b), component l1 x edge (b, c), component l2
            for c=b+1:p
              indEdge2 = _getEdgeIndex(b,c,p)
              _ri = p*K + (indEdge-1)*L + l1
              _ci = p*K + (indEdge2-1)*L + l2
              tmp = partial_derivatives[_ci, 1] * partial_derivatives[_ri, 2]
              V[ci]   += tmp
              V[ci+1] += tmp
              ci += 2
            end

          end
        end
      end
    end

  end

  # construct indices for DD
  # diagonal elements
  ci = 1
  for a=1:p
    for k1=1:K
      I[ci] = (a-1)*K+k1
      J[ci] = (a-1)*K+k1
      ci += 1
      for k2=k1+1:K
        _ri = (a-1)*K+k1
        _ci = (a-1)*K+k2

        I[ci]   = _ri
        J[ci]   = _ci
        I[ci+1] = _ci
        J[ci+1] = _ri

        ci += 2
      end
    end
  end

  indEdge = 0
  for a=1:p
    for b=a+1:p
      indEdge += 1
      for l1=1:L
        I[ci] = p*K + (indEdge-1)*L+l1
        J[ci] = p*K + (indEdge-1)*L+l1

        ci += 1
        for l2=l1+1:L
          _ri = p*K + (indEdge-1)*L+l1
          _ci = p*K + (indEdge-1)*L+l2

          I[ci]   = _ri
          J[ci]   = _ci
          I[ci+1] = _ci
          J[ci+1] = _ri

          ci += 2
        end
      end
    end
  end

  indEdge = 0
  for a=1:p
    for b=a+1:p
      indEdge += 1
      for l=1:L
        for k=1:K
          # node a, component k x edge (a,b) component l
          _ri = (a-1)*K + k
          _ci = p*K + (indEdge-1)*L + l

          I[ci]   = _ri
          J[ci]   = _ci
          I[ci+1] = _ci
          J[ci+1] = _ri

          # node b, component k x edge (a,b) component l
          _ri = (b-1)*K + k
          _ci = p*K + (indEdge-1)*L + l

          I[ci+2] = _ri
          J[ci+2] = _ci
          I[ci+3] = _ci
          J[ci+3] = _ri

          ci += 4
        end
      end
    end
  end

  indEdge = 0
  for a=1:p
    for b=a+1:p
      indEdge += 1
      for l1=1:L
        for l2=1:L

          # edges that have a in common
          # edge (a,b), component l1 x edge (a, c), component l2
          for c=b+1:p
            indEdge2 = _getEdgeIndex(a,c,p)
            _ri = p*K + (indEdge-1)*L + l1
            _ci = p*K + (indEdge2-1)*L + l2

            I[ci]   = _ri
            J[ci]   = _ci
            I[ci+1] = _ci
            J[ci+1] = _ri

            ci += 2
          end

          # edges that have b in common
          # edge (a,b), component l1 x edge (c, b), component l2
          for c=a+1:b-1
            indEdge2 = _getEdgeIndex(c,b,p)
            _ri = p*K + (indEdge-1)*L + l1
            _ci = p*K + (indEdge2-1)*L + l2

            I[ci]   = _ri
            J[ci]   = _ci
            I[ci+1] = _ci
            J[ci+1] = _ri

            ci += 2
          end
          # edge (a,b), component l1 x edge (b, c), component l2
          for c=b+1:p
            indEdge2 = _getEdgeIndex(b,c,p)
            _ri = p*K + (indEdge-1)*L + l1
            _ci = p*K + (indEdge2-1)*L + l2

            I[ci]   = _ri
            J[ci]   = _ci
            I[ci+1] = _ci
            J[ci+1] = _ri

            ci += 2
          end

        end
      end
    end
  end

  sparse(I, J, V, numColumns, numColumns)
end

function getD{T<:FloatingPoint}(
    X::StridedMatrix{T},
    nodeBasis::NodeBasis,
    edgeBasis::EdgeBasis
    )
  n, p = size(X)
  K = nodeBasis.numBasis
  L = edgeBasis.numBasis
  numColumns = round(Int, p*K + p * (p-1) / 2 * L)
  partial_derivatives = zeros(Float64, numColumns, 2)
  numNZ = p*K + p * (p-1) * L
  V = zeros(Float64, numNZ*n)
  I = zeros(Int64, numNZ*n)
  J = zeros(Int64, numNZ*n)

  indNZ = 0
  @inbounds for i=1:n
    offsetRow = (i-1)*p
    # obtain partial derivatives to construct DD matrix
    _getCompactDi!(partial_derivatives, X, i, nodeBasis, edgeBasis)

    # process nodes
    for a=1:p
      for k=1:K
        indNZ += 1
        _ci = (a-1)*K+k
        I[indNZ] = offsetRow + a
        J[indNZ] = _ci
        V[indNZ] = partial_derivatives[_ci, 1]
      end
    end

    # process edges
    indEdge = 0
    @inbounds for a=1:p
      for b=a+1:p
        indEdge += 1
        # derivative with respect to first argument
        for l=1:L
          indNZ += 1
          _ci = p*K + (indEdge-1)*L + l
          I[indNZ] = offsetRow + a
          J[indNZ] = _ci
          V[indNZ] = partial_derivatives[_ci, 1]
        end
        # derivative with respect to second argument
        for l=1:L
          indNZ += 1
          _ci = p*K + (indEdge-1)*L + l
          I[indNZ] = offsetRow + b
          J[indNZ] = _ci
          V[indNZ] = partial_derivatives[_ci, 2]
        end
      end
    end
  end
  sparse(I, J, V, n*p, numColumns)
end


########################################
#
#  Matrix E
#
########################################

# returns vector E = sum_i E(x_i)
function getE(
    X::Array{Float64, 2},
    nodeBasis::NodeBasis,
    edgeBasis::EdgeBasis
    )

  n, p = size(X)
  K = nodeBasis.numBasis
  L = edgeBasis.numBasis
  node_der2 = nodeBasis.der2
  edge_der2 = edgeBasis.der2
  E = zeros(Float64, int(p*K+p*(p-1)/2*L))
  @inbounds for rowInd=1:n
    # process node elements
    for a=1:p
      for k=1:K
        _ci = (a-1)*K+k
        E[_ci] = E[_ci] + node_der2(X[rowInd, a], k)
      end
    end
    # process edges
    numEdges = 0
    for a=1:p
      for b=a+1:p
        numEdges += 1
        for l=1:L
          _ci = p*K + (numEdges-1)*L + l
          E[_ci] += edge_der2(X[rowInd, a], X[rowInd, b], l, 1)
          + edge_der2(X[rowInd, a], X[rowInd, b], l, 2)
        end
      end
    end
  end
  E
end

