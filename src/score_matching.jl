########################################
#
#  Matrix D
#
########################################

# suppose that p(x) ~ exp{ t(x)' beta }
# where  t(x), beta ∈ R^m with X ∈ R^p  then the matrix D
# is of size p x m

# This function returns D(x_i) as a sparse matrix of size
# p x (p*K + p * (p-1) / 2 * L)
# note that the function is implemented only for pairwise graphical models
## TODO:
# could be improved by preallocating
function getDi{T<:FloatingPoint}(
    X::StridedMatrix{T}, rowInd::Int64,
    K::Int64, L::Int64,
    phi_node_d::Function, phi_edge_d::Function)

    n, p = size(X)
    # will be used to create a sparse matrix at the end
    rI = Array(Int64, 0)
    cI = Array(Int64, 0)
    V = Array(T, 0)

    # process node elements
    for a=1:p
        _ri = a
        for k=1:K
            _ci = (a-1)*K+k
            _v = phi_node_d(X[rowInd, a], k)

            push!(rI, _ri)
            push!(cI, _ci)
            push!(V, _v)
        end
    end
    # process edges
    indEdge = 0
    for a=1:p
        for b=a+1:p
            indEdge += 1

            # derivative with respect to first argument
            _ri = a
            for l=1:L
                _ci = p*K + (indEdge-1)*L + l
                _v = phi_edge_d(X[rowInd, a], X[rowInd, b], l, 1)

                push!(rI, _ri)
                push!(cI, _ci)
                push!(V, _v)
            end

            # derivative with respect to second argument
            _ri = b
            for l=1:L
                _ci = p*K + (indEdge-1)*L + l
                _v = phi_edge_d(X[rowInd, a], X[rowInd, b], l, 2)

                push!(rI, _ri)
                push!(cI, _ci)
                push!(V, _v)
            end

        end
    end

    sparse(rI, cI, V, p, int(p*K+p*(p-1)/2*L))
end

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
    K::Int64,
    L::Int64,
    phi_node_d::Function,
    phi_edge_d::Function)

    n, p = size(X)
    # process node elements
    for a=1:p
        for k=1:K
            _ci = (a-1)*K+k
            out[_ci, 1] = phi_node_d(X[rowInd, a], k)
        end
    end
    # process edges
    indEdge = 0
    for a=1:p
        for b=a+1:p
            indEdge += 1

            # derivative with respect to first argument
            for l=1:L
                _ci = p*K + (indEdge-1)*L + l
                out[_ci, 1] = phi_edge_d(X[rowInd, a], X[rowInd, b], l, 1)
            end

            # derivative with respect to second argument
            for l=1:L
                _ci = p*K + (indEdge-1)*L + l
                out[_ci, 2] = phi_edge_d(X[rowInd, a], X[rowInd, b], l, 2)
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
function getEdgeIndex(a::Int64, b::Int64, p::Int64)
    (a-1) * p - int((a-1)*a/2) + (b-a)
end


# constructs matrix DD = sum_i D(x_i)'D(x_i)
function getDD(X::Array{Float64, 2}, K::Int64, L::Int64,
               phi_node_d::Function, phi_edge_d::Function)

  n, p = size(X)
  numColumns = int(p*K + p * (p-1) / 2 * L)
  partial_derivatives = zeros(Float64, numColumns, 2)

  # number of non-zero elements in D'D
  numNZ = p * K * K +
    int(L * L * p * (p-1) / 2) +
    2 * (p-1) * K * L * p +
    L * L * p * (p - 1) * (p - 2)
  V = zeros(Float64, numNZ)
  I = zeros(Int64, numNZ)
  J = zeros(Int64, numNZ)

  for i=1:n
    # obtain partial derivatives to construct DD matrix
    _getCompactDi!(partial_derivatives, X, i, K, L, phi_node_d, phi_edge_d)

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
              indEdge2 = getEdgeIndex(a,c,p)
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
              indEdge2 = getEdgeIndex(c,b,p)
              _ri = p*K + (indEdge-1)*L + l1
              _ci = p*K + (indEdge2-1)*L + l2
              tmp = partial_derivatives[_ci, 2] * partial_derivatives[_ri, 2]
              V[ci]   += tmp
              V[ci+1] += tmp
              ci += 2
            end
            # edge (a,b), component l1 x edge (b, c), component l2
            for c=b+1:p
              indEdge2 = getEdgeIndex(b,c,p)
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
            indEdge2 = getEdgeIndex(a,c,p)
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
            indEdge2 = getEdgeIndex(c,b,p)
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
            indEdge2 = getEdgeIndex(b,c,p)
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



getNeighborhoodD{T<:FloatingPoint}(
  X::StridedMatrix{T},
  nodeInd::Int64,
  K::Int64,
  L::Int64,
  phi_node_d::Function,
  phi_edge_d::Function
  ) = getNeighborhoodD!(zeros(size(X, 1), K+(size(X,2)-1)*L), X, nodeInd, K, L, phi_node_d, phi_edge_d)



# This function returns D matrix of size n x (K + (p-1) * L)
# where
#     K is the number of basis per node
#     L is the number of basis per edge
# note that the function is implemented only for pairwise graphical models
function getNeighborhoodD!{T<:FloatingPoint}(
    out::StridedMatrix{T},
    X::StridedMatrix{T},
    nodeInd::Int64,
    K::Int64,
    L::Int64,
    phi_node_d::Function,
    phi_edge_d::Function
    )

  n, p = size(X)

  for rowInd=1:n
    # process node
    @inbounds for k=1:K
      out[rowInd, k] = phi_node_d(X[rowInd, nodeInd], k)
    end

    # process edges
    indEdge = 0
    @inbounds for b=1:p
      if nodeInd == b
        continue
      end
      indEdge += 1
      # derivative with respect to first argument
      for l=1:L
        out[rowInd, K + (indEdge-1)*L + l] = phi_edge_d(X[rowInd, nodeInd], X[rowInd, b], l, 1)
      end
    end
  end
  out
end

########################################
#
#  Matrix E
#
########################################

# returns vector E = sum_i E(x_i)
function getE(
    X::Array{Float64, 2}, K::Int64, L::Int64,
    phi_node_d_2::Function, phi_edge_d_2::Function)

    n, p = size(X)
    E = zeros(Float64, int(p*K+p*(p-1)/2*L))
    for rowInd=1:n
        # process node elements
        for a=1:p
            for k=1:K
                _ci = (a-1)*K+k
                E[_ci] = E[_ci] + phi_node_d_2(X[rowInd, a], k)
            end
        end
        # process edges
        numEdges = 0
        for a=1:p
            for b=a+1:p
                numEdges += 1
                for l=1:L
                    _ci = p*K + (numEdges-1)*L + l
                    E[_ci] += phi_edge_d_2(X[rowInd, a], X[rowInd, b], l, 1)
                               + phi_edge_d_2(X[rowInd, a], X[rowInd, b], l, 2)
                end
            end
        end
    end
    E
end

# returns matrix E  of size n x K + (p-1)*L
# first K components correspond to nodeInd
# next (p-1)*L elements correpond to edges
function getNeighborhoodE!{T<:FloatingPoint}(
    out::StridedMatrix{T},
    X::StridedMatrix{T},
    nodeInd::Int64,
    K::Int64,
    L::Int64,
    phi_node_d_2::Function,
    phi_edge_d_2::Function
    )
  n, p = size(X)
  @inbounds for rowInd=1:n
    # process node
    for k=1:K
      out[rowInd, k] = phi_node_d_2(X[rowInd, nodeInd], k)
    end

    # process edges
    numEdges = 0
    for b=1:p
      if nodeInd == b
        continue
      end
      numEdges += 1
      for l=1:L
        out[rowInd, K + (numEdges-1)*L + l] = phi_edge_d_2(X[rowInd, nodeInd], X[rowInd, b], l, 1)
      end
    end
  end
  out
end

getNeighborhoodE{T<:FloatingPoint}(
  X::StridedMatrix{T},
  nodeInd::Int64,
  K::Int64,
  L::Int64,
  phi_node_d_2::Function,
  phi_edge_d_2::Function
  ) = getNeighborhoodE!(zeros(size(X, 1), K+(size(X,2)-1)*L), X, nodeInd, K, L, phi_node_d_2, phi_edge_d_2)



#########################################################
#
#   prepare data -- helper functions
#
#########################################################


function createNodeFunction(M, xmin, xmax, numKnots)

    bs = BSpline.BSplineT(M, xmin, xmax, numKnots)
    K = BSpline.num_basis(bs)

    f_node(x, k)      = BSpline.bspline_basis(k-1, bs, x)
    f_node_der(x, k)  = BSpline.derivative_bspline_basis(k-1, bs, x)
    f_node_der2(x, k) = BSpline.derivative2_bspline_basis(k-1, bs, x)

    (f_node, f_node_der, f_node_der2, K)
end

function createEdgeFunction(M, xmin, xmax, numKnots)

    bs1 = BSpline.BSplineT(M, xmin[1], xmax[1], numKnots[1])
    bs2 = BSpline.BSplineT(M, xmin[2], xmax[2], numKnots[2])
    L1 = BSpline.num_basis(bs1)
    L2 = BSpline.num_basis(bs2)

    function f_edge(x, y, l)
        r, c = ind2sub((L1, L2), l)
        r -= 1
        c -= 1
        return BSpline.bspline_basis(r, bs1, x) * BSpline.bspline_basis(c, bs2, y)
    end
    function f_edge_der(x, y, l, whichArgument)
        r, c = ind2sub((L1, L2), l)
        r -= 1
        c -= 1
        if whichArgument == 1
            return BSpline.derivative_bspline_basis(r, bs1, x) * BSpline.bspline_basis(c, bs2, y)
        else
            return BSpline.bspline_basis(r, bs1, x) * BSpline.derivative_bspline_basis(c, bs2, y)
        end
    end
    function f_edge_der2(x, y, l, whichArgument)
        r, c = ind2sub((L1, L2), l)
        r -= 1
        c -= 1
        if whichArgument == 1
            return BSpline.derivative2_bspline_basis(r, bs1, x) * BSpline.bspline_basis(c, bs2, y)
        else
            return BSpline.bspline_basis(r, bs1, x) * BSpline.derivative2_bspline_basis(c, bs2, y)
        end
    end

    (f_edge, f_edge_der, f_edge_der2, L1*L2)
end

# takes data as input and prepares everything for optimization
# X data matrix
# node for which we want to f
function prepare_data_for_neighborhood_opt{T<:FloatingPoint}(
    X::Matrix{T},
    nodeInd::Int64,
    M::Int64,
    numKnotsNode::Int64,
    numKnotsEdge::Int64,
    xmin::T,
    xmax::T
    )

  f_node, f_node_der, f_node_der2, K = createNodeFunction(M, xmin, xmax, numKnotsNode)
  f_edge, f_edge_der, f_edge_der2, L = createEdgeFunction(M, (xmin, xmin), (xmax, xmax), (numKnotsEdge, numKnotsEdge))

  D = getNeighborhoodD(X, nodeInd, K, L, f_node_der, f_edge_der)
  E = getNeighborhoodE(X, nodeInd, K, L, f_node_der2, f_edge_der2)

  n, p = size(X)
  θ = zeros(K+(p-1)*L)

  groups = Array(UnitRange{Int64}, p)           #### TODO: change this line
  groups[1] = 1:K
  for t=1:p-1
    groups[t+1] = (K + (t-1)*L+1):(K + t*L)
  end

  A = D'D / n
  b = vec( sum(E, 1) / n )

  #
  θ, A, b, groups, K, L, f_node, f_node_der, f_node_der2, f_edge, f_edge_der, f_edge_der2
end




#########################################################
#
#   diagnostics
#
#########################################################


function trapz{T<:FloatingPoint}(Y::Vector{T}, xmin::T, xmax::T)
  ((sum(Y) - (Y[1] + Y[end]) / 2.) * (xmax - xmin) / (length(Y) - 1))
end


type CondDensity
  θ::Vector{Float64}
  indNode::Int64
  f_node::Function
  f_edge::Function
  K::Int64
  L::Int64
  xmin::Float64
  xmax::Float64
end

function condDensity{T<:FloatingPoint}(
    ed::CondDensity,
    condX::Vector{T},
    N::Int64
    )

  θ = ed.θ
  K = ed.K
  L = ed.L
  indNode = ed.indNode
  p = length(condX)

  evalXs = [linspace(ed.xmin, ed.xmax, N);]
  pX = zeros(length(evalXs))
  for i=1:N
    ex = evalXs[i]
    tmp = zero(T)
    for k=1:K
      tmp += ed.f_node(ex, k) * θ[k]
    end

    indEdge = 0
    for b=1:p
      if b == indNode
        continue
      end
      indEdge += 1
      for l=1:L
        tmp += ed.f_edge(ex, condX[b], l) * θ[K + (indEdge-1)*L + l]
      end
    end

    pX[i] = exp(tmp)
  end

  scale!(pX, 1. / trapz(pX, ed.xmin, ed.xmax))
  evalXs, pX
end


