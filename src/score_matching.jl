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
function getD(
    X::Array{Float64, 2}, rowInd::Int64, K::Int64, L::Int64,
    phi_node_d::Function, phi_edge_d::Function)

    n, p = size(X)
    # will be used to create a sparse matrix at the end
    rI = Array(Int64, 0)
    cI = Array(Int64, 0)
    V = Array(Float64, 0)

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
function getD!(
    out::Array{Float64, 2},
    X::Array{Float64, 2}, rowInd::Int64, K::Int64, L::Int64,
    phi_node_d::Function, phi_edge_d::Function)

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
        getD!(partial_derivatives, X, i, K, L, phi_node_d, phi_edge_d)

        # diagonal elements
        for ci=1:p*K
            V[ci] += partial_derivatives[ci, 1]^2
        end
        for ci=(p*K+1):numColumns
            V[ci] += partial_derivatives[ci, 1]^2 + partial_derivatives[ci, 2]^2
        end

        # off-diagonal elements
        # first computing the part of D'D corresponding to node x edge
        # iterate over edges and record two non-zero elements per edge
        # also record elements
        indEdge = 0
        ci = numColumns + 1
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
    for ci=1:numColumns
        I[ci] = ci
        J[ci] = ci
    end

    indEdge = 0
    ci = numColumns + 1
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
                    E[_ci] = E[_ci] + phi_edge_d_2(X[rowInd, a], X[rowInd, b], l)
                end
            end
        end
    end
    E
end