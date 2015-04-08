function gm_node_f(x::Float64, k::Int64)
    k == 1 ? - x^2 / 2. : 0.
end

function gm_edge_f(x::Float64, y::Float64, k::Int64)
    k == 1 ? - x * y : 0.
end

function gm_edge_der_f(x::Float64, y::Float64, k::Int64, whichArgument::Int64)
    if k == 1
        if whichArgument == 1
            return -y
        elseif whichArgument == 2
            return -x
        else
            throw(ArgumentError("whichArgument should be either 1 or 2"))
        end
    else
        return 0.
    end
end

function gm_node_der_f(x::Float64, k::Int64)
    k == 1 ? - x : 0.
end

function gm_edge_der_2_f(x::Float64, y::Float64, k::Int64, whichArgument::Int64)
    return 0.
end

function gm_node_der_2_f(x::Float64, k::Int64)
    k == 1 ? -1. : 0.
end

# extract the precision matrix from solution obtained by score matching
# out is where the precision matrix will be stored and should be allocated of size p x p
function getPrecision(out::Array{Float64, 2}, p::Int64, input::Array{Float64, 1})
    for a=1:p
        out[a,a] = input[a]
    end

    indEdge = 0
    for a=1:p
        for b=a+1:p
            indEdge += 1

            out[a,b] = input[p + indEdge]
            out[b,a] = input[p + indEdge]
        end
    end
end
