
module RKHSFunctions

import ..NPGM.NodeBasis
import ..NPGM.EdgeBasis


#gaussian kernel
function GK{T<:FloatingPoint}(x::Vector{T}, y::Vector{T}, sigma::T)
	return exp(-1/2/sigma^2 * vecdot(x-y, x-y) )
end

#general framework to compute \xi
function GK_Basis2_General{T<:FloatingPoint}(
						data::Vector{T}, X::Array{T}, sigma::T, Edge::Bool, Der::Function)
	n = size(X)[1]
	res = 0.0
	if Edge
		for t=1:n
			res = res + Der(data, vec(X[t,:]), 1) + Der(data, vec(X[t,:]), 2) 
		end
	else
		for t=1:n
			res = res + Der(data, vec(X[t,:]), 1) 
		end
	end
	return res/ n
end

function GKNodeBasis{T<:FloatingPoint}(
		     X::Vector{T}, parameters::Vector{T})
	n = length(X); sigma = parameters[1]
	function node_val(x::T, k::Int64)
		if k<=n	
			return (x-X[k])/sigma^2 * GK([x], [X[k]], sigma)
		end
		if k == n + 1
			function Der(x::Vector{T}, y::Vector{T}, j::Int64)
				return (-1 / sigma^2 + (x[1]-y[1])^2/sigma^4) * GK(x, y, sigma)
			end
			return GK_Basis2_General([x], X, sigma, false, Der) 
		else
			throw(ArgumentError("k should be less than n + 1") )
		end
	end

	function node_der(x::T, k::Int64)
		if k<=n	
			return ( 1/ sigma^2 - (x-X[k])^2/sigma^4) * GK([x], [X[k]], sigma) 
		end
		if k == n + 1
			function Der(x::Vector{T}, y::Vector{T}, j::Int64)
				val = 3/sigma^4*(x[1]-y[1]) - (x[1]-y[1])^3/sigma^6
				return val * GK(x, y, sigma)
			end
			return GK_Basis2_General([x], X, sigma, false, Der)
		else
			throw(ArgumentError("k should be less than n + 1") )
		end
	end

	function node_der2(x::T, k::Int64)
		if k<=n	
			val = -3/sigma^4 * (x-X[k]) + (x-X[k])^3/ sigma^6
			return val * GK([x], [X[k]], sigma) 
		end
		if k == n + 1
			function Der(x::Vector{T}, y::Vector{T}, j::Int64)
				val = 3/sigma^4  - 6 * (x[1]-y[1])^2 /sigma^6 + (x[1]-y[1])^4/sigma^8
				return val * GK(x, y, sigma)
			end
			return GK_Basis2_General([x], X, sigma, false, Der)
		else
			throw(ArgumentError("k should be less than n + 1") )
		end
	end
	return NodeBasis(node_val, node_der, node_der2, n+1)
end

function GKEdgeBasis{T<:FloatingPoint}(X::StridedMatrix{T}, parameters::Vector{T})
    n, p = size(X); sigma = parameters[1]
    # k = 1, ..., n for \partial x_1,         n+1, ..., 2n for \partial x_2,    2n+1 for \Xi
    function val(a::T, b::T, k::Int64)
        x = [a,b];
        if k<=n
            y = vec( X[k,:] )
            return (x[1] - y[1])/sigma^2 * GK(x, y, sigma)
        end
        if k<= 2*n
            y = vec( X[k-n, :] )
            return (x[2] - y[2])/sigma^2 * GK(x, y, sigma)
        end
        if k == 2*n + 1
            function Der(x::Vector{T}, y::Vector{T}, j::Int64)                                            
                return (-1/sigma^2 + (x[j]-y[j])^2/sigma^4) * GK(x,y,sigma)                               
            end
            return GK_Basis2_General([a,b], X, sigma, true, Der)                                          
        else                                                                                              
            throw(ArgumentError("k should be less than 2*n + 1") )                                        
        end
    end

    function der(a::T, b::T, k::Int64, i::Int64)
        x = [a,b]
        if k<= 2* n
            if k <= n
                y = vec(X[k,:]);   j = 1
            else
                y = vec(X[k-n,:]); j=2
            end
            return ( (i==j)/sigma^2 - (x[i]-y[i])*(x[j]-y[j])/sigma^4 ) * GK(x,y,sigma)
        end
        if k == 2*n + 1
            function Der(x::Vector{T}, y::Vector{T}, j::Int64)
                val = (1+2*(i==j))/sigma^4*(x[i]-y[i]) - (x[i]-y[i])*(x[j]-y[j])^2/sigma^6
                return val * GK(x,y,sigma)
            end
            return GK_Basis2_General(x, X, sigma, true, Der)
        else
            throw(ArgumentError("k should be less than 2*n + 1") )
        end
    end
    function der2(a::T, b::T, k::Int64, i::Int64)
        x = [a,b]
        if k <= 2*n
            if k <= n
                y = vec( X[k,:] );   j = 1
            else
                y = vec( X[k-n,:] ); j = 2
            end
            val = -(1 + 2*(i==j))/sigma^4* (x[j]-y[j]) + (x[i]-y[i])^2*(x[j]-y[j])/sigma^6
            return val * GK(x, y, sigma)
        end
        if k == 2*n + 1
            function Der(x::Vector{T}, y::Vector{T}, j::Int64)
                val1 = (1+2*(i==j) )/sigma^4
                val2 = ( -(x[i]-y[i])^2 - (x[j]-y[j])^2 - 4 * (i==j)*(x[i]-y[i])^2 ) / sigma^6
                val3 = (x[i]-y[i])^2 * (x[j]-y[j])^2 / sigma^8
                return (val1 + val2 + val3) * GK(x, y, sigma)
            end
            return GK_Basis2_General(x, X, sigma, true, Der)
        else
            throw(ArgumentError("k should be less than 2*n + 1") )
        end
    end
    return EdgeBasis(val, der, der2, 2*n+1);
end

end
