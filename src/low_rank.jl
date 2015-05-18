
# immutable LowRankEigen{T,V,S<:AbstractMatrix,U<:AbstractVector} <: AbstractMatrix{T}
#   vectors::S
#   values::U
#   values_inv::U

#   function LowRankEigen(vectors::AbstractMatrix{T}, values::AbstractVector{V})
#     values_inv = similar(values)
#     @inbounds for i in eachindex(values)
#       values_inv[i] = 1. / values[i]
#     end
#     new(vectors, values, values_inv)
#   end
# end
# LowRankEigen{T,V}(vectors::AbstractMatrix{T}, values::AbstractVector{V}) = LowRankEigen{T,V,typeof(vectors),typeof(values)}(vectors, values)

# function LowRankEigen{T}(A::AbstractMatrix{T}; min_eigval::T=1e-5, nev::Int64=6)
#   p = size(A, 1)

#   D = Array(T, 0)
#   U = Array(T, p, 0)
#   numEig = 0
#   eigLeft = p

#   fDone = false
#   while eigLeft > inc_eig
#     Dnew, Unew, _ = eigs(A; nev=nev)

#     D = vcat(D, Dnew)
#   end
#   = Eigen{T,V,typeof(vectors),typeof(values)}(vectors, values)
# end

