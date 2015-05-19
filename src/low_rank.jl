
immutable LowRankEigen{T,V,S<:StridedMatrix,U<:StridedVector} <: AbstractMatrix{T}
  vectors::S
  values::U
  values_inv::U
  #
  tmp1::Vector{T}        # used for division without allocation

  LowRankEigen(vectors::StridedMatrix{T}, values::StridedVector{V}, values_inv::StridedVector{V}) =
    new(vectors, values, values_inv, similar(values))
end

LowRankEigen{T,V}(vectors::AbstractMatrix{T}, values::AbstractVector{V}, values_inv::AbstractVector{V}) =
  LowRankEigen{T,V,typeof(vectors),typeof(values)}(vectors, values, values_inv)

function LowRankEigen{T,V}(vectors::AbstractMatrix{T}, values::AbstractVector{V})
  values_inv = similar(values)
  @inbounds for i in eachindex(values)
    values_inv[i] = 1. / values[i]
  end
  LowRankEigen{T,V,typeof(vectors),typeof(values)}(vectors, values, values_inv)
end


function Base.size(A::LowRankEigen)
  m = size(A.vectors, 1)
  (m, m)
end
Base.full{T}(A::LowRankEigen{T}) = convert(Matrix{T}, A)
Base.convert{T}(::Type{Matrix{T}}, A::LowRankEigen{T}) = A.vectors * (Diagonal(A.values) * A.vectors')

Base.copy(A::LowRankEigen) = LowRankEigen(A.vectors, A.values, A.values_inv)

#
# *(A::LowRankEigen, B::StridedVecOrMat) = A.vectors*(Diagonal(A.values)*(A.vectors'B))
function \(A::LowRankEigen, B::StridedVector)
  tmp1 = A.tmp1
  At_mul_B!(tmp1, A.vectors, B)
  @inbounds for i in eachindex(tmp1)
    tmp1[i] = tmp1[i] * A.values_inv[i]
  end
  A.vectors*tmp1
end
# Base.det(A::LowRankEigen) = prod(A.values)
# Base.trace(A::LowRankEigen) = sumabs(A.values)

function Base.A_ldiv_B!{T<:FloatingPoint}(A::LowRankEigen{T}, B::AbstractVector{T})
  tmp1 = A.tmp1
  At_mul_B!(tmp1, A.vectors, B)
  @inbounds for i in eachindex(tmp1)
    tmp1[i] = tmp1[i] * A.values_inv[i]
  end
  A_mul_B!(B, A.vectors, tmp1)
end

function Base.A_mul_B!{T<:FloatingPoint}(out::StridedVector{T}, A::LowRankEigen{T}, B::StridedVector{T})
  tmp1 = A.tmp1
  At_mul_B!(tmp1, A.vectors, B)
  @inbounds for i in eachindex(tmp1)
    tmp1[i] = tmp1[i] * A.values[i]
  end
  A_mul_B!(out, A.vectors, tmp1)
end
