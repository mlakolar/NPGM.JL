
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

# function LowRankEigen{T, SYM}(A::AbstractMatrix{T},
#                          symmetric::Type{Val{SYM}} = Val{false};
#                          min_eigval::T=1e-5)
function LowRankEigen{T}(A::AbstractMatrix{T},
                         symmetric=false;
                         min_eigval::T=1e-5)
  if symmetric
    p = size(A, 1)
    o = eigfact(A)
    values = o.values
    vectors = o.vectors
    ind = find(values .>= min_eigval)
    values = values[ind]
    vectors = vectors[:, ind]
  else
    n, p = size(A)
    o = svdfact(A)
    values = o.S
    vectors = o[:V]
    k = 0
    @inbounds for i in eachindex(values)
      values[i] = values[i]^2. / n
      if values[i] < min_eigval
        break
      end
      k += 1
    end
    resize!(values, k)
    vectors = vectors[:, 1:k]
  end
  values_inv = one(T) ./ values

  LowRankEigen(vectors, values, values_inv)
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
# \(A::LowRankEigen, B::StridedVecOrMat) = A.vectors*(Diagonal(A.values_inv)*(A.vectors'B))
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
