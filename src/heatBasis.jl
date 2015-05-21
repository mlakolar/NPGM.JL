#abstract Kernel
#immutable HeatKernel{T} <: Kernel
#  sigma2::T
#end
#HeatKernel{T}(a::T) = HeatKernel{T}(a)

module HeatFunctions
import ..NPGM.NodeBasis
import ..NPGM.EdgeBasis

export HeatNodeBasis, HeatEdgeBasis

# two functions per node
# -x^2 / 2. and x
function HeatNodeBasis{T<:FloatingPoint}(::T)
  function val(x::T, k::Int64)
    r = one(T)
    r
  end
  function der(x::T, k::Int64)
    r = zero(T)
    r
  end
  function der2(x::T, k::Int64)
    r = zero(T)
    r
  end

  NodeBasis(val, der, der2, 1)
end

# one basis per edge
# -xy
function HeatEdgeBasis{T<:FloatingPoint}(::T)
  function val(x::T, y::T, k::Int64)
    k == 1 ? exp(-(x-y)^2) : zero(T)
  end
  function der(x::T, y::T, k::Int64, whichArgument::Int64)
    r = zero(T)
    if k == 1
      if whichArgument == 1
        r = 2*(y-x)*exp(-(x-y)^2)
      elseif whichArgument == 2
        r = 2*(x-y)*exp(-(x-y)^2)
      else
        throw(ArgumentError("whichArgument should be either 1 or 2"))
      end
    end
    r
  end
  function der2(x::T, y::T, k::Int64, whichArgument::Int64)
    vr = zero(T)
    if k == 1
      if whichArgument == 1
        r = (4*(y-x)^2 - 2) * exp(-(x-y)^2)
      elseif whichArgument == 2
        r = (4*(y-x)^2 - 2) * exp(-(x-y)^2)
      else
        throw(ArgumentError("whichArgument should be either 1 or 2"))
      end
    end
    r
  end

  EdgeBasis(val, der, der2, 1)
end

end
