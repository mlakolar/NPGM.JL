
module GaussianFunctions

import ..NPGM.NodeBasis
import ..NPGM.EdgeBasis

export GaussianNodeBasis, GaussianEdgeBasis

# two functions per node
# -x^2 / 2. and x
function GaussianNodeBasis{T<:FloatingPoint}(::T)
  function val(x::T, k::Int64)
    r = zero(T)
    if k == 1
      r = - x^2 / 2.
    elseif k == 2
      r = x
    end
    r
  end
  function der(x::T, k::Int64)
    r = zero(T)
    if k == 1
      r = -x
    elseif k == 2
      r = one(T)
    end
    r
  end
  function der2(x::T, k::Int64)
    k == 1 ? convert(T, -1.) : zero(T)
  end

  NodeBasis(val, der, der2, 2)
end

# one basis per edge
# -xy
function GaussianEdgeBasis{T<:FloatingPoint}(::T)
  function val(x::T, y::T, k::Int64)
    k == 1 ? -x*y : zero(T)
  end
  function der(x::T, y::T, k::Int64, whichArgument::Int64)
    r = zero(T)
    if k == 1
      if whichArgument == 1
        r = -y
      elseif whichArgument == 2
        r = -x
      else
        throw(ArgumentError("whichArgument should be either 1 or 2"))
      end
    end
    r
  end
  function der2(x::T, y::T, k::Int64, whichArgument::Int64)
    zero(T)
  end

  EdgeBasis(val, der, der2, 1)
end

end
