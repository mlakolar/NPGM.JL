
module GaussianFunctions

import ..NPGM.NodeBasis
import ..NPGM.EdgeBasis

export GaussianNodeBasis, GaussianEdgeBasis

# two functions per node
# -x^2 / 2. and x
function GaussianNodeBasis()
  function val(x::FloatingPoint, k::Int64)
    k == 1 || k == 2 || throw(ArgumentError("k should be 1 or 2"))
    x = convert(Float64, x)
    k == 1 ? - x^2. / 2. : x
  end
  function der(x::FloatingPoint, k::Int64)
    k == 1 || k == 2 || throw(ArgumentError("k should be 1 or 2"))
    k == 1 ? -x : 1.
  end
  function der2(x::FloatingPoint, k::Int64)
    k == 1 || k == 2 || throw(ArgumentError("k should be 1 or 2"))
    k == 1 ? -1. : 0.
  end

  NodeBasis(val, der, der2, 2)
end

# one basis per edge
# -xy
function GaussianEdgeBasis()
  function val{T<:FloatingPoint}(x::T, y::T, ::Int64)
    -x*y
 end
  function der{T<:FloatingPoint}(x::T, y::T, ::Int64, whichArgument::Int64)
    whichArgument == 1 || whichArgument == 2 || throw(ArgumentError("whichArgument should be 1 or 2"))
    whichArgument == 1 ? -y : -x
  end
  function der2(x, y, ::Int64, ::Int64)
    0.
  end

  EdgeBasis(val, der, der2, 1)
end

end
