
module KernelFunctions

import ..NPGM.NodeBasis
import ..NPGM.EdgeBasis


export
  value, derivative, derivative2,
  HeatKernel


abstract MercerKernel{T}

value{T}(k::MercerKernel{T}, x::T, y::T) = throw(ArgumentError("value not implemented for this kernel."))
derivative{T}(k::MercerKernel{T}, x::T, y::T, whichArgument::Int64) = throw(ArgumentError("derivative not implemented for this kernel."))
derivative2{T}(k::MercerKernel{T}, x::T, y::T, whichArgument::Int64) = throw(ArgumentError("derivative2 not implemented for this kernel."))


immutable HeatKernel{T} <: MercerKernel{T}
  σ::T
  HeatKernel(a::T) = new(a)
end

value{T}(k::HeatKernel{T}, x::T, y::T) = exp( -(x-y)^2. / k.σ^2. )
function derivative{T}(k::HeatKernel{T}, x::T, y::T, whichArgument::Int64)
  t = k.σ^2.
  if whichArgument == 1
    return 2. * (y - x) * exp( -(x - y)^2. / t ) / t
  elseif whichArgument == 2
    return 2. * (x - y) * exp( -(x - y)^2. / t ) / t
  else
    throw(ArgumentError("whichArgument should be 1 or 2."))
  end
end

function derivative2{T}(k::HeatKernel{T}, x::T, y::T, whichArgument::Int64)
  t = k.σ^2.
  if whichArgument == 1
    return 2. * (y - x) * exp( -(x - y)^2. / t ) / t
  elseif whichArgument == 2
    return 2. * (x - y) * exp( -(x - y)^2. / t ) / t
  else
    throw(ArgumentError("whichArgument should be 1 or 2."))
  end
end


end
