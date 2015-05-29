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

## GaussianKernel
#
#  k(x,y) = exp( -(x-y)^2 / σ^2 )
#
immutable GaussianKernel{T} <: MercerKernel{T}
  σ::T
  GaussianKernel(a::T) = new(a)
end
value{T}(k::GaussianKernel{T}, x::T, y::T) = exp( -(x-y)^2. / k.σ^2. )
function derivative{T}(k::GaussianKernel{T}, x::T, y::T, whichArgument::Int64)
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
  if whichArgument == 1 || whichArgument == 2
    ds = (x - y)^2.
    return (4. * ds / t^2. - 2. / t ) * exp( - ds / t )
  else
    throw(ArgumentError("whichArgument should be 1 or 2."))
  end
end


end
