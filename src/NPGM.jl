module NPGM

import ProximalOPT
using Base.LinAlg.BLAS: gemv!
using Compat

export
  #types
  NodeBasis, EdgeBasis,

  # full matrices
  getDD, getE,

  # neigborhood matrices
  getNeighborhoodD,getNeighborhoodD!,
  getNeighborhoodE,getNeighborhoodE!,

  # utils
  condDensity,
  prepareLowRankNeighborhoodData,
  estimate_neighborhood,
  getNeighborhood,
  precision, recall,
  LowRankEigen,

  # sampler
  graph_sample!


# this object represents basis used to compute f_a(x_a)
# function val -- val(x, k) where k=1..K
# function der -- der(x, k) where k=1..K
# function der2 -- der2(x, k) where k=1..K
immutable NodeBasis
  val::Function
  der::Function
  der2::Function
  numBasis::Int64
end

# this object represents basis used to compute f_ab(x_a, x_b)
# function val -- val(x, y, k) where k=1..L
# function der -- der(x, y, k, which_arg) where k=1..L, which_arg tells if we are taking derivative wrt arg 1 or 2
# function der2 -- der2(x, y, k, which_arg) where k=1..L
immutable EdgeBasis
  val::Function
  der::Function
  der2::Function
  numBasis::Int64
end

include("fullMatrices.jl")
include("neighborhoodMatrices.jl")
include("gaussianBasis.jl")
include("bsplineBasis.jl")
include("utils.jl")
include("low_rank.jl")

include("graph_sample.jl")

include("heatBasis.jl")



end
