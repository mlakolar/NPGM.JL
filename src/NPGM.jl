module NPGM


export
  graph_sample!,
  getE, getD!, getD, getDD,
  gm_node_f, gm_edge_f, gm_edge_der_f, gm_node_der_f, gm_edge_der_2_f, gm_node_der_2_f,
  getPrecision


include("graph_sample.jl")
include("gaussian_function_handlers.jl")
include("score_matching.jl")

end
