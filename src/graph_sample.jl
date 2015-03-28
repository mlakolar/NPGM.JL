# sample from nonparametric graphical model on R^p
# p(x) ~ exp{sum_{node i} f_i(x_i) + sum_{edge ij} f_ij(x_i,x_j)}
#   where f_i(x_i) = sum_k beta_ik phi_node_k(x_i)
#   and f_ij(x_i,x_j) = sum_l beta_ijl phi_edge_l(x_i,x_j)
#
# input:
# n = sample size
# edges = m-by-2 matrix with edge(l,:) giving the vertices of the l-th edge
# gibbs_groups = p-vector partitioning the set of nodes for gibbs sampling
#       satisfies: if nodes i,j have same label, then (i,j) is not an edge
# phi_node_hdl = function handle pointing to the basis functions
#           which evaluates the full list of basis functions at any x,
#               i.e. phi_node_hdl maps from R to R^K
#           if input is a vector in R^q, output is q-by-K matrix
# phi_edge_hdl = same for edges: maps R^2 to R^L or R^{q*2} to R^{q*L}
# mh_sampler_hdl = proposed step for metropolis-hastings
#       i.e. mh_sampler_hdl(z1) draws a proposed value z2
#               from starting value z1 (sampe function for all nodes)
#       if input is vector, output is vector (i.e. steps z1(i) to z2(i))
# mh_ratio_hdl = gives step distribution ratio for proposal step for m-h
#       i.e. mh_probs_hdl(z1,z2) gives P(z1->z2)/P(z2->z1)
#       if input is vector, output is vector (calculated elementwise)
# beta_node = p-by-K matrix giving beta_ik coefficients
# beta_edge = m-by-L array giving beta_ijl where (i,j)=mth edge
#       (stored as m-by-L rather than p-by-p-by-L because m<<p^2).
#
# output:
# X = n-by-p matrix containing n draws from the distribution
#


function graph_sample!(X::Array{Float64,2},
                       n::Int64, edges::Array{Int64,2}, gibbs_groups::Array{Int64, 1},
                       phi_node_hdl::Function, phi_edge_hdl::Function, mh_sampler_hdl::Function, mh_ratio_hdl::Function,
                       beta_node::Array{Float64,2}, beta_edge::Array{Float64,2};
                       burnin::Int64=100, gap::Int64=100)

  numAccept = 0
  numReject = 0

  G = maximum(gibbs_groups)
  groups = cell(G)
  for g=1:G
    groups[g] = find(gibbs_groups .== g)
  end

  p, K = size(beta_node)
  m, L = size(beta_edge)
  fill!(X, 0.)
  x = zeros(Float64, p)
  xnew = zeros(Float64, p)

  node_to_edge = Array(Vector{Int64}, p)
  for i=1:p
    node_to_edge[i] = Array(Int64, 0)
  end
  for mm=1:m
    push!(node_to_edge[edges[mm,1]], mm)
    push!(node_to_edge[edges[mm,2]], mm)
  end

  numSampled = 0
  for iter=1:(burnin+gap*(n-1))
    #
    # Gibbs sampler cycles through groups
    # within each group, run Metropolis-Hastings in parallel across the nodes
    for g=1:G
      group=groups[g]
      copy!(xnew, x)
      for elem=group
        xnew[elem] = mh_sampler_hdl(x[elem])
      end
      for ii=1:length(group)
        mh_ratios = mh_ratio_hdl(x[group[ii]], xnew[group[ii]])
        logprob_change = 0.
        for k=1:K
          logprob_change += beta_node[group[ii],k] * (phi_node_hdl(xnew[group[ii]], k) - phi_node_hdl(x[group[ii]], k))
        end
        for mm=node_to_edge[group[ii]]
          for l=1:L
            logprob_change += beta_edge[mm,l] *
              (phi_edge_hdl(xnew[edges[mm,1]], xnew[edges[mm,2]], l) - phi_edge_hdl(x[edges[mm,1]], x[edges[mm,2]], l))
          end
        end
        accept_prob = exp(logprob_change) * mh_ratios
        if rand() <= accept_prob
          x[group[ii]] = xnew[group[ii]]
          numAccept += 1
        else
          numReject += 1
        end
      end
    end

    # store observation according to burnin / gap
    if (iter >= burnin) && mod(iter-burnin,gap) == 0
      numSampled += 1
      X[numSampled, :] = x
    end
  end

  (numAccept, numReject)
end
