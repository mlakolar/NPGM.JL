
module BSplineFunctions

import ..NPGM.NodeBasis
import ..NPGM.EdgeBasis
import BSpline

function BSplineNodeBasis(M, xmin, xmax, numKnots)
    bs = BSpline.BSplineT(M, xmin, xmax, numKnots)
    K = BSpline.num_basis(bs)

    f_node(x, k)      = BSpline.bspline_basis(k-1, bs, x)
    f_node_der(x, k)  = BSpline.derivative_bspline_basis(k-1, bs, x)
    f_node_der2(x, k) = BSpline.derivative2_bspline_basis(k-1, bs, x)

    NodeBasis(f_node, f_node_der, f_node_der2, K)
end

function BSplineEdgeBasis(M, xmin, xmax, numKnots)
    bs1 = BSpline.BSplineT(M, xmin[1], xmax[1], numKnots[1])
    bs2 = BSpline.BSplineT(M, xmin[2], xmax[2], numKnots[2])
    L1 = BSpline.num_basis(bs1)
    L2 = BSpline.num_basis(bs2)

    function f_edge(x, y, l)
        r, c = ind2sub((L1, L2), l)
        r -= 1
        c -= 1
        return BSpline.bspline_basis(r, bs1, x) * BSpline.bspline_basis(c, bs2, y)
    end
    function f_edge_der(x, y, l, whichArgument)
        r, c = ind2sub((L1, L2), l)
        r -= 1
        c -= 1
        if whichArgument == 1
            return BSpline.derivative_bspline_basis(r, bs1, x) * BSpline.bspline_basis(c, bs2, y)
        else
            return BSpline.bspline_basis(r, bs1, x) * BSpline.derivative_bspline_basis(c, bs2, y)
        end
    end
    function f_edge_der2(x, y, l, whichArgument)
        r, c = ind2sub((L1, L2), l)
        r -= 1
        c -= 1
        if whichArgument == 1
            return BSpline.derivative2_bspline_basis(r, bs1, x) * BSpline.bspline_basis(c, bs2, y)
        else
            return BSpline.bspline_basis(r, bs1, x) * BSpline.derivative2_bspline_basis(c, bs2, y)
        end
    end

    EdgeBasis(f_edge, f_edge_der, f_edge_der2, L1*L2)
end

end
