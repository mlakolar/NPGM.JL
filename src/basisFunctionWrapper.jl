#import ..NPGM.NodeBasis
#import ..NPGM.EdgeBasis
import NPGM.BSplineFunctions
import NPGM.RKHSFunctions

function createGKNodeBasis(X, h) 
    function createNodeBasis( ind ) 
        GKNodeBasis(X[:,ind], h)
    end
    return createNodeBasis
end

function createGKEdgeBasis(X, h)
    function createEdgeBasis(ind1, ind2)
        GKEdgeBasis(X[:,[ind1, ind2]], hs)
    end
    return createEdgeBasis
end

function createBSplineNodeBasis(M, xmin, xmax, numKnots)
    nb = NPGM.BSplineFunctions.BSplineNodeBasis(M, xmin, xmax, numKnots)
    function createNodeBasis(ind)
        return nb
    end
    return createNodeBasis
end

function createBSplineEdgeBasis(M, xmin, xmax, numKnots)
    nb = NPGM.BSplineFunctions.BSplineEdgeBasis(M, xmin, xmax, numKnots)
    function createEdgeBasis(ind1, ind2)
        return nb
    end
    return createEdgeBasis
end