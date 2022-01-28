import graphicle as gcl

from showerpipe.pipeline._base import DataFilter


class KnnTransform(DataFilter):
    def __init__(self, num_neighbours: int, weighted: bool = False):
        self.k = num_neighbours
        self.weighted = weighted
        
    def apply(self, data: gcl.Graphicle) -> gcl.Graphicle:
        final_data = data[data.final]
        dR_aff = gcl.matrix.delta_R_aff(final_data.pmu)
        dR_adj = gcl.matrix.knn_adj(dR_aff, k=self.k, weighted=self.weighted)
        final_data.adj = gcl.AdjacencyList.from_matrix(
                dR_adj, weighted=self.weighted)
        return final_data
