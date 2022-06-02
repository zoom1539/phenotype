import numpy as np
from ..utils import Union, EP_DIST_CLUSTER2UNION_THRES, EP_DIST_CLUSTER2CLUSTER_THRES
from scipy.optimize import linear_sum_assignment
from ..mvg import calc_epipolar_dist

class Match:
    def __init__(self, Ks, Ts):
        self.Ks = Ks #(4,3,3)
        self.Ts = Ts #(4,4,4)

    def run(self, kp_clusters_group):
        """
        :param kp_clusters_group: (4, cluster_num, kp_num, 3) 4 imgs per layer, 3:(x,y,score)
        :returns: unions 
        """
        unions = []

        for cam_id, kp_clusters in enumerate(kp_clusters_group):
            cost, mask = self._calc_cost(unions, kp_clusters)
            if cost == None:
                for kp_cluster in kp_clusters:
                    union = Union(self.Ks, self.Ts)
                    union.push({cam_id:kp_cluster})
                    unions.append(union)
            else:
                rows, cols = linear_sum_assignment(cost)
                selected = set(cols)
                for row, col in zip(rows, cols):
                    if cost[row, col] <  EP_DIST_CLUSTER2UNION_THRES and mask[row, col] == 0:
                        unions[row].push(kp_clusters[col])
                    else:
                        union = Union(self.Ks, self.Ts)
                        union.push({cam_id:kp_clusters[col]})
                        unions.append(union)
                
                for i in np.arange(len(kp_clusters)):
                    if i not in selected:
                        union = Union(self.Ks, self.Ts)
                        union.push({cam_id:kp_clusters[i]})
                        unions.append(union)
        
        return unions

    def _calc_cost(self, unions, kp_clusters):
        """
        :param unions: 
        :param kp_clusters: (cluster_num, kp_num, 3)
        :returns: cost, mean epipolar dist
        :returns: mask, if every epipolar dist is less than thres
        """
        cost = np.zeros((len(unions), len(kp_clusters)))
        mask = np.zeros((len(unions), len(kp_clusters)))
        for i, union in enumerate(unions):
            for j, kp_cluster in enumerate(kp_clusters):
                cost[i][j], mask[i][j] = self._calc_dist(union, kp_cluster)

        return cost, mask
    
    def _calc_dist(self, union, kp_cluster):
        dists = []
        mask = 0
        for cluster in union.clusters:
            dist = self._calc_epipolar_dist(cluster, kp_cluster)
            dists.append(dist)
            if dist > EP_DIST_CLUSTER2CLUSTER_THRES:
                mask = 1
        
        return np.mean(dists), mask

    def _calc_epipolar_dist(self, cluster1, cluster2):
        """
        :param cluster1: {cam_id:(kp_num,3)} 3:(x,y,score)
        :param cluster2: {cam_id:(kp_num,3)} 3:(x,y,score)
        :returns: epipolar dist
        """
        cid1 = list(cluster1.keys())[0]
        K1 = self.Ks[cid1]
        T1 = self.Ts[cid1]
        cid2 = list(cluster2.keys())[0]
        K2 = self.Ks[cid2]
        T2 = self.Ts[cid2]
        dists = np.zeros((len(cluster1[cid1]), len(cluster2[cid2])))
        for i, kp1 in enumerate(cluster1):
            for j, kp2 in enumerate(cluster2):
                dists[i][j] = calc_epipolar_dist(kp1, K1, T1, kp2, K2, T2)

        rows, cols = linear_sum_assignment(dists)
        dist = np.mean(dists[rows, cols])
        return dist
