from scipy.optimize import linear_sum_assignment
import numpy as np
from ..mvg import calc_epipolar_dist, triangulate2

class KPUnion:
    def __init__(self, Ks, Ts):
        self.Ks = Ks
        self.Ts = Ts
        self.kps = []
        
    def push(self, kp):
        """
        :param kp: {cam_id:(3,)} cam_id:0-3, 3:(x,y,score)
        :returns: 
        """
        self.kps.append(kp)
        
    def get_3d(self):
        """
        :returns: (x,y,z,score)
        """
        Ks = []
        Ts = []
        coords = []
        for kp in self.kps:
            cid = list(kp.keys())[0]
            K = self.Ks[cid]
            Ks.append(K)
            T = self.Ts[cid]
            Ts.append(T)
            coord = kp[cid]
            coords.append(coord)
        
        return triangulate2(Ks, Ts, coords)
            
class Union:
    def __init__(self, Ks, Ts):
        self.Ks = Ks
        self.Ts = Ts
        self.clusters = []
    
    def push(self, cluster):
        """
        :param cluster: {cam_id:(kp_num,3)} cam_id:0-3, 3:(x,y,score)
        :returns: 
        """
        self.clusters.append(cluster)
    
    def get_3d(self):
        """
        :returns: (kp_num,4) 4:(x,y,z,score)
        """
        if len(self.clusters) < 2:
            return None
        
        def key(elem):
            k = list(elem.keys())[0]
            return len(elem[k])
        self.clusters.sort(key = key, reverse = True)
        
        kp_unions = []
        
        for i, cluster in enumerate(self.clusters):
            cid = list(cluster.keys())[0]
            kps = cluster[cid]
            if i == 0: # longset kp_cluster
                for kp in kps:
                    kp_union = KPUnion(self.Ks, self.Ts)
                    kp_union.push({cid:kp})
                    kp_unions.append(kp_union)
            else:
                cost = self._calc_cost(kp_unions, cluster)
                rows, cols = linear_sum_assignment(cost)
                for row, col in zip(rows, cols):
                    kp_unions[row].push({cid:kps[col]})
        
        kp_3ds = []
        for kp_union in kp_unions:
            kp_3d = kp_union.get_3d()
            kp_3ds.append(kp_3d)
        return kp_3ds

    def _calc_cost(self, kp_unions, cluster):
        """
        :param kp_unions: 
        :param cluster: {cid:(kp_num, 3)}
        :returns: cost, epipolar dist
        """
        cid = list(cluster.keys())[0]
        kps = cluster[cid]
        cost = np.zeros((len(kp_unions), len(kps)))
        for i, kp_union in enumerate(kp_unions):
            
            first_cid = list(kp_union.kps[0].keys())[0]
            first_kp = kp_union.kps[0][first_cid]
            
            for j, kp in enumerate(kps):
                cost[i][j] = calc_epipolar_dist(first_kp, 
                                                self.Ks[first_cid], 
                                                self.Ts[first_cid],
                                                kp, 
                                                self.Ks[cid], 
                                                self.Ts[cid])
        return cost
                
        
        
        
        
        
        
        
        
        
        
        
    

    