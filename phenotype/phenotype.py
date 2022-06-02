import numpy as np
from kp import KP
from match import Match


class Phenotype:
    def __init__(self, calib_filepath):
        self.Ks = None # cam matrix (12, 3, 3)
        self.Ts = None # cam pose (12, 4, 4) world to cam
        self.read_calib(calib_filepath)
        self.kp = KP()

    def read_calib(self, calib_filepath):
        pass
    
    def extract_3d_kp(self, img_groups):
        """
        :param img_groups: (3, 4, HWC) 3 layers, 4 imgs per layer
        :returns: 3d_kp (group_num, kp_num, 4) 4:[x,y,z,score]
        """

        kp_3d_clusters_groups = []
        group_ids = [0, 1, 2]
        for img_group, group_id in zip(img_groups, group_ids):
            kp_clusters_group = self._extract_kp(img_group)
            kp_3d_clusters_group = self._extract_kp_3d(kp_clusters_group, group_id)
            kp_3d_clusters_groups += kp_3d_clusters_group
        
        kp_3d = self._nms(kp_3d_clusters_groups)
        return kp_3d

    
    def _extract_kp(self, img_group):
        """
        :param img_group: (4, HWC) 4 imgs per layer
        :param group_id: 
        :returns: [[kp_clusters1], [kp_clusters2], [kp_clusters3], [kp_clusters4]],
                  kp_clusters:(cluster_num, kp_num, 3)
        """
        
        kp_clusters_group = []

        for img in img_group:
            kp_clusters = self.kp.run(img)
            kp_clusters_group.append(kp_clusters)

        return kp_clusters_group

    def _extract_kp_3d(self, kp_clusters_group, group_id):
        """
        :param kp_clusters_group: (4, cluster_num, kp_num, 3) 4 imgs per layer, 3:(x,y,score)
        :param group_id: 
        :returns: (cluster_num, kp_num,4) 4:(x,y,z,score)
        """
        Ks =  self.Ks[group_id * 4 : group_id * 4 + 4]
        Ts =  self.Ts[group_id * 4 : group_id * 4 + 4]

        match = Match(Ks, Ts)
        unions = match.run(kp_clusters_group)

        kp_3d_clusters = []
        for union in unions:
            kp_3d_cluster = union.get_3d()
            kp_3d_clusters.append(kp_3d_cluster)
        
        return kp_3d_clusters
        
    def _nms(self, kp_3d_clusters_groups):
        """
        :param kp_3d_clusters_groups: (cluster_num, kp_num, 4) 4:(x,y,z,score)
        :returns: (cluster_num, kp_num, 4) 4:(x,y,z,score)
        """
        return kp_3d_clusters_groups








if __name__ == '__main__':
    # A = np.zeros((2,3))
    # A[0,1] = np.zeros((2,2))
    # print(A.shape)

    list0 = [0,1]
    list1 = [2,3]
    list0 += list1
    
    # for i in range(0,6):
    #     for j in range(0,i+1):
    #         list.append(j)
    #     list1.append(list)
    #     list = []
    print(list0)
