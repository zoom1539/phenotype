import numpy as np
import cv2
from .kp import KP
from .match import Match


class Phenotype:
    def __init__(self, calib_filepath):
        self.Ks = None # cam matrix (12, 3, 3)
        self.Ts = None # cam pose (12, 4, 4) world to cam
        self._read_calib(calib_filepath)
        self.kp = KP()

    def _read_calib(self, calib_filepath):
        pass
    
    def extract_3d_kp(self, img_groups):
        """
        :param img_groups: (3, 4, HWC) 3 layers, 4 imgs per layer
        :returns: 3d_kp (cluster_num, kp_num, 4) 4:[x,y,z,score]
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
        :returns: (4, cluster_num, kp_num, 3)
                  [[kp_clusters1], [kp_clusters2], [kp_clusters3], [kp_clusters4]],
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
    

class PhenotypeTest:
    def __init__(self):
        
        self._read_calib()
        self.kp = KP()

    def _read_calib(self):
        self.Ks = [] # cam matrix (4, 3, 3)
        self.Ts = [] # cam pose (4, 4, 4) world to cam
        
        folder_pre = '/home/ubuntu/Documents/phenotype_device/data/test2/calib/OneCamera/'
        for i in [1,6,7,12]:
            fs = cv2.FileStorage(folder_pre + str(i) + '/' + 'result.yml', cv2.FILE_STORAGE_READ)
            K = fs.getNode('matrix').mat()
            self.Ks.append(K)
        
        #
        folder_pre = '/home/ubuntu/Documents/phenotype_device/data/test2/calib/MultiCamera/'
        
        fs = cv2.FileStorage(folder_pre + '1_6/' + 'result.yml', cv2.FILE_STORAGE_READ)
        T_1_6 = fs.getNode('T_1_6').mat()

        fs = cv2.FileStorage(folder_pre + '6_7/' + 'result.yml', cv2.FILE_STORAGE_READ)
        T_6_7 = fs.getNode('T_6_7').mat()

        fs = cv2.FileStorage(folder_pre + '7_12/' + 'result.yml', cv2.FILE_STORAGE_READ)
        T_7_12 = fs.getNode('T_7_12').mat()

        fs = cv2.FileStorage(folder_pre + '12_1/' + 'result.yml', cv2.FILE_STORAGE_READ)
        T_12_1 = fs.getNode('T_12_1').mat()
        
        fs = cv2.FileStorage(folder_pre + 'g_6/' + 'result.yml', cv2.FILE_STORAGE_READ)
        T_w_6 = fs.getNode('T_w_6').mat()
        
        T_w_1 = T_w_6.dot(np.linalg.inv(T_1_6))
        self.Ts.append(T_w_1)
        self.Ts.append(T_w_6)
        
        T_w_7 = T_w_6.dot(T_6_7)
        self.Ts.append(T_w_7)
        
        T_w_12 = T_w_7.dot(T_7_12)
        self.Ts.append(T_w_12)
    
    def extract_3d_kp(self):
        """
        :param img_groups: (3, 4, HWC) 3 layers, 4 imgs per layer
        :returns: 3d_kp (cluster_num, kp_num, 4) 4:[x,y,z,score]
        """

        kp_clusters_group = self._extract_kp()
        kp_3d_clusters = self._extract_kp_3d(kp_clusters_group)
    
        return kp_3d_clusters

    
    def _extract_kp(self):
        """
        :param img_group: (4, HWC) 4 imgs per layer
        :returns: (4, cluster_num, kp_num, 3)
                  [[kp_clusters1], [kp_clusters2], [kp_clusters3], [kp_clusters4]],
                  kp_clusters:(cluster_num, kp_num, 3)
        """
        
        kp_clusters_group = []

        for i in [1,6,7,12]:
            kp_clusters = self.kp.run_test(i)
            kp_clusters_group.append(kp_clusters)

        return kp_clusters_group

    def _extract_kp_3d(self, kp_clusters_group):
        """
        :param kp_clusters_group: (4, cluster_num, kp_num, 3) 4 imgs per layer, 3:(x,y,score)
        :returns: (cluster_num, kp_num,4) 4:(x,y,z,score)
        """

        match = Match(self.Ks, self.Ts)
        unions = match.run(kp_clusters_group)

        kp_3d_clusters = []
        for union in unions:
            sts, kp_3d_cluster = union.get_3d()
            if sts:
                kp_3d_clusters.append(kp_3d_cluster)
        
        return kp_3d_clusters
        



