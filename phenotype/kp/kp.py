import numpy as np
import json

class KP:
    def __init__(self):
        pass

    def run(self, img):
        """
        :param img: 
        :returns: (cluster_num, kp_num, 3),3:[x,y,score]
        """
        pass
    
    def run_test(self, img_id):
        """
        :param img_id: 
        :returns: (cluster_num, kp_num, 3),3:[x,y,score]
        """
        folder_pre = '/home/ubuntu/Documents/phenotype_device/data/test2/test/6_1_7_12/'
        img_name = folder_pre + str(img_id) + '_20220609141841'

        with open(img_name + '.json', 'r') as f:
            data = json.load(f)

        clusters = {}
        for shape in data['shapes']:
            point = shape['points'][0]
            point.append(1.0)
            
            gid = shape['group_id']
            if gid in clusters.keys():
                clusters[gid].append(point)
            else:
                clusters[gid] = [point]
        
        clusters_list = []
        for cluster in clusters.values():
            clusters_list.append(cluster)
        
        return clusters_list
        
        
    
