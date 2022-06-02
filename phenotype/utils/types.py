
class Union:
    def __init__(self, Ks, Ts):
        self.Ks = Ks
        self.Ts = Ts
        self.clusters = []
    
    def push(self, cluster):
        """
        :param cluster: {cam_id:(kp_num,3)} 3:(x,y,score)
        :returns: 
        """
        self.clusters.append(cluster)
    
    def get_3d(self):
        """
        :returns: (kp_num,4) 4:(x,y,z,score)
        """
        pass

    