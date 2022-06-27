from phenotype import  PhenotypeTest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

phenotype = PhenotypeTest()

kp_3d_clusters = phenotype.extract_3d_kp()


# show
fig = plt.figure()
ax3d = Axes3D(fig)

kp_pre = [0, 0, 0]
for i, kp_3d_cluster in enumerate(kp_3d_clusters):
    for j, kp_3d in enumerate(kp_3d_cluster):
        # set z-axis up
        kp = np.array(kp_3d[:3]) * np.array([1,-1,-1])
        x,y,z = kp
        ax3d.scatter(x,y,z,c="g",marker="o")

        if j != 0:
            x = np.linspace(kp[0], kp_pre[0])
            y = np.linspace(kp[1], kp_pre[1])
            z = np.linspace(kp[2], kp_pre[2])
            ax3d.plot(x,y,z,c="r")
        
        kp_pre = kp

ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
plt.show()