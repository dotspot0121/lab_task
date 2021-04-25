import numpy as np
from sklearn import manifold
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

#MNISTデータから取り出した.csv形式のテスト用画像データからラベルを取り除いて，ndarray型でdatumに格納
datum = np.loadtxt("mnist_test2_1.csv",delimiter=",",usecols=range(1,785))

#datumに格納されている画像データから，距離行列dist_Mをndarray型で格納
y=distance.pdist(datum)
dist_M = distance.squareform(y)

#mdsオブジェクトを生成
mds = manifold.MDS(n_components=3,  random_state=6,dissimilarity="precomputed")
#diet_Mを基に，投影座標を格納する2次元配列posを生成
pos = mds.fit_transform(dist_M)

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax = Axes3D(fig)
colorlist=colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
ax.scatter(pos[0:84, 0], pos[0:84, 1],pos[0:84, 2],c=colorlist[0], marker = '.',label='0')
ax.scatter(pos[85:210, 0], pos[85:210, 1], pos[85:210, 2],c=colorlist[1], marker = '.',label='1')
ax.scatter(pos[210:326, 0], pos[210:326, 1],pos[210:326, 2],c=colorlist[2], marker = '.',label='2')
ax.scatter(pos[327:433, 0], pos[327:433, 1],pos[327:433, 2],c=colorlist[3], marker = '.',label='3')
ax.scatter(pos[434:544, 0], pos[434:544, 1],pos[434:544, 2],c=colorlist[4], marker = '.',label='4')
ax.scatter(pos[545:630, 0], pos[545:630, 1],pos[545:630, 1],c=colorlist[5], marker = '.',label='5')
ax.scatter(pos[631:717, 0], pos[631:717, 1],pos[631:717, 2],c=colorlist[6], marker = '.',label='6')
ax.scatter(pos[718:999, 0], pos[718:999, 1],pos[718:999, 2],c=colorlist[7], marker = '.',label='7')

#----------ラベル-----------------------------------------------------------
#labels = np.genfromtxt("mnist_test.csv",delimiter=",",usecols=0,dtype=str)
# for label, x, y in zip(labels, pos[:, 0], pos[:, 1]):
#     plt.annotate(
#         label,
#         xy = (x, y), xytext = (70, -20),
#         textcoords = 'offset points', ha = 'right', va = 'bottom',
#         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
#     )

ax.legend(loc='upper left')
plt.show()