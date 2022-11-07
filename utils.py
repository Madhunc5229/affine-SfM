import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as plt_g
from scipy.linalg import sqrtm

def positiveDef(M):
  """
  method to compute nearest positive definite matrix
  ref: http://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

  parameters: M motion matrix
  """
  M = (M + M.T) * 0.5
  k = 0
  I = np.eye(M.shape[0])
  while True:
      try:
          _ = np.linalg.cholesky(M)
          break
      except np.linalg.LinAlgError:
          k += 1
          w, v = np.linalg.eig(M)
          min_eig = v.min()
          M += (-min_eig * k * k + np.spacing(min_eig)) * I
  return M

def normalizeMean(point_set):
  """
  function to normalize the point set(P x F) to zero mean

  parameters: point_set array of points to be normalized
  """
  for i in range(point_set.shape[1]):
    point_set[:,i] = point_set[:,i] - np.mean(point_set[:,i])
  
  return point_set



def plotShape(S):
  """
  method to plot the structure

  paramters: S structure matrix 
  """
  x = S[0,:]
  y = S[1,:]
  z = S[2,:]
  fig = plt_g.Figure(data=[plt_g.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color='red'))])
  fig.show()

  fig = plt.figure(figsize=(18.5, 6))
  fig.suptitle('Reconstructed image')
  ax1 = fig.add_subplot(131, projection='3d')
  ax1.scatter(x, y, z, color='red', lw=1)
  ax1.view_init(-70, 0)
  ax2 = fig.add_subplot(132, projection='3d')
  ax2.scatter(x, y, z, color='red', lw=1)
  ax2.view_init(45, 180)
  ax3 = fig.add_subplot(133, projection='3d')
  ax3.scatter(x, y, z, color='red', lw=1)
  ax3.view_init(-90, 90)
  plt.savefig('results/shape.png')
  plt.show()

def plotCameraMotion(M):
  """
  function to plot the motion of points

  parameters: M motion matrix
  """
  camera_positions = np.zeros((M.shape[0] // 2, M.shape[1]))

  # Normalizing the position vector
  M_mean = np.mean(M, axis=0)
  for i in range(M.shape[1]):
    M[:, i] = M[:, i] / M_mean[i]

  # Finding the cross-product
  for i in range(camera_positions.shape[0]):
    a_k = np.cross(M[(i *2), :], M[(i * 2) + 1, :])
    camera_positions[i, :] = np.divide(a_k, np.linalg.norm(a_k)) 
  
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
  fig.suptitle('Camera motion')
  ax1.plot(camera_positions[:, 0])
  ax2.plot(camera_positions[:, 1])
  ax3.plot(camera_positions[:, 2])
  plt.savefig('results/motion.png')
  plt.show()
  


def affineSFM(track_x, track_y):
  """
  Function: Affine structure from motion algorithm
  Normalize x, y to zero mean
  Create measurement matrix
  D = [xn' ; yn'];
  Decompose and enforce rank 3
  Apply orthographic constraints
  """
  # Remove the nan value
  nan_idx = []
  for i in range(track_x.shape[1]):
    for j in range(track_x.shape[0]):
      if np.isnan(track_x[j,i]) or np.isnan(track_y[j,i]):
        if not j in nan_idx:
          nan_idx.append(j)

  track_x = np.delete(track_x, nan_idx, 0)
  track_y = np.delete(track_y, nan_idx, 0)
  track_x = normalizeMean(track_x)
  track_y = normalizeMean(track_y)

  m = track_x.shape[1]
  n = track_x.shape[0]

  # Construct the D matrix of shape 2m X n
  D = np.empty((2*m,n))

  #loop for n points and append x and y
  for i in range(2*m):
    j = i//2
    if i%2==0:
      D[i] = track_x[:,j]
    else:
      D[i] = track_y[:,j]


  Ud, Sd, Vd = np.linalg.svd(D)

  U3 = Ud[:,0:3]
  W3 = np.diag(Sd[:3])
  V3 = Vd.T[:, :3]


  M = np.dot(U3, sqrtm(W3))
  S = np.dot(sqrtm(W3), V3.T)

  K = np.array([1,1,0])
  l = []
  for i in range(m):
    
    a1 = np.array(M[2*i]).reshape((3,1))
    a2 = np.array(M[2*i+1]).reshape((1,3))
    temp = np.dot(a1,a2)
    temp = positiveDef(temp)
    temp = temp.reshape(1,9)
    l.append(temp)

  l = np.array(l).reshape((-1,9))

  ul, sl, dl = np.linalg.svd(l)
  L = dl.T[:,-1]
  L = L.reshape(3,3)
  L = positiveDef(L)

  C = np.linalg.cholesky(L)

  M = np.dot(M,C)
  S = np.dot(np.linalg.inv(C),S)

  return M, S