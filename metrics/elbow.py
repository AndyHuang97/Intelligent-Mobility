import numpy as np

def getElbows(wss, k_offset, add_first_k=False, elbow_angle=5, d_dk_th=0.001):
  wss /= np.max(wss) # normalize
  # print(wss)
  l = []
  if add_first_k:
    l.append(k_offset)
  for k in range(1, wss.shape[0] - 1):
    d_dk_prev = wss[k] - wss[k-1]
    d_dk_next = wss[k+1] - wss[k]
    if abs(d_dk_next - d_dk_prev) < 10*d_dk_th:
      d_dk_prev = 0
      d_dk_next = 1.0
    elif abs(d_dk_next) < d_dk_th:
      d_dk_next = -0.0
    # print(k + k_offset, d_dk_prev, d_dk_next, d_dk_prev/d_dk_next, sep='\t')
    if (d_dk_prev/d_dk_next) > elbow_angle:
      l.append(k + k_offset)
  return l