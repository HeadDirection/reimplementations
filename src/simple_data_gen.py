import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy.stats as stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def random_walk_on_s1(tsteps, dt, win_size,ang0 = None):
  ahv = np.random.randn(tsteps+win_size-1)
  ahv = np.convolve(ahv, np.ones(win_size)/win_size,'valid')

  if ang0 == None:
    init_thet = np.random.randn()
  else:
    init_thet = ang0

  thet = init_thet + np.cumsum(ahv)*dt

  return np.vstack([np.sin(thet),np.cos(thet), ahv, thet]).T

def landmark_ts(random_walk_res, landmark_hd, wid, landmark_v):
  thet = random_walk_res[:,-1]
  t = np.arange(thet.shape[0])
  diff = np.abs(landmark_hd + landmark_v * t- thet) % (2 * np.pi)
  diff = np.minimum(diff, 2*np.pi-diff)
  scores = stats.norm.pdf(diff,0, .1 * wid)
  scores[scores < .01] = 0.0
  return scores

def tensor_gen(random_walk_res, landmark_list, landmark_wid, landmark_v):
  mean_ahv = np.mean(np.abs(random_walk_res[:,2:3]))

  for i in range(len(landmark_list)):
    if landmark_list[i] == None:
      landmark_list[i] = np.random.rand() * np.pi * 2
  if len(landmark_list)>0:
    all_landmark_ts = np.vstack([landmark_ts(random_walk_res, landmark_list[i], landmark_wid, landmark_v[i]) for i in range(len(landmark_list))]).T
    input_ts = np.hstack([random_walk_res[:,2:3]/mean_ahv, all_landmark_ts])
  else:
    input_ts = random_walk_res[:,2:3]/mean_ahv
  output_ts = random_walk_res[:,:2]
  h0_dat = random_walk_res[0, :2]

  fin_landmark_list = torch.from_numpy(np.array(landmark_list).reshape(1,1,len(landmark_list))).float().to(device)
  fin_h0_dat = torch.from_numpy(h0_dat.reshape(1,1,2)).to(device)
  fin_input = torch.from_numpy(input_ts.reshape(input_ts.shape[0], 1, input_ts.shape[1])).to(device)
  fin_output = torch.from_numpy(output_ts.reshape(output_ts.shape[0], 1, output_ts.shape[1])).to(device)

  return fin_landmark_list, fin_h0_dat, fin_input, fin_output
  
def gen_batch(n, tsteps, dt, win_size, landmark_list, landmark_wid, landmark_v=0.0):

  rw = random_walk_on_s1(tsteps, dt, win_size)
  _, fin_h0_dat, fin_input, fin_output = tensor_gen(rw, landmark_list, landmark_wid, landmark_v)

  for _ in range(n):

    rw = random_walk_on_s1(tsteps, dt, win_size)
    _, fin_h0_dat2, fin_input2, fin_output2 = tensor_gen(rw, landmark_list, landmark_wid, landmark_v)

    fin_h0_dat = torch.cat((fin_h0_dat, fin_h0_dat2), axis=1).float().to(device)
    fin_input = torch.cat((fin_input, fin_input2), axis=1).float().to(device)
    fin_output = torch.cat((fin_output, fin_output2), axis=1).float().to(device)

  return fin_h0_dat, fin_input, fin_output