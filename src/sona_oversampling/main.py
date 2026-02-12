from scipy.spatial.distance import cdist
import numpy as np

def SONA(X,y, min_label, new_label = 0):
  X_gen_min = (X)[y== min_label]
  X_gen_maj = (X)[y != min_label]

  maj_size = len(X_gen_maj)
  minor_size = len(X_gen_min)

  ## Negative border
  dist_gen_maj2min= cdist(X_gen_maj, X_gen_min)
  rank_dist_maj2min = np.argsort(dist_gen_maj2min)

  neg_border = np.zeros(len(X_gen_min))
  for i in range(maj_size):
    near_point = rank_dist_maj2min[i][0]
    neg_border[near_point] += 1

  ## Positive border
  dist_gen_min2maj = cdist(X_gen_min,X_gen_maj)
  rank_dist_min2maj = np.argsort(dist_gen_min2maj)

  pos_border = np.zeros(len(X_gen_maj))
  for i in range(minor_size):
    near_point = rank_dist_min2maj[i][0]
    pos_border[near_point] += 1

  ## Find radius
  neg_radius = np.zeros(len(X_gen_min))
  rank_dist_gen_pos = np.argsort(dist_gen_min2maj)

  for i in range(minor_size):
    pos_list = rank_dist_gen_pos[i]

    for j in range(maj_size):
      pos_point = pos_list[j]

      if pos_border[pos_point] == 0:
        neg_radius[i] = dist_gen_min2maj[i][pos_point]
        break;

  prop_min = 1 / (neg_border+1)
  prop_min = prop_min / np.sum(prop_min)
  # prop_min = neg_border / np.sum(neg_border)

  syn_list = []

  dist_min2min = cdist(X_gen_min,X_gen_min)

  synthese_len = maj_size - minor_size

  for t in range(synthese_len):      # assume to 1:1
    i = np.random.choice(len(X_gen_min), p = prop_min)

    terminal_prop = 1/dist_min2min[i]
    terminal_prop = np.nan_to_num(terminal_prop, nan=0, posinf=0, neginf=0) # Use np.nan_to_num
    terminal_prop = terminal_prop / np.sum(terminal_prop) # equal smote
    j = np.random.choice(len(X_gen_min), p = terminal_prop)

    direction_vector =  X_gen_min[j] - X_gen_min[i]
    norm_v = np.linalg.norm(direction_vector)
    direction_vector = direction_vector / norm_v

    alpha = np.random.random()

    syn_x = X_gen_min[i] + alpha* direction_vector *min(neg_radius[i], norm_v)
    syn_list.append(syn_x)

  return (
            np.vstack([X, syn_list]),
            np.hstack([y, np.repeat(min_label + new_label, len(syn_list))]),
        )

def hello():
  print("Hello")