#!/usr/bin/env python

import struct
import numpy as np
import cv2
from sklearn.decomposition import PCA

def read_siftgeo(filename, max_desc = None):
  """
  Read a set of SIFT descriptors using the siftgeo format.
  
  - filename: Filename including the descriptors in binary format.

  RETURNS:
  - kps: A list with the detected keypoints (cv2.KeyPoint)
  - des: A numpy array of shape (number_of_desc, 128) of type float32
  """  

  # Computing the number of descriptors
  with open(filename, "rb") as f:
    f.seek(0, 2)
    possible_desc = f.tell() // (9 * 4 + 1 * 4 + 128)
  
  # Checking the number of descriptors to get
  if max_desc is None:
    num_desc = possible_desc
  else:
    num_desc = min(possible_desc, max_desc)

  kps = []
  des = np.zeros((num_desc, 128), dtype=np.float32)

  # Reading descriptors
  with open(filename, "rb") as f:
    for i in range(num_desc):
      ### Reading header
      
      # X / Y 
      bts = f.read(4);
      x = struct.unpack('<f', bts)[0]    
      bts = f.read(4);
      y = struct.unpack('<f', bts)[0]
    
      # Scale
      f.read(4)

      # Angle
      bts = f.read(4)
      ang = struct.unpack('<f', bts)[0]

      # Skip affine matrix components
      for _ in range(4):
        f.read(4)
      
      # Cornerness
      bts = f.read(4)
      corn = struct.unpack('<f', bts)[0]

      kp = cv2.KeyPoint(x=x, y=y, _size=0, _angle=ang, _response=corn)    
      kps.append(kp)

      ### Reading descriptor
      # Dim
      bts = f.read(4)
      dim = struct.unpack('<i', bts)
      d = dim[0]
      
      for j in range(d):
        bts = f.read(1)
        val = struct.unpack('<B', bts)
        des[i, j] = np.float32(val[0])

  return kps,des

def read_fvecs(filename, c_contiguous=True):
  """
  Read a set of SIFT descriptors using the fvecs format.
  
  - filename: Filename including the descriptors in binary format.

  RETURNS:
  - fv: A numpy array of shape (number_of_desc, 128) of type float32
  """
  fv = np.fromfile(filename, dtype=np.float32)
  if fv.size == 0:
      return np.zeros((0, 0))
  dim = fv.view(np.int32)[0]
  assert dim > 0
  fv = fv.reshape(-1, 1 + dim)
  if not all(fv.view(np.int32)[:, 0] == dim):
      raise IOError("Non-uniform vector sizes in " + filename)
  fv = fv[:, 1:]
  if c_contiguous:
      fv = fv.copy()
  return fv

def load_visual_vocab(filename, ntrees = 4):
  """
  Load a visual vocabulary and train a set of kd-tree for a fast access
  
  - filename: Filename of the vocabulary, in fvecs format.
  - ntrees: Number of trees to train.
  
  RETURNS:
  - An OpenCV2 FLANN index based on a set of kd-trees.
  """
  vocab = read_fvecs(filename)
  
  index_params = dict(algorithm = 1, trees = ntrees)
  search_params = dict(checks = 32)
  index = cv2.FlannBasedMatcher(index_params, search_params)  
  index.add([vocab])
  index.train()
  
  return index

def load_SIFT_descriptors(img_names, max_desc = None):
  """
  Load SIFT descriptors for the set of indicated images.

  RETURNS: 
  - kps: A list of lists of keypoints (cv2.KeyPoint) extracted from the indicated images.
  - desc: A list of lists of keypoints extracted from the query images.
  """
  kps = []
  desc = []
  for img_name in img_names:
    imno = img_name[:-len(".jpg")]
    kp,d = read_siftgeo('siftgeo/' + imno + '.siftgeo', max_desc) # It assumes this directory structure

    # Appending results
    kps.append(kp)
    desc.append(d)
  
  return kps,desc

def apply_PCA(list_of_descriptors, ncomps = 1024):

  pca = PCA(n_components=ncomps)

  # Compute the transformation
  desc_set = [d for l in list_of_descriptors for d in l]
  desc_stack = np.vstack(desc_set)  
  pca.fit(desc_stack)

  result = []
  for l in list_of_descriptors:
    # Transforming each set of descriptors
    desc_stack = np.vstack(l)    
    desc_stack_red = pca.transform(desc_stack)
    desc_stack_red = [np.array(d) for d in desc_stack_red.tolist()]
    result.append(desc_stack_red)
  
  return result
