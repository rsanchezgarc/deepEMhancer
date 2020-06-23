import shutil
import json
import os
import mrcfile
import numpy as np


def loadTextFile(fname, commentChar="#"):
  ids=[]
  with open(fname) as f:
    for l in f :
      if l.startswith(commentChar): continue
      ids.append( l.strip())
  return set(ids)


def getSamplingRate(fname):
  try:
    with mrcfile.open(fname, permissive=True) as f:
      return f.voxel_size.x
  except ValueError:
    print("Error with %s"%fname)

def loadVol(fname, normalize=None, returnBoxSize=False):
  try:
    with mrcfile.open(fname, permissive=True) as f:
      emMap = f.data.astype(np.float32).copy()
      boxSize= f.voxel_size
      # print("loading", fnameIn, emMap.mean(), emMap.std(), emMap.min(), emMap.max())
      if normalize is not None and normalize is not False:
        if callable(normalize):
          emMap= normalize(emMap)
        elif normalize is True or normalize.startswith("minMax"):
          emMap = (emMap - emMap.min()) / (emMap.max() - emMap.min())
        else:
          raise ValueError("Normalization type not recognized")
  except ValueError:
    print("Error with %s"%fname)
    raise
  assert emMap.shape[0]>0, "Error, %s is empty"%fname
  if returnBoxSize:
    boxSize= boxSize.x
    return emMap, boxSize
  else:
    return emMap


def loadVolIfFnameOrIgnoreIfMatrix(vol_fname_or_matrix, normalize):
  if isinstance(vol_fname_or_matrix, str):
    vol_fname_or_matrix = os.path.expanduser(vol_fname_or_matrix)
    vol, boxSize = loadVol(vol_fname_or_matrix, normalize=normalize, returnBoxSize=True)
  else:
    vol, boxSize = vol_fname_or_matrix, None
    if normalize is not None:
      vol= normalize(vol)

  return vol, boxSize

def saveVol(fname, data, fname_headerLike=None, samplingRate=None):
  '''

  :param fname: name where mrc file will be saved
  :param data: numpy array HxLxD that conatins the data
  :param fname_headerLike: a mrc filename that will be read to fill the header of the new file. If None, samplingRate must be provided
  :param samplingRate: the sampling rate of the data. If fname_headerLike is provided, samplingRate is ignored
  :return: None
  '''

  if fname_headerLike is None:
    assert samplingRate is not None, "Error, when saving volume, either sampling rate or fnameIn to same size volume required"
    with mrcfile.new(fname) as f:
      f.set_data(data.astype(np.float32))
      f.voxel_size = tuple([samplingRate]*3)
  else:
    assert fname_headerLike != fname, "Error, saveVol won't overwrite an existing fnameIn %s" % fname
    shutil.copyfile(fname_headerLike, fname)
    with mrcfile.open(fname,  mode='r+', permissive=True) as f:
      f.data[:]= data.astype(np.float32)


def readIdsFromJson(trainValJsonPath, label=None):
    with open(trainValJsonPath) as f:
      data= json.load(f)
      if label is not None:
        return list(set( data[label] ))
      else:
        ids=set([])
        for key in data:
          ids=ids.union( data[key] )
        return ids