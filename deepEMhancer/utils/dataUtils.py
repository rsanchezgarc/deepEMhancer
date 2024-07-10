from scipy.ndimage import grey_dilation, grey_erosion, gaussian_filter
from scipy.stats import pearsonr
import numpy as np
from skimage.transform import resize


def morphDilation(x, size=None):
  if size is None:
    size= tuple([3]*len(x.shape))
  x= grey_dilation(x, size=size)
  return x

def morphErosion(x, size=None):
  if size is None:
    size= tuple([3]*len(x.shape))
  x= grey_erosion(x, size=size)
  return x

def applyRelativeThr(x, r_thr=0.05, robust=False):
  if robust:
    min_val = np.percentile(x.flatten(), 5)
    max_val = np.percentile(x.flatten(), 95)
  else:
    min_val = x.min()
    max_val = x.max()
  x_range= max_val-min_val
  thr= min_val+r_thr*x_range
  out= x.copy()
  condition= out< thr
  out[condition] =0
  out[~ condition] =1
  return out

def gaussianFilter(x, sigma=1):
  x= gaussian_filter(x, sigma)
  return x

def _generateRadialDistances(data):
  xlimit, ylimit, zlimit = data.shape[:3]
  x, y, z = np.meshgrid(np.arange(xlimit), np.arange(ylimit),  np.arange(zlimit), indexing='ij')
  centralCoords = [elem // 2 for elem in data.shape]
  R = np.sqrt((x - centralCoords[0]) ** 2 + (y - centralCoords[1]) ** 2 + (z - centralCoords[2]) ** 2)
  return R

def radial_profile(data):
  r = _generateRadialDistances(data).astype(np.int32)
  tbin = np.bincount(r.ravel(), data.ravel())
  nr = np.bincount(r.ravel())
  radialprofile = tbin / nr
  return radialprofile

def getCoordsWithinSphere(vol, maxDistFrac=None, maxDist=None):
  if maxDist is None:
    maxDist= np.min(vol.shape)//2
    if maxDistFrac is not None:
      assert 0< maxDistFrac <1
      maxDist= int(maxDist*maxDistFrac)
  R=  _generateRadialDistances(vol)
  coords= np.where(R< maxDist)
  return coords

def getCoordsOutsideSphere(vol, maxDistFrac=None, maxDist=None, deltaMaxDist=0):
  if maxDist is None:
    maxDist= np.min(vol.shape)//2 +deltaMaxDist
    if maxDistFrac is not None:
      assert 0< maxDistFrac <1
      maxDist= int(maxDist*maxDistFrac) +deltaMaxDist
  R=  _generateRadialDistances(vol)
  coords= np.where(R> maxDist)
  return coords

def computeCorrCoef(queryData, referenceData, ignoreOutOfRadius=True, percentil_val_inQuery=25, ignoreZeros=False):
  if ignoreOutOfRadius:
    R= _generateRadialDistances(queryData)
    selectionCriterion = ((R < np.min(queryData.shape) // 2) &
                          (queryData > np.percentile(queryData, percentil_val_inQuery)))
    queryData = queryData[selectionCriterion]
    referenceData = referenceData[selectionCriterion]

  if ignoreZeros:
    selection= (referenceData > 0) # (queryData > 0) & (referenceData > 0)
    queryData= queryData[selection]
    referenceData= referenceData[selection]

  if len(queryData)<2:
    corr=-1
  else:
    corr= pearsonr(np.reshape(queryData, (-1,)), np.reshape(referenceData, (-1,)))[0]
  return corr

def getRadialStd_andCC(vol, otherVol, step=4., plot=False):

  R = _generateRadialDistances(vol)

  def f(r):
    selectionCriterion= (R >= r - step / 2.) & (R < r + step / 2.)
    vol1_slice= vol[selectionCriterion]
    corr= pearsonr( vol1_slice, otherVol[selectionCriterion] )[0]
    if np.isnan(corr):
      corr=-1
    return vol1_slice.std(), corr
  r = np.arange(1, int(np.max(R)),step=step)
  profiles = np.vectorize(f)(r)
  profileStd, profileCorr= profiles

  if plot:
    from matplotlib import pyplot as plt
    fig = plt.figure()
    fig.add_subplot(311)
    plt.title("std")
    plt.plot(r, profileStd)
    fig.add_subplot(312)
    plt.title("corr")
    plt.plot(r, profileCorr)
    fig.add_subplot(313)
    plt.title("prod")
    plt.plot(r, profileStd*profileCorr)
    plt.show(block=True)

  return (profileStd, profileCorr), r

def getByPlaneStd_andCC(vol, otherVol, step=4., plot=False):

  centralCoords= [elem//2 for elem in vol.shape]

  # calculate the mean
  def f(i, axis):
    vol1_slice= np.take(vol, indices=[centralCoords[axis]+i], axis=axis)
    corr= pearsonr( vol1_slice.flatten(), np.take(otherVol, indices=[centralCoords[axis]+i], axis=axis).flatten() )[0]
    if np.isnan(corr):
      corr=-1
    return vol1_slice.std(), corr, vol1_slice.mean()
  for axis in range(3):
    i_array = np.arange(0, centralCoords[axis],step=step)
    profiles = np.vectorize(f)(i_array, axis)
    profileStd, profileCorr, meanProfile= profiles
    if plot:
      from matplotlib import pyplot as plt
      fig = plt.figure()
      fig.add_subplot(311)
      plt.title("by plane std")
      plt.plot(i_array, profileStd)
      fig.add_subplot(312)
      plt.title("corr")
      plt.plot(i_array, profileCorr)
      fig.add_subplot(313)
      plt.plot(i_array, meanProfile)
      plt.show(block=True)

  return (profileStd, profileCorr), i_array


def getNewShapeForResize(vol, boxSize, newBoxSize):
  downFactor= boxSize/newBoxSize
  output_shape= [ round(s*downFactor) for s in vol.shape]
  return output_shape


def resizeVol(vol, output_shape):
  vol= resize(vol, output_shape, order=3, mode='reflect', cval=0, clip=True, preserve_range=False,
                           anti_aliasing=True, anti_aliasing_sigma=None)
  return vol
