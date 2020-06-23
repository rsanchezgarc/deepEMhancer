import numpy as np
from scipy.stats import iqr

from ..config import MAX_VAL_AFTER_NORMALIZATION
from ..utils.dataUtils import morphDilation, gaussianFilter, radial_profile, applyRelativeThr, getCoordsWithinSphere
from ..utils.genericException import GenericError

DEFAULT_PERCENTIL = 95
DEFAULT_BINARY_MASK_THR= 0.01

def targetNormalizationRegr(x, mask):
  mask = morphDilation(mask, 1)
  binary_mask = np.where(mask > DEFAULT_BINARY_MASK_THR, 1, 0)
  background = np.median(x[binary_mask < 1])
  background_median = np.median(background)
  background_upper_percentil = np.percentile(background, DEFAULT_PERCENTIL)

  x = np.clip(x, background_median, None)
  x = x * binary_mask

  target_inside_mask = x[binary_mask > 0]
  target_inside_mask = target_inside_mask[target_inside_mask > background_median]

  target_upper_percentil = np.percentile(target_inside_mask, DEFAULT_PERCENTIL)
  target_iqr = target_upper_percentil - background_upper_percentil
  if target_iqr <= 0:
    raise ValueError("Bad iqr %.3f. Is your input masked?. Unmasked inputs required" % target_iqr)
  x = x / target_iqr
  x = np.clip(x, None, MAX_VAL_AFTER_NORMALIZATION) #Just to prevent outliers
  return x

def targetNormalizationLocscale(x, mask):

  binary_mask = np.where(morphDilation(mask, 1) > DEFAULT_BINARY_MASK_THR, 1, 0)
  background = np.median(x[binary_mask < 1])
  background_median = np.median(background)
  background_upper_percentil = np.percentile(background, DEFAULT_PERCENTIL)

  x = np.clip(x, background_median, None)
  x = x * mask

  target_inside_mask = x[binary_mask > 0]
  target_inside_mask = target_inside_mask[target_inside_mask > background_median]

  target_upper_percentil = np.percentile(target_inside_mask, DEFAULT_PERCENTIL)
  target_iqr = target_upper_percentil - background_upper_percentil
  if target_iqr <= 0:
    raise ValueError("Bad iqr %.3f. Is your input masked?. Unmasked inputs required" % target_iqr)
  x = x / target_iqr
  x = np.clip(x, None, MAX_VAL_AFTER_NORMALIZATION) #Just to prevent outliers
  return x


def targetNormalization_2(x, y, mask):

  inside_x = x[morphDilation(mask, 1)>= DEFAULT_BINARY_MASK_THR]
  mean_x, std_x = np.mean(inside_x), np.std(inside_x)

  inside_y= y[mask>= DEFAULT_BINARY_MASK_THR]
  mean_y, std_y = np.mean(inside_y), np.std(inside_y)

  y= ((y-mean_y)/std_y)*std_x + mean_x

  return y


def targetNormalizationClassif(x):
  x= np.clip(x, np.percentile(x,0.1), np.percentile(x,99.9))
  x_norm= minMaxNormalization(x)
  return x_norm

def inputNormalizationWithMask(x, mask):
  mask = morphDilation(mask, 3)
  mask= applyRelativeThr(mask, DEFAULT_BINARY_MASK_THR)
  median_val = np.median( x[mask>0] )
  iqr_val = iqr(x[mask > 0], rng=(10,DEFAULT_PERCENTIL))
  x_norm= (x-median_val)/iqr_val
  x_norm*= mask
  return x_norm

def inputNormalizationWithMask_2(x, mask):
  mask = morphDilation(mask, 3)
  mask= applyRelativeThr(mask, DEFAULT_BINARY_MASK_THR)
  selection=  (mask>0) & (x>0)

  median_val = np.median( x[selection ] )
  iqr_val = iqr(x[selection], rng=(10,DEFAULT_PERCENTIL))
  # iqr_val= x[selection].max()- x[selection].min()
  x_norm= (x-median_val)/iqr_val
  x_norm*= mask
  return x_norm


def inputNormalizationWithMask_3(x, mask): #This might is too tight for general purposes
  mask = np.where(morphDilation(mask, 1) > DEFAULT_BINARY_MASK_THR, 1, 0)
  selection=  (mask>0) & (x>0)
  median_val = np.median( x[selection ] )
  iqr_val = iqr(x[selection], rng=(10,DEFAULT_PERCENTIL))
  # iqr_val= x[selection].max()- x[selection].min()
  x_norm= (x-median_val)/iqr_val
  x_norm*= mask
  return x_norm

def inputNormalization_classification(x):
  x_min= -x.min()
  x_range= x.max()-x_min
  midPoint= x_min+ x_range*.5
  conditionSplit= x<midPoint
  x_min= np.percentile(x[conditionSplit], 5)
  x_max = np.percentile(x[~ conditionSplit], DEFAULT_PERCENTIL)
  if not np.isclose(x_min, x_max):
    x= x/(x_max-x_min)
  x = np.clip(x, None, MAX_VAL_AFTER_NORMALIZATION)  # Just to prevent outliers
  return x

def inputNormalization(x):
  x= robustNormalization(x )
  x = np.clip(x, None, MAX_VAL_AFTER_NORMALIZATION)  # Just to prevent outliers
  return x

def inputNormalization_2(x):
  from skimage.filters import threshold_otsu
  otsu_thr= threshold_otsu(x)
  out_mean= np.mean(x[x<otsu_thr])
  inner_range= iqr(x[x>=otsu_thr], rng= (10, DEFAULT_PERCENTIL) )
  if inner_range==0:
    raise NormalizationError("warning, bad iqr %.3f. Is your input masked?. Unmasked inputs required" % inner_range)
  x=(x- out_mean)/inner_range
  x = np.clip(x, None, MAX_VAL_AFTER_NORMALIZATION)  # Just to prevent outliers
  return x

def inputNormalization_3(x, noise_stats=None):
  '''
  Performs input normalization using typical cryo-em schema of normalizing according noise:
        let noise mean be 0 and noise std=0.1
  :param x: input volume
  :param noise_stats=(mean_noise, std_noise): The statistics of the noise for the input volumen. If none, it will try to automatically
                                              guess them
  :return:  normalized input
  '''
  if noise_stats is None:
    meanInNoise, stdInNosise= _guessNoiseStats_radialProfile(x)
    print("Noise stats: mean=%f std=%f"%(meanInNoise, stdInNosise))
  else:
    meanInNoise, stdInNosise= noise_stats

  x_norm= (x-meanInNoise)/ (stdInNosise*10)  #Desired noise distribution mean=0 and std=0.1
  assert not np.any(np.isnan(x_norm)), "Error normalizing input. Some nans were generated in the volume. Try an alternative normalization option"
  return x_norm

def _guessNoiseStats_radialProfile(x):
  from .dataUtils import resizeVol
  from scipy import ndimage
  from scipy.signal import argrelextrema

  #First part is a set of heuristics to identify the circular noise around the protein

  x_gauss= gaussianFilter(resizeVol(x, (100, 100, 100)), 0.1 ) #Resize to seep up and filter to reduce noise level.

  win_size=5
  win_mean = ndimage.uniform_filter(x_gauss, win_size)
  win_sqr_mean = ndimage.uniform_filter(x_gauss ** 2, win_size) #This is a good estimation of the protein region
  win_var = win_sqr_mean - win_mean ** 2

  interestingCurve= radial_profile(win_var)-radial_profile(win_mean)
  energyCurve= radial_profile(win_sqr_mean)

  # import matplotlib.pyplot as plt
  # f= plt.figure()
  # plt.plot(radial_profile(win_mean),label='win_mean')
  # plt.plot(radial_profile(win_sqr_mean),label='win_sqr_mean')
  # plt.plot(radial_profile(win_var),label='win_var')
  # plt.plot(radial_profile(win_var)-radial_profile(win_mean),label='win_var_minus_win_mean')
  # plt.legend()
  # f.show()
  #  #plt.show()
  #
  # from devel_code.trainNet.dataManager import plot_vol_and_target
  # plot_vol_and_target(x, x_gauss, win_sqr_mean)


  candidateMinima= argrelextrema(interestingCurve, np.less)[0]
  if len(candidateMinima)>0:
    toLookIndex= np.min(candidateMinima)
    if interestingCurve[toLookIndex]>=0:
      toLookIndex = np.min(np.argmin(interestingCurve))  # Noise border will be at index > toLookIndex
  else:
    toLookIndex = np.min(np.argmin(interestingCurve))  # Noise border will be at index > toLookIndex

  if toLookIndex>50:  #Radial noise, the most typical, has 50 over 100 voxels radius
    candidateNoiseDist = x_gauss.shape[0] // 2
    print("Automatic radial noise detection may have failed. No suitable min index found. Guessing radial noise of radius %s%%"%(candidateNoiseDist))
  else:
    maxInterestingIdx= np.min(np.argmax(interestingCurve[toLookIndex:51])).astype(np.int32)+toLookIndex

    if ( energyCurve[maxInterestingIdx]> interestingCurve[maxInterestingIdx] and interestingCurve[maxInterestingIdx]>0)  : #np.isclose(maxInterestingIdx, maxWinMean, rtol=1e-2):
      raise NormalizationError("Warning, the input might be hollow structure. Automatic masking might fail. Aborting...")
    try:
      toLookIndex2= np.min(np.where(interestingCurve[toLookIndex:]>0))+toLookIndex
      try:
        toLookIndex3 = np.min(np.where(interestingCurve[toLookIndex2:] <= 0)) + toLookIndex2
        candidateNoiseDist = round((toLookIndex2 + toLookIndex3) * 0.5)
        grad_1 = np.mean(np.diff(interestingCurve[-10:]))
        grad_2 = np.mean(np.diff(interestingCurve[-25:-10]))
        if grad_1 > 1e-8 and grad_2 >1e-8 and grad_1 > 3 * grad_2:
          candidateNoiseDist = np.sqrt(3 * (x_gauss.shape[0] // 2) ** 2)
      except ValueError:
        candidateNoiseDist = x_gauss.shape[0] // 2
        print("Automatic radial noise detection may have failed. No trend change found. Guessing radial noise of radius %s %%" % (candidateNoiseDist))
    except ValueError:
      candidateNoiseDist = x_gauss.shape[0] // 2
      print("The input might be a fiber, assuming no masking till radius %d %%"%candidateNoiseDist)

  print("Automatic radial noise detected beyond %s %% of volume side" % (candidateNoiseDist))
  noiseAndProt_regionIdxs= getCoordsWithinSphere(x_gauss, maxDist=candidateNoiseDist)
  protein_region= applyRelativeThr(morphDilation(win_sqr_mean, size=5), r_thr=0.05, robust=False)


  noise_regionMask= np.zeros_like(x_gauss)
  noise_regionMask[noiseAndProt_regionIdxs]=1
  noise_regionMask-=protein_region

  # from devel_code.trainNet.dataManager import plot_vol_and_target
  # plot_vol_and_target(beforeProtein_mask, protein_region, noise_regionMask)

  noise_regionMask= (resizeVol(noise_regionMask, x.shape) >0.5)
  interestingPart= x[noise_regionMask]
  meanInNoise= np.mean(interestingPart)
  stdInNosise = np.std(interestingPart)
  return meanInNoise, stdInNosise

def sigmoid(x, factor=1, offset=0):
  return 1. / (1 + np.exp(-factor*x)) - offset


def minMaxNormalization(x):
  return (x- x.min())/(x.max()-x.min() )


def normalNormalization(x):
  x=(x- x.mean())/(x.std() )
  return x


def robustNormalization(img, iqrRange=(10, DEFAULT_PERCENTIL), raiseInsteadWarn=True, ignoreExtrema=False):
  if ignoreExtrema:
    iqr_val= iqr(img[(img>img.min()) & (img<img.max())], rng= iqrRange )
  else:
    iqr_val= iqr(img, rng= iqrRange )
  if iqr_val==0:
    if raiseInsteadWarn:
      raise NormalizationError("Error, bad iqr %.3f. Is your input masked?. Unmasked inputs required" % iqr_val)
    else:
      iqr_val = iqr(img+ np.random.normal(img.mean(), img.std()*1e-4), rng=iqrRange)
      if iqr_val == 0:
        print("warning, bad iqr", iqr_val)
        iqr_val= (np.max(img)-np.min(img)) + 1e-12
  newImg=(img- np.median(img))/iqr_val
  return newImg


def trimmedNormalization(x, percentil=95, binarizeThr=None):
  x= x.astype(np.float32)
  try:
    x= np.clip(x, 0, np.percentile(x[x>0], percentil))
    x= (x-x.min())/(x.max()-x.min())
  except IndexError:
    pass
  if binarizeThr:
    x[x >= binarizeThr]=1
    x[x < binarizeThr] = 0
  return x

class NormalizationError(GenericError):
  pass

