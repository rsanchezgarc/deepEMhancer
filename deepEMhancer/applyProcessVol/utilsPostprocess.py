
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label as label_components

from ..utils.dataUtils import getCoordsWithinSphere, getCoordsOutsideSphere, morphDilation, morphErosion


def morphologicalDilation(img, dilate_size):
  return morphDilation(img, size=dilate_size)

def removeSmallCCs(img, estimated_background_thr=None, min_size_to_preserve=0.1, dilate_size=3):

  if estimated_background_thr is None:
    estimated_background, thr= estimateBackground(img)
    print( estimated_background, thr)
    assert estimated_background < thr, "Error, background should have smaller intensities than signal. Automatic cleaning failed"


  minVal= np.min(img)
  bin_img= morphDilation(img, size=dilate_size)
  bin_img= morphErosion(bin_img, size=dilate_size)

  bin_img[bin_img<thr]=0
  bin_img[bin_img >=thr] = 1
  bin_img= bin_img.astype(bool)
  print("Post-processing step: ", end=".")

  labels, n_labels= label_components(bin_img, return_num=True, background=estimated_background)

  print(".", end="")
  labels_and_num=getLabels2NumVoxels(labels, n_labels)
  print(".", end="")
  allComponents_size= np.sum(labels>0)


  keepIfBiggerThr= min_size_to_preserve*allComponents_size
  final_img= np.ones_like(img)*minVal
  for l, count in labels_and_num:
    if count > keepIfBiggerThr:
      selection= (labels==l)
      final_img[selection]= img[selection]
  print("!")
  return final_img

def getLabels2NumVoxels(labels, n_labels):
  label2Size= np.zeros(n_labels+1, dtype=np.int64)
  flattenLabels= labels.flatten()
  for i in range(flattenLabels.shape[0]):
    label2Size[flattenLabels[i]]+=1

  return [ (l, size) for l, size in enumerate(label2Size) ][1:]


def estimateBackground(img):
  coordsNotInSphere= getCoordsOutsideSphere(img)
  background= np.mean( img[coordsNotInSphere])
  coordsInSmallerSphere= getCoordsWithinSphere(img,0.33)
  notBackground= np.mean( img[coordsInSmallerSphere])
  otsu_thr= threshold_otsu(img)
  thr_select= .5*(notBackground+otsu_thr)
  return background, thr_select


try:
  import numba
  getLabels2NumVoxels= numba.jit(getLabels2NumVoxels, nopython=True, cache=True, parallel=True)
except ImportError:
  pass


def test():
  from ..utils.ioUtils import loadVol
  fname= "devel_code/tests/data/EMD-7023_predLocscale.mrc"
  x= loadVol( fname )
  x_clean= removeSmallCCs(x)

if __name__=="__main__":
  test()