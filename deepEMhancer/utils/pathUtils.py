import glob
import os
import shutil


def tryToRemove(fname):
  try:
    if os.path.isdir(fname):
      shutil.rmtree(fname)
    else:
      os.remove(fname)
  except OSError:
    pass


def tryToCreateDir(fname):
  try:
    os.mkdir(fname)
  except OSError as e:
    # print(e)
    pass


def tryToSymLink(fnameIn, fnameOut):
  try:
    os.symlink(fnameIn, fnameOut)
  except (OSError, FileExistsError):
    pass

def getFilesInPaths(pathsList, extensions, abortIfEmpty=True):
  if pathsList is None or len(pathsList)<1:
    fnames=[]
    errorPath=pathsList
  elif isinstance(pathsList, str) or 1 == len(pathsList):
    if not isinstance(pathsList, str) and len(pathsList)==1:
      pathsList= pathsList[0]
    if os.path.isdir(pathsList):
      pathsList= os.path.join(pathsList, "*")
    fnames=glob.glob(pathsList)
    assert len(fnames)>=1 and not os.path.isdir(pathsList), "Error, %s path not found or incorrect"%(pathsList)
    errorPath= pathsList
  else:
    fnames= pathsList
    try:
      errorPath= os.path.split(pathsList[0])[0]
    except IndexError:
      raise Exception("Error, pathList contains erroneous paths "+str(pathsList))
  extensions= set(extensions)
  fnames= [ fname for fname in fnames if fname.split(".")[-1] in extensions ]
  if abortIfEmpty:
    assert len(fnames)>0, "Error, there are no < %s > files in path %s"%(" - ".join(extensions), errorPath)
  return fnames