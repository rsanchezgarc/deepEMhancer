import os
import shutil
from subprocess import check_output, CalledProcessError


def resolveDesiredGpus(gpusStr):
  '''
  :param gpusStr: a string representing a gpu selection. Eg: "1,2"
          special options:
            "-1" will select only cpu
            "all" will select all gpus

  :return: [gpuId: int], numberGPUs
  '''

  if gpusStr == '' or gpusStr is None or gpusStr == '-1':
      return [None], 1
  elif gpusStr.startswith("all"):
    if 'CUDA_VISIBLE_DEVICES' in os.environ: # this is for schedulers such as Slurm
      gpus = [elem.strip() for elem in os.environ['CUDA_VISIBLE_DEVICES'].split(",") if elem.strip()]
      if not gpus or gpus == ['-1']:
        return [None], 1
      return gpus, len(gpus)
    else:
      for command in ("nvidia-smi", "nvidia-smi.exe"):
        executable = shutil.which(command)
        if executable is None:
          continue
        try:
          output = check_output([executable, "-L"], text=True)
        except (CalledProcessError, FileNotFoundError, OSError):
          continue
        nGpus = len([line for line in output.splitlines() if line.strip()])
        if nGpus > 0:
          gpus = list(range(nGpus))
          return gpus, nGpus
      return [None], 1
  else:
    gpus= [ int(num.strip()) for num in gpusStr.split(",") ]
    return gpus, len(gpus)

def mask_CUDA_VISIBLE_DEVICES(gpuList):
  '''
  Mask out the GPUs that are not included in gpuList
  :param gpuList:  [gpuId: int]
  :return: None
  '''
  if gpuList is None or (isinstance(gpuList, list) and all(elem is None for elem in gpuList)):
    gpusStr="-1"
  elif isinstance(gpuList, int):
    gpusStr = int(gpuList)
  elif isinstance(gpuList, list):
    gpusStr = ",".join([ str(elem).strip() for elem in gpuList])
  else:
    gpusStr= gpuList
  gpusStr = str(gpusStr).replace(" ", "")
  if os.environ.get('CUDA_VISIBLE_DEVICES') != gpusStr:
    print("updating environment to select gpu: %s" % (gpuList))
  os.environ['CUDA_VISIBLE_DEVICES'] = gpusStr


def configureGpuEnvironment(gpuIds):
  """Resolve a device selection and apply it before TensorFlow is imported."""
  if isinstance(gpuIds, str) or gpuIds is None:
    gpuList, nGpus = resolveDesiredGpus(gpuIds)
  else:
    gpuList = gpuIds
    if isinstance(gpuList, (list, tuple)):
      nGpus = max(1, len(gpuList))
    else:
      nGpus = 1
  mask_CUDA_VISIBLE_DEVICES(gpuList)
  return gpuList, nGpus
