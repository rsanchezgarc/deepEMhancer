import os
from subprocess import check_output, CalledProcessError


def resolveDesiredGpus(gpusStr):
  '''
  :param gpusStr: a string representing a gpu selection. Eg: "1,2"
          special options:
            "-1" will select only cpu
            "all" will select all gpus

  :return: [gpuId: int], numberGPUs
  '''

  if gpusStr == '' or gpusStr is None or gpusStr=='-1':
      return [None], 1
  elif gpusStr.startswith("all"):
    if 'CUDA_VISIBLE_DEVICES' in os.environ: #this is for slurm
      gpus= [ elem.strip() for elem in os.environ['CUDA_VISIBLE_DEVICES'].split(",") ]
      return gpus, len(gpus)
    else:
      try:
        nGpus= int(check_output("nvidia-smi -L | wc -l", shell=True))
        gpus= list(range(nGpus))
        return gpus, nGpus
      except (CalledProcessError, FileNotFoundError, OSError):
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
  print("updating environment to select gpu: %s" % (gpuList))
  if gpuList is None:
    gpusStr="-1"
  elif isinstance(gpuList, int):
    gpusStr = int(gpuList)
  elif isinstance(gpuList, list):
    gpusStr = ",".join([ str(elem).strip() for elem in gpuList])
  else:
    gpusStr= gpuList
  os.environ['CUDA_VISIBLE_DEVICES'] = str(gpusStr).replace(" ", "")

