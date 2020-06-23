
class GenericError(Exception):
  def __init__(self, message):
    # Call the base class constructor with the parameters it needs
    super(GenericError, self).__init__(message)
    self.message= message
  def appendToMsg(self, oneStr):
    self.message+= oneStr

  def __str__(self):
      return self.message