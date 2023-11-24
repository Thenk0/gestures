import cv2, threading
import queue
from config import (
    WIDTH,
    HEIGHT,
)
# bufferless VideoCapture
class BufferlessVideoCapture:

  def __init__(self, name, screen):
    if not screen:
      self.cap = cv2.VideoCapture(name)
    if screen:
      self.cap = cv2.VideoCapture(name, cv2.CAP_DSHOW)
      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()