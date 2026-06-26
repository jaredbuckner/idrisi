## A canvas widget with some map-rendering functions

import threading
import tkinter
import idrisi.jutil as jutil
import unittest

class MapCanvasWorker(threading.Thread):
    def __init__(self, inqueue, outqueue, quitevent):
        super().__init__(target=self._work)
        self._inqueue = inqueue
        self._outqueue = outqueue
        self._quitevent = quitevent

    def _work(self):
        while self._quitevent.wait(timeout=0.2) is False:
            try:
                
class MapCanvas(tkinter.Canvas):
    def __init__(self, *args, **kwargs):        
        kwargs.setdefault('bg', 'black')
        super().__init__(*args, **kwargs)

class _ut_MapCanvas(unittest.TestCase):
    def test_mapcanvas(self):
        root = tkinter.Tk()
        myMapCanvas = MapCanvas(root)
        myMapCanvas.pack()

        root.mainloop()
        
        try:
            root.quit()
        except:
            pass

