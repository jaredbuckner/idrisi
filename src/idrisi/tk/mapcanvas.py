## A canvas widget with some map-rendering functions

import tkinter
import unittest

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

