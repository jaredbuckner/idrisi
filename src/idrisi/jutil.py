## Some random utilities

import math

def in_radial_fn(pCenter, base, evenAmplSeq, oddAmplSeq):
    '''Create a function that takes a point and returns true if the point is within
    its boundary, false otherwise.  The boundary is defined radially from
    center pCenter, where the average radial distance is given by base and
    whose distance varies based on a sequence of amplitudes given by
    evenAmplSeq and oddAmplSeq, representing the amplitudes of the even and odd
    halves of the frequency componentscos((n+1) * theta) and sin((n+1) *theta)
    portions of the amplitude.
    '''
    def _in_radial(point):
        xdel = point[0]-pCenter[0]
        ydel = point[1]-pCenter[1]
        r = math.sqrt(xdel*xdel + ydel*ydel)
        th = math.atan2(ydel, xdel)
        ampl = base
        for mul, ampls in enumerate(zip(evenAmplSeq, oddAmplSeq),
                                    start=1):
            freq = th*(mul)
            ampl += ampls[0] * math.cos(freq) + ampls[1] * math.sin(freq)

        return(r < ampl)
    
    return (_in_radial)

def make_linear_interp(a, b):
    span = a - b

    def _linterp(v):
        aW = (v-b)/span
        bW = 1 - aW
        return(aW, bW)

    return _linterp

def make_simplex_interp(aPoint, bPoint, cPoint):
    det = (bPoint[1]-cPoint[1])*(aPoint[0]-cPoint[0]) + (cPoint[0]-bPoint[0])*(aPoint[1]-cPoint[1])

    def _sinterp(vPoint):
        aW = ((bPoint[1]-cPoint[1])*(vPoint[0]-cPoint[0]) + (cPoint[0]-bPoint[0])*(vPoint[1]-cPoint[1]))/det
        bW = ((cPoint[1]-aPoint[1])*(vPoint[0]-cPoint[0]) + (aPoint[0]-cPoint[0])*(vPoint[1]-cPoint[1]))/det
        cW = 1 - aW - bW

        return(aW, bW, cW)

    return(_sinterp)

def make_grid_interp(minPoint, maxPoint):
    ## Weights are: (0,0, aW)  (1,0, bW)
    ##              (0,1, cW)  (1,1, dW)
    det = (maxPoint[0]-minPoint[0], maxPoint[1]-minPoint[1])

    def _ginterp(vPoint):
        xW = (vPoint[0] - minPoint[0]) / det[0]
        yW = (vPoint[1] - minPoint[1]) / det[1]
        nxW = 1-xW
        nyW = 1-yW

        return((1-xW)*(1-yW),(xW)*(1-yW),(1-xW)*yW,xW*yW)

    return(_ginterp)


def make_array_interp(tgtArraySize, minVal, maxVal):
    factor = (tgtArraySize - 1) / (maxVal - minVal)
    
    def _ainterp(val):
        pointer = (val - minVal) * factor
        pLow, wHigh = divmod(pointer, 1)
        pHigh = pLow + 1
        wLow = 1 - wHigh
        pLow = int(max(0, min(pLow, tgtArraySize - 1)))
        pHigh = int(max(0, min(pHigh, tgtArraySize - 1)))

        return(pLow, wLow, pHigh, wHigh)

    return(_ainterp)


class Viewport():
    '''Class objects maintain a relationship between the map grid dimensions and a
    view of that grid.
    
    gridSize:    The map is defined as spanning the region between 0,0 and
                 gridSize, inclusive.  Map points may exist beyond this grid.
    
    gridSelMin:  The lower-left corner of the map selection
    gridSelMax:  The upper-right corner of the map selection
    
    viewSize:    The view is defined as spanning the region between 0,0 and
                 viewSize, endpoint-exclusive.  That is, view 0,0 maps to
                 gridSelMin, and view viewSize[0]-1,viewSize[1]-1 maps to
                 gridSelMax
    '''
    
    def __init__(self, gridSize, viewSize):
        self._gridSize = tuple(gridSize)
        self._gridInterp = make_grid_interp((0,0), self._gridSize)
        
        self.set_view_size(viewSize)
        self.reset_grid_sel()
        
    def grid_size(self):
        return self._gridSize

    def grid2weights(self, vPoint):
        return self._gridInterp(vPoint)

    def weights2grid(self, aW, bW, cW, dW):
        return((bW+dW) * self._gridSize[0],
               (cW+dW) * self._gridSize[1])
    
    def grid_sel_size(self):
        return (self._gridSelMin, self._gridSelmax)
    
    def grid_sel_min(self):
        return self._gridSelMin

    def grid_sel_max(self):
        return self._gridSelMax

    def grid_sel2weights(self, vPoint):
        return self._gridSelInterp(vPoint)

    def weights2grid_sel(self, aW, bW, cW, dW):
        return((aW+cW) * self._gridSelMin[0] + (bW+dW) * self._gridSelMax[0],
               (aW+bW) * self._gridSelMin[1] + (cW+dW) * self._gridSelMax[1])
    
    def set_grid_sel(self, gridSelMin, gridSelMax):
        self._gridSelMin = tuple(gridSelMin)
        self._gridSelMax = tuple(gridSelMax)
        self._gridSelInterp = make_grid_interp(self._gridSelMin, self._gridSelMax)

    def reset_grid_sel(self):
        self.set_grid_sel(gridSelMin=(0.0, 0.0),
                          gridSelMax=self._gridSize)

    def zoom_grid_sel(self, factor, *, center=None):
        '''Zoom (resize) the grid selection.
        
        Factor is the amount of zoom.  If center is given, grid zoom is
        centered on the given point, otherwise on the center point.
        '''
        if center is None:
            center = ((self._gridSelMax[0] + self._gridSelMin[0]) / 2.0,
                      (self._gridSelMax[1] + self._gridSelMin[1]) / 2.0)

        selW = 1.0 / factor
        cenW = 1.0 - selW
        
        self.set_grid_sel((self._gridSelMin[0] * selW + center[0] * cenW,
                           self._gridSelMin[1] * selW + center[1] * cenW),
                          (self._gridSelMax[0] * selW + center[0] * cenW,
                           self._gridSelMax[1] * selW + center[1] * cenW))
    
    def recenter_grid_sel(self, center):
        '''Recenter the grid selection
        '''
        fiff = ((self._gridSelMax[0] - self._gridSelMin[0]) / 2.0,
                (self._gridSelMax[1] - self._gridSelMin[1]) / 2.0)

        self.set_grid_sel((center[0] - fiff[0], center[1] - fiff[1]),
                          (center[0] + fiff[0], center[1] + fiff[1]))

    def reaspect_grid_sel(self, ratio=None):
        '''Resize the grid selection to the given aspect ratio (height/width).
        
        If no ratio is given, use the view ratio.
        '''
        if ratio is None:
            ratio = self._viewSize[1] / self._viewSize[0]

        xCenter = (self._gridSelMax[0]+self._gridSelMin[0]) / 2.0
        yCenter = (self._gridSelMax[1]+self._gridSelMin[1]) / 2.0        
        xHSpan = (self._gridSelMax[0]-self._gridSelMin[0]) / 2.0
        yHSpan = (self._gridSelMax[1]-self._gridSelMin[1]) / 2.0

        xHSpanNew = yHSpan / ratio
        if(xHSpanNew >= xHSpan):
            self.set_grid_sel((xCenter-xHSpanNew, yCenter-yHSpan),
                              (xCenter+xHSpanNew, yCenter+yHSpan))
        else:
            yHSpanNew = xHSpan * ratio
            self.set_grid_sel((xCenter-xHSpan, yCenter-yHSpanNew),
                              (xCenter+xHSpan, yCenter+yHSpanNew))
        
    def view_size(self):
        return self._viewSize

    def view2weights(self, vXY):
        return(self._viewInterp(vXY))

    def weights2view(self, aW, bW, cW, dW):
        return ((bW+dW) * self._viewSize[0],
                (cW+dW) * self._viewSize[1])
    
    def set_view_size(self, viewSize):
        self._viewSize = tuple(viewSize)
        self._viewInterp = make_grid_interp((0,0), self._viewSize)
    
    def grid2view(self, point):
        aW, bW, cW, dW = self.grid_sel2weights(point)
        return self.weights2view(aW, bW, cW, dW)

    def view2grid(self, xy):
        aW, bW, cW, dW = self.view2weights(xy)
        return self.weights2grid_sel(aW, bW, cW, dW)
    
