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
    span = b - a

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
    def __init__(self, gridSize, viewSize, *,
                 gridExpand = 1.0):
        self._gridSize = gridSize
        self._viewSize = viewSize
        minFactor = (1.0 - gridExpand) / 2.0
        maxFactor = (gridExpand + 1.0) / 2.0
        self._overGridMin = (gridSize[0] * minFactor, gridSize[1] * minFactor)
        self._overGridMax = (gridSize[0] * maxFactor, gridSize[1] * maxFactor)

    def gridSize(self):
        return self._gridSize

    def viewSize(self):
        return self._viewSize

    def overGridMin(self):
        return self._overGridMin

    def overGridMax(self):
        return self._overGridMax

    def grid2view(self, point):
        return (point[0] * self._viewSize[0] / self._gridSize[0],
                point[1] * self._viewSize[1] / self._gridSize[1])

    def view2grid(self, xy):
        return (xy[0] * self._gridSize[0] / self._viewSize[0],
                xy[1] * self._gridSize[1] / self._viewSize[1])

