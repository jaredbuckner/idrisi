## Some random utilities

import math

from typing import Any, Callable, Iterable, Optional, Tuple, TypeAlias, TypeVar, Union

## Named types for typing
FPoint: TypeAlias = Tuple[float, float]
IPoint: TypeAlias = Tuple[int, int]
APoint = TypeVar('APoint', FPoint, IPoint)
Interval: TypeAlias = Tuple[Optional[float], Optional[float]]


def in_radial_fn(pCenter: APoint, base: float, evenAmplSeq: Iterable[float], oddAmplSeq: Iterable[float]) -> Callable[[APoint], bool]:
    '''Create a function that takes a point and returns true if the point is within
    its boundary, false otherwise.  The boundary is defined radially from
    center pCenter, where the average radial distance is given by base and
    whose distance varies based on a sequence of amplitudes given by
    evenAmplSeq and oddAmplSeq, representing the amplitudes of the even and odd
    halves of the frequency componentscos((n+1) * theta) and sin((n+1) *theta)
    portions of the amplitude.
    '''
    def _in_radial(point: APoint) -> bool:
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

def make_linear_interp(a: float, b: float) -> Callable[[float], Tuple[float, float]]:
    span = a - b
    if span == 0: raise ValueError('Interp values cannot be the same')

    def _linterp(v: Any) -> Tuple[float, float]:
        aW = (v-b)/span
        bW = 1 - aW
        return(aW, bW)

    return _linterp

def make_simplex_interp(aPoint: APoint, bPoint: APoint, cPoint: APoint) -> Callable[[APoint], Tuple[float, float, float]]:
    det = (bPoint[1]-cPoint[1])*(aPoint[0]-cPoint[0]) + (cPoint[0]-bPoint[0])*(aPoint[1]-cPoint[1])

    def _sinterp(vPoint: APoint) -> Tuple[float, float, float]:
        aW = ((bPoint[1]-cPoint[1])*(vPoint[0]-cPoint[0]) + (cPoint[0]-bPoint[0])*(vPoint[1]-cPoint[1]))/det
        bW = ((cPoint[1]-aPoint[1])*(vPoint[0]-cPoint[0]) + (aPoint[0]-cPoint[0])*(vPoint[1]-cPoint[1]))/det
        cW = 1 - aW - bW

        return(aW, bW, cW)

    return(_sinterp)

def make_grid_interp(minPoint: APoint, maxPoint: APoint) -> Callable[[APoint], Tuple[float, float, float, float]]:
    ## Weights are: (0,0, aW)  (1,0, bW)
    ##              (0,1, cW)  (1,1, dW)
    det = (maxPoint[0]-minPoint[0], maxPoint[1]-minPoint[1])
    if det[0] == 0 or det[1] == 0: raise ValueError('Interp points cannot be gridwise coincidental')

    def _ginterp(vPoint: APoint) -> Tuple[float, float, float, float]:
        xW = (vPoint[0] - minPoint[0]) / det[0]
        yW = (vPoint[1] - minPoint[1]) / det[1]
        nxW = 1-xW
        nyW = 1-yW

        return((1-xW)*(1-yW),(xW)*(1-yW),(1-xW)*yW,xW*yW)

    return(_ginterp)


def make_array_interp(tgtArraySize: int, minVal: float, maxVal: float) -> Callable[[float], Tuple[int, float, int, float]]:
    span = maxVal - minVal
    if span == 0: raise ValueError('Interp values must be different')

    factor = (tgtArraySize - 1) / span
    
    def _ainterp(val: float) -> Tuple[int, float, int, float]:
        pointer = (val - minVal) * factor
        pLow, wHigh = divmod(pointer, 1)
        pHigh = pLow + 1
        wLow = 1 - wHigh
        pLow = int(max(0, min(pLow, tgtArraySize - 1)))
        pHigh = int(max(0, min(pHigh, tgtArraySize - 1)))

        return(pLow, wLow, pHigh, wHigh)

    return(_ainterp)


## Interval helper methods
def invl_conj(a: Interval) -> Interval:
    return (None if a[1] is None else -a[1],
            None if a[0] is None else -a[0])

def invl_sum(a: Interval, b: Interval) -> Interval:
    return(None if a[0] is None or b[0] is None else a[0] + b[0],
           None if a[1] is None or b[1] is None else a[1] + b[1])

def invl_scale(a: Interval, factor: float) -> Interval:
    # A negative scaling factor can create an invalid interval.  It is highly
    # advised to check afterward with invl_valid()

    return(None if a[0] is None else a[0] * factor,
           None if a[1] is None else a[1] * factor)

def invl_parallel(a: Interval, b: Interval) -> Interval:
    return(b[0] if a[0] is None else a[0] if b[0] is None else max(a[0], b[0]),
           b[1] if a[1] is None else a[1] if b[1] is None else min(a[1], b[1]))

def invl_valid(a: Interval) -> bool:
    return(a[0] is None or a[1] is None or a[0] <= a[1])

def invl_closed(a: Interval) -> bool:
    return(a[0] is not None and a[1] is not None and a[0] <= a[1])


class Viewport():
    '''Maintains mappings between a logical grid and a selected sub-rectangle
    of that grid to a continuous view space (e.g., pixel coordinates before
    rounding). The grid spans (0,0) to gridSize inclusive; map data may exist
    outside this box. A grid selection (gridSelMin, gridSelMax) defines the
    current visible region and defaults to the whole grid. The view spans
    (0,0) to viewSize with exclusive ends, so (0,0) maps to gridSelMin and
    (viewSize[0], viewSize[1]) maps to gridSelMax; callers decide how/if to
    clamp or round to actual pixel indices. Conversion helpers accept/return
    bilinear weights (aW,bW,cW,dW) ordered as (top-left, top-right,
    bottom-left, bottom-right) and weights always sum to 1.'''
    
    def __init__(self, gridSize: FPoint, viewSize: FPoint):
        self._gridSize = tuple(gridSize)
        self._gridInterp = make_grid_interp((0.0, 0.0), self._gridSize)
        
        self.set_view_size(viewSize)
        self.reset_grid_sel()
        
    def grid_size(self) -> FPoint:
        '''Return the full grid dimensions as (width, height).'''
        return self._gridSize

    def grid2weights(self, vPoint: APoint) -> Tuple[float, float, float, float]:
        '''Bilinear weights for a grid point relative to the full grid.'''
        return self._gridInterp(vPoint)

    def weights2grid(self, aW: float, bW: float, cW: float, dW: float) -> FPoint:
        '''Convert full-grid weights back to a grid point.'''
        return((bW+dW) * self._gridSize[0],
               (cW+dW) * self._gridSize[1])
    
    def grid_sel_size(self) -> Tuple[FPoint, FPoint]:
        return (self._gridSelMin, self._gridSelMax)
    
    def grid_sel_min(self) -> FPoint:
        '''Lower-left corner of the current grid selection.'''
        return self._gridSelMin

    def grid_sel_max(self) -> FPoint:
        '''Upper-right corner of the current grid selection.'''
        return self._gridSelMax

    def grid_sel2weights(self, vPoint: APoint) -> Tuple[float, float, float, float]:
        '''Bilinear weights for a grid point relative to the selection.'''
        return self._gridSelInterp(vPoint)

    def weights2grid_sel(self, aW: float, bW: float, cW: float, dW: float) -> FPoint:
        '''Convert selection-relative weights to a grid point.'''
        return((aW+cW) * self._gridSelMin[0] + (bW+dW) * self._gridSelMax[0],
               (aW+bW) * self._gridSelMin[1] + (cW+dW) * self._gridSelMax[1])
    
    def set_grid_sel(self, gridSelMin: APoint, gridSelMax: APoint) -> None:
        self._gridSelMin = tuple((float(v) for v in gridSelMin))
        self._gridSelMax = tuple((float(v) for v in gridSelMax))
        self._gridSelInterp = make_grid_interp(self._gridSelMin, self._gridSelMax)

    def reset_grid_sel(self) -> None:
        '''Reset selection to the full grid.'''
        self.set_grid_sel(gridSelMin=(0.0, 0.0),
                          gridSelMax=self._gridSize)

    def zoom_grid_sel(self, factor: float, *, center: Optional[APoint] = None) -> None:
        '''Zoom selection by factor around center (default: selection center).'''
        if center is None:
            center = ((self._gridSelMax[0] + self._gridSelMin[0]) / 2.0,
                      (self._gridSelMax[1] + self._gridSelMin[1]) / 2.0)

        selW = 1.0 / factor
        cenW = 1.0 - selW
        
        self.set_grid_sel((self._gridSelMin[0] * selW + center[0] * cenW,
                           self._gridSelMin[1] * selW + center[1] * cenW),
                          (self._gridSelMax[0] * selW + center[0] * cenW,
                           self._gridSelMax[1] * selW + center[1] * cenW))
    
    def recenter_grid_sel(self, center: APoint) -> None:
        '''Recenter selection to a new center point without scaling.'''
        fiff = ((self._gridSelMax[0] - self._gridSelMin[0]) / 2.0,
                (self._gridSelMax[1] - self._gridSelMin[1]) / 2.0)

        self.set_grid_sel((center[0] - fiff[0], center[1] - fiff[1]),
                          (center[0] + fiff[0], center[1] + fiff[1]))

    def reaspect_grid_sel(self, ratio: Optional[float] = None) -> None:
        '''Resize selection to ratio (h/w), defaulting to the view ratio.'''
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
        
    def view_size(self) -> FPoint:
        '''Return the view dimensions as (width, height).'''
        return self._viewSize

    def view2weights(self, vXY: APoint) -> Tuple[float, float, float, float]:
        '''Bilinear weights for a view point relative to the view.'''
        return(self._viewInterp(vXY))

    def weights2view(self, aW: float, bW: float, cW: float, dW: float) -> FPoint:
        '''Convert view-relative weights to a view point.'''
        return ((bW+dW) * self._viewSize[0],
                (cW+dW) * self._viewSize[1])
    
    def set_view_size(self, viewSize: FPoint) -> None:
        '''Update view dimensions and refresh its interpolator.'''
        self._viewSize = tuple(viewSize)
        self._viewInterp = make_grid_interp((0.0, 0.0), self._viewSize)
    
    def grid2view(self, point: APoint) -> FPoint:
        '''Map a grid point in the selection to view coordinates.'''
        aW, bW, cW, dW = self.grid_sel2weights(point)
        return self.weights2view(aW, bW, cW, dW)

    def view2grid(self, xy: APoint) -> FPoint:
        '''Map a view point to the corresponding grid point in selection.'''
        aW, bW, cW, dW = self.view2weights(xy)
        return self.weights2grid_sel(aW, bW, cW, dW)
    
