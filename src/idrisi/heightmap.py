## This builds on the levelmap to make a heightmap

import math
import idrisi.jrandom as jrandom
import idrisi.jutil as jutil
import idrisi.levelmap as levelmap
import os
import PIL.Image
import subprocess
import unittest

class HeightMapper(levelmap.LevelMapper):
    _seacolors = ((0x00, 0x00, 0x83),
                 (0x00, 0xFB, 0xFF))
    _landcolors = ((0x00, 0x62, 0x00),
                   (0xFF, 0xFF, 0xA2),
                   (0xC5, 0x00, 0x00))
    
    def __init__(self, *args, jr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._jr = jrandom.JRandom() if jr is None else jr
        self._height = [0.0] * self.point_count()
        self._update_height_stats()

    def height(self, pID):
        return(self._height[pID])

    def enumerate_heights(self):
        yield from enumerate(self._height)

    def height_count(self):
        return(len(_height))
    
    def _update_height_stats(self):        
        self._lowest = min(self._height)        
        self._highest = max(self._height)

        self._scinterp = jutil.make_array_interp(len(HeightMapper._seacolors), min(self._lowest, -0.01), 0.0)
        self._lcinterp = jutil.make_array_interp(len(HeightMapper._landcolors), 0.0, max(0.01, self._highest))

    def gen_height_limit_map(self, slope_fn):
        '''Given a function:

          minSlope, desiredSlope, maxSlope = slope_fn(pID)

        ... where minSlope sets the minimum required slope for a point of a
        given level above all lower levels, maxSlope is the maximum allowed
        slope for any connected point to pID, and desiredSlope is the desired
        slope for a point of a given level above all lower levels, create a
        height limit map hlmap such that:

          minDelHeight, desiredDelHeight, maxDelHeight = hlmap[(pID, qID)]

        ... where a valid heightmap is one for which all edges (pID, qID) obey the relationship

          minDelHeight <= hmap.height(qID) - hmap.height(pID) <= maxDelHeight

        ... and a heightmap may be scored according to the sum of all errors:

          (hmap.height(qID) - hmap.height(pID) - desiredDelHeight) ** 2

        Note that minSlope, maxSlope, and desiredSlope are allowed to be None,
        which indicates no restriction.  The appropriate hlmap entries will
        also have a value of None.

        slope_fn is not required to accept any argument for levels of False or
        None.

        '''

        hlmap = dict()
        for pID, pPoint in self.enumerate_points():
            pLevel = self.level(pID)

            pMinSlope, pDesiredSlope, pMaxSlope = ((None, None, None) if pLevel is False or pLevel is None else
                                                   slope_fn(pID))
            assert(pMinSlope is None or pDesiredSlope is None or pMinSlope <= pDesiredSlope)
            assert(pMaxSlope is None or pDesiredSlope is None or pDesiredSlope <= pMaxSlope)
            assert(pMinSlope is None or pMaxSlope is None or pMinSlope <= pMaxSlope)
            
            for qID in self.neighbors(pID):
                if qID <= pID:
                    continue
                
                qPoint = self.point(qID)
                qLevel = self.level(qID)
                
                delPoint = (qPoint[0] - pPoint[0], qPoint[1] - pPoint[1])
                dist = math.sqrt(delPoint[0] * delPoint[0] + delPoint[1] * delPoint[1])
                assert(not math.isnan(dist))
                
                if(self.is_lower(pLevel, qLevel)):
                    # q is lower than p
                    minDel = None if pMinSlope is None else pMinSlope * dist
                    desDel = None if pDesiredSlope is None else pDesiredSlope * dist
                    maxDel = None if pMaxSlope is None else pMaxSlope * dist
                    minDelN = None if minDel is None else -minDel
                    desDelN = None if desDel is None else -desDel
                    maxDelN = None if maxDel is None else -maxDel
                    #hlmap[(qID, pID)] = (minDel, desDel, maxDel)
                    hlmap[(pID, qID)] = (maxDelN, desDelN, minDelN)
                elif(self.is_lower(qLevel, pLevel)):
                    # p is lower than q
                    qMinSlope, qDesiredSlope, qMaxSlope = ((None, None, None) if qLevel is False or qLevel is None else
                                                           slope_fn(qID))
                    
                    minDel = None if qMinSlope is None else qMinSlope * dist
                    desDel = None if qDesiredSlope is None else qDesiredSlope * dist
                    maxDel = None if qMaxSlope is None else qMaxSlope * dist
                    minDelN = None if minDel is None else -minDel
                    desDelN = None if desDel is None else -desDel
                    maxDelN = None if maxDel is None else -maxDel
                    
                    hlmap[(pID, qID)] = (minDel, desDel, maxDel)
                    #hlmap[(qID, pID)] = (maxDelN, desDelN, minDelN)
                else:
                    # p is equal-level to q
                    maxDel = None if pMaxSlope is None else pMaxSlope * dist
                    maxDelN = None if maxDel is None else -maxDel

                    
                    hlmap[(pID, qID)] = (maxDelN, None, maxDel)
                    #hlmap[(qID, pID)] = (None, 0.0, None)

        return hlmap       
        
    
    def _gen_height_errors(self, hlmap):
        '''Returns (minArray, errSqArray, errDelArray, maxArray)'''

        pointCount = self.point_count()
        minArray = [None] * pointCount
        errSqArray = [0.0] * pointCount
        errDelArray = [0.0] * pointCount
        maxArray = [None] * pointCount

        for edge, properties in hlmap.items():
            pID, qID = edge
            minDel, desDel, maxDel = properties
            
            pHeight = self._height[pID]
            qHeight = self._height[qID]
            pqDel = qHeight - pHeight

            if(minDel is not None):
                pMax = qHeight - minDel
                if maxArray[pID] is None or pMax < maxArray[pID]:
                    maxArray[pID] = pMax

                qMin = pHeight + minDel
                if minArray[qID] is None or minArray[qID] < qMin:
                    minArray[qID] = qMin

            if(maxDel is not None):
                pMin = qHeight - maxDel
                if minArray[pID] is None or minArray[pID] < pMin:
                    minArray[pID] = pMin

                qMax = pHeight + maxDel
                if maxArray[qID] is None or qMax < maxArray[qID]:
                    maxArray[qID] = qMax

            if(desDel is not None):
                err = pqDel - desDel
                esq = err * err
                errDel = 2 * err

                if(not esq >= 0.0):
                    print(f"FOOBAR! {pHeight} {qHeight} {pqDel} {desDel} {err} {esq}")
                
                assert(esq >= 0)

                ## Account half the error to each node
                errSqArray[pID] += esq / 2.0
                errSqArray[qID] += esq / 2.0

                ## Account for signage in correction factor
                errDelArray[pID] -= errDel
                errDelArray[qID] += errDel
                
        return (minArray, errSqArray, errDelArray, maxArray)
    
    @staticmethod
    def _feedbackRpt(numInvalids, numUnskooshed, totalMapSize):
        print(f'I({numInvalids}) S({numUnskooshed}) T({totalMapSize})')
    
    def gen_heights(self, slope_fn, sea_height, *,
                    maxHeight=8848, underwaterMul = 3,
                    epsilon=0.001, selectRange=(0.5, 0.5),
                    skooshWithin=1, feedbackCB=_feedbackRpt):

        alpha = 0.49
        maxStep = 2.0
        ignoreLimits = False
        
        refractedMin = sea_height / underwaterMul
        
        hlmap = self.gen_height_limit_map(slope_fn)
        for pID, pLevel in self.enumerate_levels():
            if pLevel is None or pLevel is False:
                self._height[pID] = refractedMin

        tErrArray = [None] * 100
        maxIter = 1000
        iterCnt = 0
        isValid = False
        
        while(iterCnt < maxIter):
            iterCnt += 1
            minArray, errSqArray, errDelArray, maxArray = self._gen_height_errors(hlmap)

            invalidCnt = 0
            totalErr = 0;
            worstErr = 0;
            biggestStep = 0;
            
            for pID, pData in enumerate(zip(minArray, errSqArray, errDelArray, maxArray)):
                localInvalid = False
                
                pMin, pErrSq, pErrDel, pMax = pData
                pLevel = self.level(pID)

                if(pMin is None or pMin < refractedMin):
                    pMin = refractedMin

                if(pMax is None or pMax > maxHeight):
                    pMax = maxHeight
                
                if pMin > pMax:
                    localInvalid = True
                    invalidCnt += 1

                if pLevel is None or pLevel is False:
                    continue
                
                if(not ignoreLimits):
                    if(self._height[pID] < pMin):
                        if(pMin - self._height[pID] < 1.0):
                            self._height[pID] = pMin
                        else:
                            self._height[pID] = (self._height[pID] + pMin) / 2.0
                        continue
                    if(self._height[pID] > pMax):
                        if(self._height[pID] - pMax < 1.0):
                            self._height[pID] = pMax
                        else:
                            self._height[pID] = (self._height[pID] + pMax) / 2.0
                            
                        continue
                        
                if pErrSq is not None and pErrDel is not None:
                    localErr = math.sqrt(pErrSq)
                    if(not pErrSq >= 0.0):
                        print(f'AAAH! {pErrSq}')
                    
                    assert(not math.isnan(localErr))
                    totalErr += localErr
                    if(localErr > worstErr):
                        worstErr = localErr

                    assert(not math.isnan(pErrDel))
                    step = -pErrDel / 2.0  ## The double-derivative for each
                                           ## error type is a constant 2
                    if abs(step) > abs(biggestStep):
                        biggestStep = step

                    assert(not math.isnan(alpha))
                    step *= alpha

                    if step > maxStep:
                        step = maxStep
                    elif step < -maxStep:
                        step = -maxStep

                    assert(not math.isnan(step))
                        
                    self._height[pID] += step
                
            print(f'C:{iterCnt} I:{invalidCnt} T:{totalErr} W:{worstErr} S:{biggestStep}')

            if(biggestStep < -maxStep or maxStep < biggestStep):
                alpha *= 0.9
            else:
                alpha *= 1.2
            
            #alpha = abs(maxStep / biggestStep)
            
            if(ignoreLimits or invalidCnt == 0):
                if totalErr == 0 or (tErrArray[0] is not None and tErrArray[0] <= totalErr):
                    isValid = True
                    break
                
                tErrArray.append(totalErr)
                del tErrArray[0]

        for pID in range(self.point_count()):
            if self._height[pID] < 0:
                self._height[pID] *= underwaterMul
        
        self._update_height_stats()
        return isValid
    
    def anneal(self, stddev, r):
        for pID in range(self.point_count()):
            self._height[pID] += r.uniform(-2 * stddev, 2 * stddev)
    
    def height_color(self, pID):
        height = self._height[pID]
        if(height <= 0):
            aIdx, aWt, bIdx, bWt = self._scinterp(height)
            return tuple(a * aWt + b * bWt for a,b in zip(HeightMapper._seacolors[aIdx],
                                                          HeightMapper._seacolors[bIdx]))
        else:
            aIdx, aWt, bIdx, bWt = self._lcinterp(height)
            return tuple(a * aWt + b * bWt for a,b in zip(HeightMapper._landcolors[aIdx],
                                                          HeightMapper._landcolors[bIdx]))


class _ut_HeightMapper(unittest.TestCase):
    def setUp(self):
        self.jr = jrandom.JRandom();
        self.vp = jutil.Viewport(gridSize = (18000, 18000),
                                 viewSize = (1024, 1024))
        self.separate = 110
        
    def quickview(self, view):
        view.save("unittest.png");
        proc = subprocess.Popen(("display", "unittest.png"))
        proc.wait();
        os.remove("unittest.png");

    def quickgrid(self, *, filter=None):
        points = list(p for p in self.jr.punctillate_rect(pMin = self.vp.grid_sel_min(),
                                                          pMax = self.vp.grid_sel_max(),
                                                          distsq = self.separate * self.separate)
                      if filter is None or filter(p))
        lmap = HeightMapper(points, jr=self.jr)
        lmap.forbid_long_edges(5 * self.separate)
        self.assertTupleEqual(tuple(lmap.isolated_nodes()), ())
        
        return lmap
        
    def test_gen_heights(self):
        self.vp.zoom_grid_sel(1.05)
        
        hmap = self.quickgrid()
        self.vp.reset_grid_sel()
        
        hmap.set_hull_sea()
        hmap.levelize()

        for turn in (int(3500 / self.separate), int(1100 / self.separate)):
            nines = list(pID for pID,lev in enumerate(hmap._level) if lev == turn)
            while(nines):
                hmap.add_river_source(self.jr.choice(nines))
                hmap.levelize()
                nines = list(pID for pID,lev in enumerate(hmap._level) if lev == turn)

        hmap.remove_river_stubs(int(1200 / self.separate))
        hmap.levelize()

        ml = hmap.max_level()
        if ml is None or ml is False:
            ml = 0

        view = PIL.Image.new('RGB', self.vp.view_size())
        hmap.draw_edges(view, grid2view_fn=self.vp.grid2view,
                        edge_color_fn = lambda pID, qID: (hmap.level_color(pID, maxLevel=ml),
                                                          hmap.level_color(qID, maxLevel=ml)))
        self.quickview(view)

        for adev in (0.0, 5.0, 2.0):
            if(adev > 0.0):
                hmap.anneal(adev, self.jr)
            
            hmap.gen_heights(lambda pID: (0, 0.01, 1) if hmap.level(pID) <= 0 else (0, 0.30, 1), -40, maxHeight=984, selectRange=(0.0, 1.0))
            # hmap.gen_heights(lambda pID: (0, None, 0.10) if hmap.level(pID) <= 0 else (0, None, 0.60), -40, maxHeight=984, selectRange=(0.0, 1.0))
            

        
        print(f'{hmap._lowest} - {hmap._highest}')
        
        def edge_color_fn(pID, qID):
            pLevel = hmap._level[pID]
            qLevel = hmap._level[qID]
            pIsSea = pLevel is None
            pIsRiver = pLevel is not None and pLevel <=0
            qIsSea = qLevel is None
            qIsRiver = qLevel is not None and qLevel <= 0
            if (pIsRiver and (qIsRiver or qIsSea)) or (pIsSea and qIsRiver):
                return (levelmap.LevelMapper._riverColor,
                        levelmap.LevelMapper._riverColor)
            else:
                return (hmap.height_color(pID),
                        hmap.height_color(qID))
        
        view = PIL.Image.new('RGB', self.vp.view_size())
        hmap.draw_edges(view, grid2view_fn=self.vp.grid2view,
                        edge_color_fn = edge_color_fn)
        self.quickview(view)

        
        
        
