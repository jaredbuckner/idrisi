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
        self._height = dict()
        self._update_height_stats()

    def _update_height_stats(self):
        self._lowest = None
        self._highest = None
        for height in self._height.values():
            if self._lowest is None or height < self._lowest:
                self._lowest = height
            if self._highest is None or height > self._highest:
                self._highest = height

        if self._lowest is not None:
            self._scinterp = jutil.make_array_interp(len(HeightMapper._seacolors), self._lowest, 0.0)

        if self._highest is not None:
            self._lcinterp = jutil.make_array_interp(len(HeightMapper._landcolors), 0.0, self._highest)

    ## Returns minHeight, nomHeight, maxHeight
    def _heights_from_neighbors(self, pID, *,
                                nominalHeight=None,
                                minSlope=0.001,
                                maxSlope=100):
        minHeight = None
        maxHeight = None
        
        pLevel = self.level(pID)
        pPoint = self.point(pID)
        
        for qID in self.neighbors(pID):
            if qID not in self._height: continue

            qPoint = self.point(qID)
            qDel = (qPoint[0] - pPoint[0], qPoint[1] - pPoint[1])
            qDist = math.sqrt(qDel[0] * qDel[0] + qDel[1] * qDel[1])
            qHeight = self._height[qID]
            qLevel = self.level(qID)
            
            aHeight = qHeight + maxSlope * qDist;
            if(maxHeight is None or maxHeight > aHeight):
                maxHeight = aHeight
            
            if(self.is_lower(pLevel, qLevel)):
                dHeight = qHeight + minSlope * qDist;
                if(minHeight is None or minHeight < dHeight):
                    minHeight = dHeight
            ## else:
            ##     nHeight = qHeight - maxSlope * qDist;
            ##     if(minHeight is None or minHeight < nHeight):
            ##         minHeight = nHeight

        if (nominalHeight is None):
            nominalHeight = (maxHeight if minHeight is None else
                             minHeight if maxHeight is None else
                             minHeight / 2 + maxHeight / 2)

        ## If there is no valid region, no reason to clip
        if (minHeight is None or maxHeight is None or maxHeight >= minHeight):
            if (maxHeight is not None and nominalHeight > maxHeight):
                nominalHeight = maxHeight
                
            if (minHeight is not None and nominalHeight < minHeight):
                nominalHeight = minHeight
        
        return(minHeight, nominalHeight, maxHeight)
    
    def is_valid_height(self, pID, *,
                        minSlope=0.001,
                        maxSlope=100):
        pHeight = self._height.get(pID, None)
        if pHeight is None: return(False)
        
        minHeight, nominalHeight, maxHeight = self._heights_from_neighbors(pID,
                                                                           minSlope=0.001,
                                                                           maxSlope=100);
        return(minHeight <= pHeight <= maxHeight)


    def gen_heights(self, slope_fn, sea_height, *,
                    underwaterMul = 3, variance=0.1, epsilon=0.1):
        ## nominalHeight, minSlope, maxSlope = slope_fn(pLevel, pID)
        self._height = dict()

        invalids = dict()

        for pID, pLevel in self._level.items():
            if pLevel is None:
                self._height[pID] = sea_height + self._jr.uniform(-variance, +variance)
                for qID in self.neighbors(pID):
                    qLevel = self.level(qID)
                    if qLevel is not None:
                        invalids[qID] = qLevel

        pvSeq = list(invalids.keys())
        while(pvSeq or invalids):
            if not pvSeq:
                pvSeq = list(invalids.keys())
            
            print(f'{len(pvSeq)}/{len(invalids)}/{len(self._level)-len(self._height)}')
            pID = pvSeq.pop()
            pLevel = invalids.pop(pID)
            
            nominalHeight, minSlope, maxSlope = slope_fn(pLevel, pID)
            minHeight, pHeight, maxHeight = self._heights_from_neighbors(pID,
                                                                         nominalHeight=nominalHeight,
                                                                         minSlope=minSlope,
                                                                         maxSlope=maxSlope)
            ## print(f'{pID=} {nominalHeight=} {minSlope=} {maxSlope=}')
            ## print(f'  {minHeight=}, {pHeight=}, {maxHeight=}')
            if pHeight is not None:
                oldHeight = self._height.get(pID, None)
                if(oldHeight is not None):
                    pHeight = pHeight * 0.9 + oldHeight * 0.1                    
                    if minHeight is not None and pHeight < minHeight:
                        pHeight = minHeight

                self._height[pID] = pHeight
                
                if oldHeight is None or abs(pHeight - oldHeight) > epsilon:
                    for qID in self.neighbors(pID):
                        qLevel = self.level(qID)
                        if qLevel is not None:
                            invalids[qID] = qLevel
                
        self._update_height_stats()            

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
            nines = list(pID for pID in hmap._level if hmap._level[pID] == turn)
            while(nines):
                hmap.add_river_source(self.jr.choice(nines))
                hmap.levelize()
                nines = list(pID for pID in hmap._level if hmap._level[pID] == turn)

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
        
        hmap.gen_heights(lambda pLevel, pID: (-50, 0.005, 0.005) if pLevel <=0 else (pLevel * 900 / ml, 0.02, 0.10), -40)
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

        
        
        
