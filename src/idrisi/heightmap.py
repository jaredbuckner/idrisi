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
        
    def gen_heights(self, slope_fn, sea_height, *,
                    underwaterMul = 3, variance=0.1):
        ## nominalHeight, minSlope, maxSlope = slope_fn(pLevel, pID)
        self._height = dict()
        
        level2nodes = dict()

        for pID, pLevel in self._level.items():
            if pLevel is None:
                self._height[pID] = sea_height + self._jr.uniform(-variance, +variance)
            else:
                level2nodes.setdefault(pLevel, list()).append(pID)

        levelorder = sorted(level2nodes.keys())
        for pLevel in levelorder:
            IDSeq = level2nodes[pLevel]
            # IDSeq.sort(key=lambda pID: min(self._height[qID] for qID in self.neighbors(pID) if qID in self._height))
            for pID in IDSeq:
                pMinHeight = None  ## The minimum height -- the highest + minslope                
                pMaxHeight = None  ## The maximum height -- the lowest + maxslope
                pPoint = self.point(pID)
                pHeight, slopeMin, slopeMax = slope_fn(pLevel, pID)
                for qID in self.neighbors(pID):
                    if(qID in self._height):
                        qHeight = self._height[qID]
                        qLevel = self._level[qID]
                        qPoint = self.point(qID)
                        delta = (pPoint[0] - qPoint[0],
                                 pPoint[1] - qPoint[1])
                        dist = math.sqrt(delta[0]*delta[0] + delta[1]*delta[1])
                        newMinIncr = (dist * slopeMin * (1 if qHeight >=0 else underwaterMul)
                                      + self._jr.uniform(-variance, +variance))
                        if newMinIncr < 0:
                            newMinIncr = 0
                        newMaxIncr = (dist * slopeMax * (1 if qHeight >=0 else underwaterMul)
                                      + self._jr.uniform(-variance, +variance))
                        if qLevel is None or qLevel < pLevel:
                            newMinHeight = qHeight + newMinIncr
                        else:
                            newMinHeight = qHeight - newMaxIncr
                            
                        if pMinHeight is None or pMinHeight < newMinHeight:
                            pMinHeight = newMinHeight
                            
                        newMaxHeight = qHeight + newMaxIncr                            
                        if pMaxHeight is None or pMaxHeight > newMaxHeight:
                            pMaxHeight = newMaxHeight

                self._height[pID] = max(min(pHeight if pHeight is not None else pMaxHeight, pMaxHeight), pMinHeight)
                
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

        
        
        
