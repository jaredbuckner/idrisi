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

          minSlope, maxSlope = slope_fn(hmap, pID)

        ... generate a height limit map hlmap such that:

          minDelHeight, maxDelHeight = hlmap[(pID, qID)]

        ... where a valid heightmap is one for which all edges (pID, qID) obey the relationship

        minDelHeight <= hmap.height(qID) - hmap.height(pID) <= maxDelHeight
        '''

        hlmap = dict()
        for pID, pPoint in self.enumerate_points():
            pLevel = self.level(pID)
            pMinSlope, pMaxSlope = slope_fn(self, pID)
            
            for qID in self.neighbors(pID):
                if qID <= pID:
                    continue
                qPoint = self.point(qID)
                qLevel = self.level(qID)

                delPoint = (qPoint[0] - pPoint[0], qPoint[1] - pPoint[1])
                dist = math.sqrt(delPoint[0] * delPoint[0] + delPoint[1] * delPoint[1])

                if(self.is_lower(pLevel, qLevel)):
                    # q is lower than p
                    qMinSlope, qMaxSlope = slope_fn(self, qID)
                    minSlope, maxSlope = qMinSlope * dist, qMaxSlope * dist
                    hlmap[(qID, pID)] = (minSlope, maxSlope)
                    hlmap[(pID, qID)] = (-maxSlope, -minSlope)
                elif(self.is_lower(qLevel, pLevel)):
                    # p is lower than q
                    minSlope, maxSlope = pMinSlope * dist, pMaxSlope * dist
                    hlmap[(pID, qID)] = (minSlope, maxSlope)
                    hlmap[(qID, pID)] = (-maxSlope, -minSlope)
                else:
                    # p is equal-level to q
                    qMinSlope, qMaxSlope = slope_fn(self, qID)
                    maxSlope = min(pMaxSlope, qMaxSlope) * dist
                    hlmap[(pID, qID)] = (-maxSlope, maxSlope)
                    hlmap[(qID, pID)] = (-maxSlope, maxSlope)

        return hlmap
    
    def _local_height_limits(self, pID, hlmap, plmap):
        '''Returns (newMin, newMax)'''

        newMin = None
        newMax = None
        
        for qID in self.neighbors(pID):
            qHn, qHx = plmap[qID]
            if(qID > pID):
                ## del = qH - pH  =>  pH = qH - del
                minDel, maxDel = hlmap[(pID, qID)]
                pHx = qHx - minDel
                pHn = qHn - maxDel
            else:
                ## del = pH - qH  =>  pH = qH + del
                minDel, maxDel = hlmap[(qID, pID)]
                pHx = qHx + maxDel
                pHn = qHn + minDel

            if(newMin is None or newMin < pHn):
                newMin = pHn
            if(newMax is None or newMax > pHx):
                newMax = pHx

        return(newMin, newMax)
   
    def gen_heights(self, slope_fn, sea_height, *,
                    maxHeight=8848, underwaterMul = 3,
                    epsilon=0.1, selectRange=(0.5, 0.5)):

        refractedMin = sea_height / underwaterMul
        
        hlmap = self.gen_height_limit_map(slope_fn)
        plmap = list((refractedMin, refractedMin) if pLevel is None or pLevel is False else (refractedMin, maxHeight) for pID, pLevel in self.enumerate_levels())
        
        invalids = set(pID for pID,pLevel in self.enumerate_levels() if pLevel is not None and pLevel is not False)
        needsSkooshing = list(invalids)
        
        while(invalids):
            shuffledinvalids = list(invalids)
            self._jr.shuffle(shuffledinvalids)
            print(f'I({len(invalids)}) S({len(needsSkooshing)}) T({len(plmap)}))')
            
            for pID in shuffledinvalids:
                invalids.remove(pID)
                pHn, pHx = plmap[pID]
                newMin, newMax = self._local_height_limits(pID, hlmap, plmap)

                if(newMin > newMax + epsilon or newMin > pHx + epsilon or pHn > newMax + epsilon):
                    raise RuntimeError(f"For pID={pID} at level {self.level(pID)} with limits ({pHn}, {pHx}), the surrounding points have created impossible limits ({newMin}, {newMax})!")

                changed = False
                if(newMin > pHn + epsilon):
                    changed = True
                    pHn = newMin
                if(newMax < pHx - epsilon):
                    changed = True
                    pHx = newMax

                if changed:
                    plmap[pID] = (pHn, pHx)
                    invalids.update(self.neighbors(pID))
            
            if(not invalids and needsSkooshing):                
                pID = needsSkooshing.pop()
                pHn, pHx = plmap[pID]
                pWx = self._jr.uniform(selectRange[0], selectRange[1])
                pHmid = pHx * pWx + pHn * (1.0 - pWx)
                plmap[pID] = (pHmid, pHmid)
                invalids.update(self.neighbors(pID))


        for pID, limits in enumerate(plmap):
            pHn, pHx = limits
            self._height[pID] = (pHn + pHx) / 2.0
        
        for pID in range(self.point_count()):
            if self._height[pID] < 0:
                self._height[pID] *= underwaterMul
                
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
        
        hmap.gen_heights(lambda hmap, pID: (0.00, 0.02) if hmap.level(pID) is None or hmap.level(pID) is False else (0.00, 0.05) if hmap.level(pID) <= 0 else (0.02, 1.20), -40, maxHeight=984, selectRange=(0.5, 1.0))
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

        
        
        
