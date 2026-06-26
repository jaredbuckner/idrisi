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

          minSlope, maxSlope = slope_fn(pID)

        ... where minSlope sets the minimum required slope for a point of a
        given level above all lower levels, and maxSlope is the maximum allowed
        slope for any connected point to pID, create a height limit map hlmap
        such that:

          minDelHeight, maxDelHeight = hlmap[(pID, qID)]

        ... where a valid heightmap is one for which all edges (pID, qID) obey the relationship

          minDelHeight <= hmap.height(qID) - hmap.height(pID) <= maxDelHeight

        Note that minSlope, and maxSlope are allowed to be None, which
        indicates no restriction.  The appropriate hlmap entries will also have
        a value of None.

        slope_fn is not required to accept any argument for levels of False or
        None.

        '''

        hlmap = dict()
        for pID, pPoint in self.enumerate_points():
            pLevel = self.level(pID)

            pSlopeInvl = ((None, None) if pLevel is False or pLevel is None else
                          slope_fn(pID))

            assert(jutil.invl_valid(pSlopeInvl))
            
            for qID in self.neighbors(pID):
                if qID <= pID:
                    continue
                
                qPoint = self.point(qID)
                qLevel = self.level(qID)
                
                delPoint = (qPoint[0] - pPoint[0], qPoint[1] - pPoint[1])
                dist = math.sqrt(delPoint[0] * delPoint[0] + delPoint[1] * delPoint[1])
                assert(not math.isnan(dist))

                pq_cmp = self.cmp(pLevel, qLevel)
                
                if(pq_cmp > 0):
                    # q is lower than p
                    qpInvl = jutil.invl_scale(pSlopeInvl, dist)
                    assert(jutil.invl_valid(qpInvl))
                    hlmap[(pID, qID)] = jutil.invl_conj(qpInvl)
                    
                elif(pq_cmp < 0):
                    # p is lower than q
                    qSlopeInvl = ((None,None) if qLevel is False or qLevel is None else
                                  slope_fn(qID))
                    assert(jutil.invl_valid(qSlopeInvl))

                    pqInvl = jutil.invl_scale(qSlopeInvl, dist)
                    assert(jutil.invl_valid(pqInvl))
                    hlmap[(pID, qID)] = pqInvl
                else:
                    # p is equal-level to q
                    maxDel = None
                    if(pSlopeInvl[0] is not None):
                        maxDel = abs(pSlopeInvl[0])

                    if(pSlopeInvl[1] is not None and
                       (maxDel is None or abs(pSlopeInvl[1]) > maxDel)):
                        maxDel = abs(pSlopeInvl[1])
                        
                    pqInvl = (None,None) if maxDel is None else jutil.invl_scale((-maxDel, maxDel), dist)
                    assert(jutil.invl_valid(pqInvl))

                    hlmap[(pID, qID)] = pqInvl
                    
        return hlmap       
    
    
    def gen_heights(self, slope_fn, sea_height, *,
                    maxHeight=8848, underwaterMul = 3,
                    epsilon=0.1, selectRange=(0.49, 0.51),
                    completedCB=None):
        
        ## This holds the valid height intervals
        nodeIntervals = []
        refractedMin = sea_height / underwaterMul
        hlmap = self.gen_height_limit_map(slope_fn)
        updatedIDs = []

        #for arrow, invl in hlmap.items():
        #    print(f'E [{arrow}] => {invl}')

        for pID, pLevel in self.enumerate_levels():
            ## Populate constraints as we walk forward
            pInterval = ((refractedMin - epsilon),
                         (refractedMin + epsilon if pLevel is None or pLevel is False else
                          maxHeight))
            assert(jutil.invl_valid(pInterval))
            nodeIntervals.append(pInterval)
            updatedIDs.append(pID)
            #print(f'N [{pID}] => {pInterval}')
            
        
        for sID in range(self.point_count()):            
            while(updatedIDs):
                #print(".", end="")
                pID = updatedIDs.pop(0)
                pInterval = nodeIntervals[pID]
                for qID in self.neighbors(pID):
                    qInterval = nodeIntervals[qID]
                    pqInterval = (jutil.invl_conj(hlmap[(pID, qID)]) if pID < qID else
                                  hlmap[(qID, pID)])
                    qConstraint = jutil.invl_sum(qInterval, pqInterval)
                    pInterval = jutil.invl_parallel(pInterval, qConstraint)

                    if(not jutil.invl_valid(pInterval)):
                        raise RuntimeError(f'During constraint propagation at pID {pID} the height interval {pInterval} vanished!  The grid is overconstrained.')
                    
                if(pInterval != nodeIntervals[pID]):
                    #print("!", end="")
                    nodeIntervals[pID] = pInterval
                    updatedIDs.extend(self.neighbors(pID))

            #print("@", end="")
            sInterval = nodeIntervals[sID]
            if(not jutil.invl_closed(sInterval)):
                raise RuntimeError(f'During constraint propagation at pID {sID} the height interval {sInterval} did not close!  The grid is underconstrained.')

            bW = self._jr.uniform(selectRange[0], selectRange[1])
            aW = 1.0 - bW
            sHeight = sInterval[0] * aW + sInterval[1] * bW
            self._height[sID] = sHeight
            nodeIntervals[sID] = (sHeight - epsilon, sHeight + epsilon)
            updatedIDs.extend(self.neighbors(sID))

            #for pID, pInterval in enumerate(nodeIntervals):
            #    print(f'N [{pID}] => {pInterval}')
            if(completedCB is not None):
                completedCB(sID, self.point_count())
            
        for pID in range(self.point_count()):
            if self._height[pID] < 0:
                self._height[pID] *= underwaterMul
        
        self._update_height_stats()
        return 1
    
    def anneal(self, stddev, r):
        for pID in range(self.point_count()):
            self._height[pID] += self._jr.uniform(-2 * stddev, 2 * stddev)
    
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

        hmap.gen_heights(lambda pID: (0, 0.10) if hmap.level(pID) <= 0 else (0, 0.60), -40, maxHeight=984, selectRange=(0.3, 0.7))
            

        
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

        
        
        
