## The basic Delaunay mapping class
## This will get subclassed all sorts of ways to make up the full mapping package

import idrisi.delmap as delmap
import idrisi.jrandom as jrandom
import idrisi.jutil as jutil
import math
import os
import PIL.Image
import subprocess
import unittest

class LevelMapper(delmap.DelMapper):
    _forbiddenColor = (0xA4, 0x00, 0x74)
    _errorColor     = (0xD7, 0xAD, 0xFF)
    _uninitColor    = (0x00, 0x00, 0x46)
    _seaColor       = (0x01, 0x5E, 0x89)
    _riverColor     = (0x03, 0xDB, 0xFF)
    _landColorSeq   = ( (0x00, 0xFF, 0xCF),
                        (0x00, 0x42, 0x00),
                        (0x00, 0x9E, 0x00),
                        (0xD0, 0xEF, 0x8C),
                        (0xFF, 0xFF, 0x00),
                        (0xAE, 0x79, 0x00),
                        (0xFF, 0x03, 0x00),
                        (0x66, 0x00, 0x00),
                        (0xFF, 0x8F, 0x91),
                        (0xFF, 0xFF, 0xFA) )
    
    def __init__(self, pointSeq):
        super().__init__(pointSeq)

        ## The levelization for each location by pId.  Levels are:
        ##  False : Point has not yet been levelized
        ##   None : Point belongs to the sea floor        
        ##   <= 0 : Point belongs to a river.  Zero represents a river source,
        ##          and decrements for each step away from the source.  When
        ##          rivers join, the minimum value (maximal negative distance
        ##          from source) is kept at the joining node.
        ##    > 0 : Point belongs to land.  Value represents the number of
        ##          steps required to reach a river or sea node.
        ## Missing points have not yet been levelized
        self._level = [False] * self.point_count()
        
        ## A set of forbidden edges.  These edges are not allowed in the slope
        ## graph.  Typically these are used to keep the algorithms from
        ## utilizing the long outer edges when constructing rivers or mountain
        ## slopes.
        self._forbiddenEdges = set()

    def level(self, pID):
        return(self._level[pID])

    def enumerate_levels(self):
        yield from enumerate(self._level)

    def level_count(self):
        return(len(self._levels))
    
    def is_edge_forbidden(self, aPID, bPID):
        return(((aPID, bPID) if aPID < bPID else (bPID, aPID)) in self._forbiddenEdges)
               
    def forbid_edge(self, aPID, bPID):
        if(aPID < bPID):
            self._forbiddenEdges.add((aPID, bPID))
        else:
            self._forbiddenEdges.add((bPID, aPID))

    def forbid_long_edges(self, limit):
        ## If edge is longer than limit, forbid it!
        limitsq = limit * limit
        for aPID, aPoint in self.enumerate_points():
            for bPID in self.adjacent_nodes(aPID):
                if(bPID < aPID):
                    continue
                bPoint = self.point(bPID)

                xdel = aPoint[0] - bPoint[0]
                ydel = aPoint[1] - bPoint[1]
                norm = xdel * xdel + ydel * ydel
                if(norm > limitsq):
                    self.forbid_edge(aPID, bPID)
        
    def isolated_nodes(self):
        ## Yield any nodes which are isolated -- all its adjacent edges are
        ## forbidden!
        for aPID, aPoint in self.enumerate_points():
            if all(self.is_edge_forbidden(aPID, bPID) for bPID in self.adjacent_nodes(aPID)):
                yield aPID

    def set_hull_sea(self):
        for aPID, bPID in self.convex_hull_edges():
            self._level[aPID] = None
            self._level[bPID] = None
        for sID, simplex in self.enumerate_simplices():
            pID, qID, rID = simplex
            if(self.is_edge_forbidden(pID, qID) or
               self.is_edge_forbidden(qID, rID) or
               self.is_edge_forbidden(rID, pID)):
                self._level[pID] = None
                self._level[qID] = None
                self._level[rID] = None
        
    def set_functional_sea(self, *select_fn_list):
        ## This is named backward.  The functions in the list should return
        ## true if the point is part of a land or river area.  A point is
        ## marked as sea if all of the functions applied to it return false
        for pID, point in self.enumerate_points():
            if not any(inside(point) for inside in select_fn_list):
                self._level[pID] = None

    def set_simplex_sea(self, pointSeq):
        ## Given a sequence of points (x, y), find the simplex containing each
        ## point and mark it as sea
        for point in pointSeq:
            sID = self.containing_simplex(point)
            if(sID != -1):
                for pID in self.simplex(sID):
                    self._level[pID] = None

    def set_fill_sea(self, pID):
        ## Given a point, fill it and its unset neighbors (recursively) as sea
        stuffToFill = {pID}
        while(stuffToFill):
            pID = stuffToFill.pop()
            if self._level[pID] is False:
                self._level[pID] = None
                stuffToFill.update(self.neighbors(pID))
    

    def neighbors(self, pID):
        ## Yield the adjacent nodes whose edges are not forbidden
        yield from (qID for qID in self.adjacent_nodes(pID) if not self.is_edge_forbidden(pID, qID))

    def community(self, pID, maxDist):
        ## Return a dict[node] => distance for nodes within dist of pID
        ## (including pID), as measured along non-forbidden edges
        result = dict()
        result[pID] = 0.0
        toCheck = {pID}
        
        while(toCheck):
            pID = toCheck.pop()
            pPoint = self.point(pID)
            pDist = result[pID]
            rDist = maxDist - pDist
            rDistSq = rDist * rDist
            for qID in self.neighbors(pID):
                qPoint = self.point(qID)
                pqPoint = (qPoint[0] - pPoint[0], qPoint[1]-pPoint[1])
                pqLenSq = pqPoint[0] * pqPoint[0] + pqPoint[1] * pqPoint[1]
                if pqLenSq <= rDistSq:
                    pqLen = math.sqrt(pqLenSq)
                    qDist = pDist + pqLen
                    if qID not in result or qDist < result[qID]:
                        result[qID] = qDist
                        toCheck.add(qID)

        return result

    def is_lower(self, pLevel, qLevel):
        ## Returns true if qLevel is lower than pLevel
        return(pLevel is not None and
               (qLevel is None or
                (qLevel is not None and qLevel < pLevel)));
    
    def max_level(self):
        ## Returns False if no level has been set,
        ##         None if only sea levels have been set, or
        ##         the max level value found
        ml = False
        
        for pID, level in enumerate(self._level):
            if (ml is False or ml is None or (level is not False and level is not None and level > ml)):
                ml = level

        return ml

    def _level_from_neighbors(self, pID, *, shoreLevel=1):
        nLevel = None
        
        for qID in self.neighbors(pID):
            qLevel = self._level[qID]
            if qLevel is False:
                continue
            if(qLevel is None):
                if nLevel is None or nLevel >= shoreLevel:
                    nLevel = shoreLevel - 1
            elif(qLevel <= 0):
                return(1)
            else:
                if nLevel is None or nLevel > qLevel:
                    nLevel = qLevel

        return nLevel if nLevel is None else nLevel + 1
        
    def is_valid_level(self, pID, *, shoreLevel=1):
        ## Check a point and return True if the level value for this point
        ## exists and obeys all rules, False otherwise
        
        pLevel = self._level[pID]

        ## Empty level is not valid
        if pLevel is False:
            return(False)
        
        ## Sea level is always valid
        if pLevel is None:
            return(True)
        
        ## Rivers must have an outflow
        if(pLevel <= 0):            
            return(any(self.is_lower(pLevel, self._level[qID])
                       for qID in self.neighbors(pID) if self._level[qID] is not False))
        
        ## Non-shore land must be one level one higher than its lowest neighbor
        ##
        pOughtToBe = self._level_from_neighbors(pID, shoreLevel=shoreLevel)
        return pOughtToBe is not None and pOughtToBe == pLevel

    def levelizables(self, *, shoreLevel=1):
        yield from (pID for pID, pPoint in self.enumerate_points() if
                    not self.is_valid_level(pID, shoreLevel=shoreLevel) and
                    any(self.is_valid_level(qID, shoreLevel=shoreLevel) for qID in self.neighbors(pID)))    
        
    def levelize(self, *, shoreLevel=1):
        invalids = set(self.levelizables(shoreLevel=shoreLevel))
        
        while(invalids):
            pID = invalids.pop()
            pLevel = self._level_from_neighbors(pID, shoreLevel=shoreLevel)
            if(pLevel is not None):
                self._level[pID] = pLevel
                for qID in self.neighbors(pID):
                    if(not self.is_valid_level(qID, shoreLevel=shoreLevel)):
                        invalids.add(qID)

        
    def add_river_source(self, pID):
        pLevel = self._level[pID]
        river = list()
        while(pLevel is not None and pLevel > -len(river)):
            river.append(pID)
            for qID in self.neighbors(pID):
                qLevel = self._level[qID]
                if qLevel is None:
                    pID = qID
                    pLevel = None
                    break
                if qLevel < pLevel:
                    pLevel = qLevel
                    pID = qID

        for negLevel, pID in enumerate(river):
            self._level[pID] = -negLevel

    def remove_river_stubs(self, minLength):
        ## Look for forks, mark for deletion
        toDelete = set()
        for pID, pLevel in enumerate(self._level):
            if pLevel is None or pLevel is False or pLevel < 0:
                longInBranches = list()
                shortInBranches = list()
                
                for qID in self.neighbors(pID):
                    qLevel = self._level[qID]
                    if (qLevel is not None and
                        (pLevel is None or pLevel is False or pLevel < qLevel) and
                        qLevel < 1):
                        if -qLevel < minLength:
                            shortInBranches.append(qID)
                        else:
                            longInBranches.append(qID)
                if(pLevel is None or pLevel is False or longInBranches):
                    toDelete.update(shortInBranches)
                else:
                    toDelete.update(shortInBranches[1:])

        # Delete
        while(toDelete):
            pID = toDelete.pop()
            pLevel = self._level[pID]
            self._level[pID] = False
            for qID in self.neighbors(pID):
                qLevel = self._level[qID]
                if(qLevel is not None and qLevel is not False and pLevel < qLevel < 1):
                    toDelete.add(qID)

    def level_color(self, pID, *, maxLevel=1):
        
        level = self._level[pID]
        if(level is False):
            return LevelMapper._uninitColor
        
        if(level is None):
            return LevelMapper._seaColor
        
        if(level <= 0):
            return LevelMapper._riverColor

        if(maxLevel <= len(LevelMapper._landColorSeq)):
            clu = level - 1
            if clu < 0:
                return(LevelMapper._landColorSeq[0])
            elif clu < len(LevelMapper._landColorSeq):
                return(LevelMapper._landColorSeq[clu])
            else:
                return LevelMapper._landColorSeq[-1]

        else:
            clu = (level-1)*(len(LevelMapper._landColorSeq)-1)/(maxLevel - 1)
            cfidx, wc = divmod(clu, 1)
            cfidx = int(cfidx)
            wf = 1 - wc
            ccidx = cfidx + 1
            if(ccidx == len(LevelMapper._landColorSeq)):
                ccidx = cfidx

            return tuple(a * wf + b * wc for a,b in zip(LevelMapper._landColorSeq[cfidx],
                                                        LevelMapper._landColorSeq[ccidx]))
    
    def gen_drain_levels(self):
        drains = dict()

        for pID, pLevel in enumerate(self._level):
            if pLevel == 0:
                while(pID is not None):
                    drains[pID] = 1 + drains.get(pID, 0)
                    for qID in self.neighbors(pID):
                        qLevel = self._level[qID]
                        if qLevel is None or qLevel is False:
                            pID = None
                            break
                        if qLevel < pLevel:                            
                            pID = qID
                            pLevel = qLevel

        return drains                    
                
class _ut_LevelMapper(unittest.TestCase):
    def setUp(self):
        self.jr = jrandom.JRandom();
        self.vp = jutil.Viewport(gridSize = (1024, 768),
                                 viewSize = (1024, 768))
        self.separate = 11
        
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
        lmap = LevelMapper(points)
        lmap.forbid_long_edges(5 * self.separate)
        self.assertTupleEqual(tuple(lmap.isolated_nodes()), ())
        
        return lmap
        
    def test_forbidden_edges(self):
        self.vp.zoom_grid_sel(1.1)
        lmap = self.quickgrid()
        self.vp.reset_grid_sel()
        
        def color_edge(pID, qID):
            self.assertEqual(lmap.is_edge_forbidden(pID, qID), lmap.is_edge_forbidden(qID, pID))
            if lmap.is_edge_forbidden(pID, qID):
                return LevelMapper._forbiddenColor
            else:
                return LevelMapper._uninitColor

        self.vp.set_view_size((1920,1080))
        self.vp.reaspect_grid_sel()
        view = PIL.Image.new('RGB', self.vp.view_size())
        lmap.draw_edges(view, grid2view_fn=self.vp.grid2view,
                        edge_color_fn = lambda pID, qID: (color_edge(pID, qID), color_edge(qID, pID)))
        self.quickview(view)


    def test_community(self):
        self.vp.zoom_grid_sel(1.1)
        lmap = self.quickgrid()
        self.vp.reset_grid_sel()

        community = lmap.community(0, self.separate * 10)

        def color_node(pID):
            if(pID in community):
                return LevelMapper._forbiddenColor
            else:
                return LevelMapper._uninitColor

        self.vp.set_view_size((1920,1080))
        self.vp.reaspect_grid_sel()
        view = PIL.Image.new('RGB', self.vp.view_size())
        lmap.draw_edges(view, grid2view_fn=self.vp.grid2view,
                        edge_color_fn = lambda pID, qID: (color_node(pID), color_node(qID)))
        self.quickview(view)


    def test_functional_sea(self):
        self.vp.zoom_grid_sel(1.1)
        lmap = self.quickgrid()

        gCenter = self.vp.weights2grid_sel(0.5, 0, 0, 0.5)
        gHSpan = self.vp.weights2grid_sel(-0.5, 0, 0, 0.5)
        self.vp.reset_grid_sel()
        
        maxrad = min(gHSpan[0], gHSpan[1])
        minrad = maxrad / 1.5
        base, evenAmplSeq, oddAmplSeq = self.jr.tonal_rand(minrad, maxrad, 11)

        rfun = jutil.in_radial_fn(gCenter, base, evenAmplSeq, oddAmplSeq)

        lmap.set_functional_sea(rfun)

        self.vp.set_view_size((1920,1080))
        self.vp.reaspect_grid_sel()
        view = PIL.Image.new('RGB', self.vp.view_size())
        lmap.draw_edges(view, grid2view_fn=self.vp.grid2view,
                        edge_color_fn = lambda pID, qID: (lmap.level_color(pID),
                                                          lmap.level_color(qID)))
        self.quickview(view)

    def test_simplex_sea(self):
        self.vp.zoom_grid_sel(1.1)
        self.separate = 7
        lmap = self.quickgrid()

        pFrom = self.vp.weights2grid_sel(0.5, *self.jr.rand_sum_to_n(0.5, 3))
        pTo = self.vp.weights2grid_sel(*self.jr.rand_sum_to_n(0.5, 3), 0.5)
        self.vp.reset_grid_sel()
        
        kp = list(self.jr.koch2_path(pFrom, pTo, self.separate * self.separate / 100.0, fixedR=None, leanLeft=False))
        kp.append(pTo)
        
        lmap.set_simplex_sea(kp)

        self.vp.set_view_size((1600,1200))
        self.vp.reaspect_grid_sel()
        view = PIL.Image.new('RGB', self.vp.view_size())
        lmap.draw_edges(view, grid2view_fn=self.vp.grid2view,
                        edge_color_fn = lambda pID, qID: (lmap.level_color(pID),
                                                          lmap.level_color(qID)))
        for kPoint in kp:
            kXY = tuple(int(p) for p in self.vp.grid2view(kPoint))
            view.putpixel(kXY, LevelMapper._errorColor)
        
        self.quickview(view)

    def test_fill_sea(self):
        self.vp = jutil.Viewport(gridSize = (1920, 1080),
                                 viewSize = (1920, 1080))
        self.vp.zoom_grid_sel(1.1)
        self.separate = 7
        lmap = self.quickgrid()

        anchorweights = ((0.1, 0.1), (0.1, 0.5), (0.5, 0.5), (0.5, 0.9), (0.9, 0.9), (0.9, 0.3), (0.4, 0.3), (0.1, 0.1))
        anchorpoints = tuple(self.vp.weights2grid_sel((1-a)*(1-b),
                                                      a*(1-b),
                                                      (1-a)*b,
                                                      a*b) for a,b in anchorweights)

        anchorleans = (True, None, None, True, None, None, True)

        self.vp.reset_grid_sel()
        
        kp = list()
        for p0, p1, lean in zip(anchorpoints, anchorpoints[1:], anchorleans):
            kp.extend(self.jr.koch2_path(p0, p1, self.separate * self.separate / 100.0, fixedR=1/4, leanLeft=lean))

        lmap.set_simplex_sea(kp)
        for pID, qID in lmap.convex_hull_edges():
            lmap.set_fill_sea(pID)

        lmap.levelize()
        ml = lmap.max_level()
        if ml is None or ml is False:
            ml = 0
            
        view = PIL.Image.new('RGB', self.vp.view_size())
        lmap.draw_edges(view, grid2view_fn=self.vp.grid2view,
                        edge_color_fn = lambda pID, qID: (lmap.level_color(pID, maxLevel=ml),
                                                          lmap.level_color(qID, maxLevel=ml)))
        for kPoint in kp:
            kXY = tuple(int(p) for p in self.vp.grid2view(kPoint))
            if(0 <= kXY[0] < self.vp.grid_size()[0] and 0 <= kXY[1] < self.vp.grid_size()[1]):
                #view.putpixel(kXY, LevelMapper._errorColor)
                pass
        
        self.quickview(view)

    
    def test_levelize(self):
        self.separate = 7
        self.vp.zoom_grid_sel(1.05)
        lmap = self.quickgrid()
        self.vp.zoom_grid_sel(1.1)
        
        kp = list()
        weights = ((0.5, 0.5, 0, 0), (0, 0.5, 0, 0.5), (0, 0, 0.5, 0.5), (0.5, 0, 0.5, 0), (0.5, 0.5, 0, 0))
        anchors = tuple(self.vp.weights2grid_sel(*w) for w in weights)

        self.vp.reset_grid_sel()
        
        for p0, p1 in zip(anchors, anchors[1:]):
            kp.extend(self.jr.koch2_path(p0, p1, self.separate * self.separate / 100.0,
                                         fixedR=self.jr.uniform(1/5,1/3), leanLeft=None))

        lmap.set_simplex_sea(kp)
        
        for pID, qID in lmap.convex_hull_edges():
            lmap.set_fill_sea(pID)

        lmap.levelize()
        
        for turn in (7,):
            nines = list(pID for pID,lev in enumerate(lmap._level) if lev == turn)
            while(nines):
                lmap.add_river_source(self.jr.choice(nines))
                lmap.levelize()
                nines = list(pID for pID,lev in enumerate(lmap._level) if lev == turn)

        lmap.remove_river_stubs(12)
        
        for shoreLevel in (1, 5, 1):
            lmap.levelize(shoreLevel=shoreLevel)

            ml = lmap.max_level()
            if ml is None or ml is False:
                ml = 0

            view = PIL.Image.new('RGB', self.vp.view_size())
            lmap.draw_edges(view, grid2view_fn=self.vp.grid2view,
                            edge_color_fn = lambda pID, qID: (lmap.level_color(pID, maxLevel=ml),
                                                              lmap.level_color(qID, maxLevel=ml)))
            self.quickview(view)

    
    
    def test_alternate_start(self):
        self.separate = 7
        self.vp.zoom_grid_sel(1.05)
        
        gCenter = self.vp.weights2grid_sel(0.5, 0, 0, 0.5)
        gHSpan = self.vp.weights2grid_sel(-0.5, 0, 0, 0.5)
        self.vp.reset_grid_sel()

        maxrad = min(gHSpan[0], gHSpan[1])
        minrad = maxrad / 1.5
        base, evenAmplSeq, oddAmplSeq = self.jr.tonal_rand(minrad, maxrad, 11)

        rfilter = jutil.in_radial_fn(gCenter, base, evenAmplSeq, oddAmplSeq)
        lmap = self.quickgrid(filter=rfilter)

        lmap.set_hull_sea()
        lmap.levelize()

        ml = lmap.max_level()
        if ml is None or ml is False:
            ml = 0
        
        view = PIL.Image.new('RGB', self.vp.view_size())
        lmap.draw_edges(view, grid2view_fn=self.vp.grid2view,
                        edge_color_fn = lambda pID, qID: (lmap.level_color(pID, maxLevel=ml),
                                                          lmap.level_color(qID, maxLevel=ml)))
        self.quickview(view)
