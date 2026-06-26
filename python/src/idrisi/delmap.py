## The basic Delaunay mapping class
## This will get subclassed all sorts of ways to make up the full mapping package

from scipy.spatial import Delaunay
import unittest

## I only need these for unit testing, but how to import conditionally?
import idrisi.jrandom as jrandom
import idrisi.jutil as jutil
import os
import PIL.Image
import subprocess

class DelMapper:
    def __init__(self, pointSeq):
        self._grid = Delaunay(pointSeq)
        self._nindptr, self._nindices = self._grid.vertex_neighbor_vertices

    def point(self, pID):
        return(self._grid.points[pID])
    
    def enumerate_points(self):
        yield from enumerate(self._grid.points)

    def point_count(self):
        return(len(self._grid.points))

    def adjacent_nodes(self, pID):
        yield from self._nindices[self._nindptr[pID]:self._nindptr[pID+1]]

    def convex_hull_edges(self):
        for face in self._grid.convex_hull:
            yield tuple(face)

    def simplex(self, sID):
        return self._grid.simplices[sID][0]
    
    def enumerate_simplices(self):
        yield from enumerate(self._grid.simplices)
    
    def containing_simplex(self, point):
        return(self._grid.find_simplex((point,)))

    def draw_edges(self, view, *,
                   grid2view_fn,
                   edge_color_fn):
        XYs = tuple(grid2view_fn(pPoint) for pPoint in self._grid.points)
        for pID, pXY in enumerate(XYs):

            for qID in self.adjacent_nodes(pID):                
                if qID < pID:
                    continue
                
                pColor, qColor = edge_color_fn(pID, qID)
                if pColor is None or qColor is None:
                    continue

                qXY = XYs[qID]
                xspan = qXY[0]-pXY[0]
                yspan = qXY[1]-pXY[1]
                ispan = int(2 * max(abs(xspan), abs(yspan)) + 1)
                for part in range(ispan + 1):
                    weight = part / ispan
                    unweight = 1 - weight
                    vXY = (int(weight * pXY[0] + unweight * qXY[0]),
                           int(weight * pXY[1] + unweight * qXY[1]))
                    if(0 <= vXY[0] < view.width and
                       0 <= vXY[1] < view.height):
                        vColor = tuple(int(weight * p + unweight * q) for p,q in zip(pColor, qColor))
                        if len(vColor) == 1:
                            vColor = vColor[0]
                        view.putpixel(vXY, vColor)

    def draw_nodes(self, view, *,
                   grid2view_fn,
                   node_color_fn):
        for pID, pPoint in self.enumerate_points():
            pColor = node_color_fn(pID)
            if pColor is None:
                continue
            
            pXY = grid2view_fn(pPoint)
            if(0 <= pXY[0] < view.width and
               0 <= pXY[1] < view.height):
                if len(pColor) == 1:
                    pColor = pColor[0]
                view.putpixel((int(pXY[0]), int(pXY[1])), pColor)

    def draw_simplices(self, view, *,
                       view2grid_fn,
                       simplex_color_fn,
                       antialias=1,
                       combine_fn=min):
        ## Cache the last simplex interpreter, since nearby points are likely
        ## to be in the same simplex
        sinterp = None
        aColor = None
        bColor = None
        cColor = None
        
        for x in range(view.width):
            for y in range(view.height):
                vXY = (x, y)
                vPoint = view2grid_fn(vXY)

                vColorSet = []
                for xBit in range(antialias):
                    for yBit in range(antialias):
                        bitXY = (x + xBit / antialias,
                                 y + yBit / antialias)
                        vPoint = view2grid_fn(bitXY)
                        
                        aW, bW, cW = sinterp(vPoint) if sinterp is not None else (None, None, None)
                        
                        ## But if there was no interperter, or if one of the weights is
                        ## out of range...
                        if (sinterp is None or not (0 <= aW <= 1 and
                                                    0 <= bW <= 1 and
                                                    0 <= cW <= 1)):
                            ## Recalculate!
                            sinterp = None
                            sID = self.containing_simplex(vPoint)
                            if(sID == -1):
                                continue
                            
                            aID, bID, cID = self.simplex(sID)
                            aPoint = self.point(aID)
                            bPoint = self.point(bID)
                            cPoint = self.point(cID)
                            
                            aColor, bColor, cColor = simplex_color_fn(aID, bID, cID)
                            
                            if(aColor is None or bColor is None or cColor is None):
                                continue
                            
                            sinterp = jutil.make_simplex_interp(aPoint, bPoint, cPoint)
                            
                            aW, bW, cW = sinterp(vPoint)

                        vColor = tuple(int(aW * a + bW * b + cW * c) for a,b,c in zip(aColor, bColor, cColor))
                        if len(vColor) == 1:
                            vColor = vColor[0]

                        vColorSet.append(vColor)

                if(vColorSet):
                    vColor = min(vColorSet)                
                    view.putpixel(vXY, vColor)
                
                

class _ut_DelMapper(unittest.TestCase):
    def setUp(self):
        self.jr = jrandom.JRandom();
        self.vp = jutil.Viewport(gridSize = (1024, 768),
                                 viewSize = (1024, 768))
        self.separate = 10
        
    def quickview(self, view):
        view.save("unittest.png");
        proc = subprocess.Popen(("display", "unittest.png"))
        proc.wait();
        os.remove("unittest.png");

    def quickgrid(self):
        points = list(self.jr.punctillate_rect(pMin = self.vp.grid_sel_min(),
                                               pMax = self.vp.grid_sel_max(),
                                               distsq = self.separate * self.separate))
        dmap = DelMapper(points)

        for pID in range(dmap.point_count()):
            qCnt = sum(1 for qID in dmap.adjacent_nodes(pID))
            self.assertGreater(qCnt, 1)

        return(dmap)
        
    def test_random_nodes(self):
        self.vp.zoom_grid_sel(1.1);
        dmap = self.quickgrid();
        self.vp.reset_grid_sel();
        view = PIL.Image.new('RGB', self.vp.view_size());
        dmap.draw_nodes(view, grid2view_fn=self.vp.grid2view, node_color_fn=lambda nID: (255,255,255))
        self.quickview(view)

    def test_random_edges(self):
        dmap = self.quickgrid();
        self.vp.zoom_grid_sel(0.9);
        view = PIL.Image.new('RGB', self.vp.view_size());
        dmap.draw_edges(view,
                        grid2view_fn=self.vp.grid2view,
                        edge_color_fn=lambda pID, qID: ((255,255,255),
                                                        (255,255,255)))
        self.quickview(view)
        
    def test_random_simplexes(self):
        colors = ( (255, 0, 0),
                   (0, 255, 0),
                   (0, 0, 255) )

        eColors = ( (0, 255, 255),
                    (255, 0 ,255),
                    (255, 255, 0) )

        self.separate = 100
        dmap = self.quickgrid();
        view = PIL.Image.new('RGB', self.vp.view_size());
        dmap.draw_simplices(view,
                            view2grid_fn=self.vp.view2grid,
                            simplex_color_fn=lambda pID, qID, rID: (colors[pID%3],
                                                                    colors[qID%3],
                                                                    colors[rID%3]),
                            antialias=4)
        dmap.draw_edges(view,
                        grid2view_fn=self.vp.view2grid,
                        edge_color_fn=lambda pID, qID: (eColors[pID%3], eColors[qID%3]))
        
        self.quickview(view)

    def test_adjacent_nodes(self):
        self.vp.zoom_grid_sel(1.1);
        self.separate = 50
        dmap = self.quickgrid();
        self.vp.reset_grid_sel()

        rootID = 0
        adjacents = set(dmap.adjacent_nodes(rootID))

        def colorNode(pID):
            return ((255, 0, 0) if pID == rootID else
                    (128, 255, 128) if pID in adjacents else
                    (64, 64, 64))

        view = PIL.Image.new('RGB', self.vp.view_size())
        dmap.draw_edges(view,
                        grid2view_fn=self.vp.grid2view,
                        edge_color_fn = lambda pID, qID: (colorNode(pID), colorNode(qID)))
        
        self.quickview(view)
