## Stuff useful for Cities Skylines
import idrisi.levelmap as levelmap
import idrisi.heightmap as heightmap
import idrisi.jrandom as jrandom
import idrisi.jutil as jutil
import math
import os
import PIL.Image
import subprocess
import tqdm

class CS(heightmap.HeightMapper):
    ## These are for displaying heights in a CS-centric manner
    seasinterp = jutil.make_array_interp(len(heightmap.HeightMapper._seacolors), -40, 0)
    landinterp = jutil.make_array_interp(len(heightmap.HeightMapper._landcolors), 0, 984)
    
    @staticmethod
    def heightcolor(h):
        if(h <= 0):
            aIdx, aWt, bIdx, bWt = CS.seasinterp(h)
            return tuple(a * aWt + b * bWt for a,b in zip(heightmap.HeightMapper._seacolors[aIdx],
                                                          heightmap.HeightMapper._seacolors[bIdx]))
        else:
            aIdx, aWt, bIdx, bWt = CS.landinterp(h)
            return tuple(a * aWt + b * bWt for a,b in zip(heightmap.HeightMapper._landcolors[aIdx],
                                                          heightmap.HeightMapper._landcolors[bIdx]))    

    def __init__(self, *,
                 gridSize=(18000, 18000),  ## A Cities Skylines map covers 18km x 18km
                 viewSize=(1081, 1081),    ## The import file must be 1081px x 1081 px
                 displaySize=(1200, 1200), ## The size for intermediate displays
                 fillRect=(-2000, -2000,   ## The area to be filled with grid points
                           20000, 20000),  ## Default to 2km beyond each edge
                 separate=None,            ## The average distance between grid points
                                           ## Default to one point per px, calculated
                 seed=None,                ## Random seed, per random.Random.seed(a=seed)
                 ):
        
        self.jr = jrandom.JRandom(seed)
        
        self.vp = jutil.Viewport(gridSize = gridSize,
                                 viewSize = displaySize)
        self.gridSize = gridSize
        self.viewSize = viewSize
        self.displaySize = displaySize
        self.fillRect = fillRect
        if separate is None:
            self.separate = math.sqrt(gridSize[0] * gridSize[1]
                                      / viewSize[0] / viewSize[1])
        else:
            self.separate = separate

        
        self.vp.set_grid_sel((self.fillRect[0], self.fillRect[1]),
                             (self.fillRect[2], self.fillRect[3]))
    
        points = list(self.jr.punctillate_rect(pMin = self.vp.grid_sel_min(),
                                               pMax = self.vp.grid_sel_max(),
                                               distsq = self.separate * self.separate))

        super().__init__(points, jr=self.jr)
        self.forbid_long_edges(10*self.separate)

    ######################
    ## Helper functions ##
    ######################

    def quickview(self, view, fname="cs.png", keep=False):
        view.save(fname)
        proc = subprocess.Popen(("display", fname))
        if not keep:
            proc.wait()
            os.remove(fname)
        return

    def level_edge_color_fn(self, *, maxLevel):
        def _edge_color_fn(pID, qID):
            return(self.level_color(pID, maxLevel=maxLevel),
                   self.level_color(qID, maxLevel=maxLevel))
        return _edge_color_fn
    
    def height_edge_color_fn(self, *, drainMap):
        def _edge_color_fn(pID, qID):
            pLevel = self._level[pID]
            qLevel = self._level[qID]
            pIsSea = pLevel is None
            pIsRiver = pLevel is not None and pLevel <=0 and pID in drainMap
            qIsSea = qLevel is None
            qIsRiver = qLevel is not None and qLevel <= 0 and qID in drainMap
            if (pIsRiver and (qIsRiver or qIsSea)) or (pIsSea and qIsRiver):
                return (levelmap.LevelMapper._riverColor,
                        levelmap.LevelMapper._riverColor)
            else:
                return (CS.heightcolor(self.height(pID)),
                        CS.heightcolor(self.height(qID)))
            
        return _edge_color_fn

    def draw_levels(self, *, maxLevel):
        view = PIL.Image.new('RGB', self.vp.view_size())
        self.draw_edges(view,
                        grid2view_fn=self.vp.grid2view,
                        edge_color_fn=self.level_edge_color_fn(maxLevel=maxLevel))

        return view

    def draw_heights(self, *, drainMap):
        view = PIL.Image.new('RGB', self.vp.view_size())
        self.draw_edges(view,
                        grid2view_fn=self.vp.grid2view,
                        edge_color_fn=self.height_edge_color_fn(drainMap=drainMap))

        return view

    def overlay_shore_edges(self, view, shorePointSeq):
        for kPoint in shorePointSeq:
            kXY = tuple(int(p) for p in self.vp.grid2view(kPoint))
            if(0 <= kXY[0] < view.width and 0 <= kXY[1] < view.height):
                view.putpixel(kXY, levelmap.LevelMapper._errorColor)

    def overlay_parcels(self, view):
        rectMin = tuple(int(v) for v in self.vp.grid2view((4000, 4000)))
        rectMax = tuple(int(v) for v in self.vp.grid2view((14000, 14000)))

        for x in range(rectMin[0], rectMax[0]+1, 2):
            if 0 <= x < view.width:
                for y in (rectMin[1], rectMax[1]):
                    if 0 <= y < view.height:
                        view.putpixel((x, y), (192, 192, 192))

        for y in range(rectMin[1], rectMax[1]+1, 2):
            if 0 <= y < view.height:
                for x in (rectMin[0], rectMax[0]):
                    if 0 <= x < view.width:
                        view.putpixel((x, y), (192, 192, 192))
                
    ## This creates a set of weights which "drag" rivers toward the centerPoint
    def gen_select_weights(self, centerPoint):
        sw = list()
        for pID, point in self.enumerate_points():
            delPoint = (point[0] - centerPoint[0],
                        point[1] - centerPoint[1])
            sw.append(delPoint[0] * delPoint[0] + delPoint[1] * delPoint[1])

        swmax = max(sw)
        for pID, weight in enumerate(sw):
            sw[pID] = swmax - weight + 1

        return sw
    
    def extend_rivers(self, *,
                      meterStr=None,
                      riverSeparation,
                      riverSegmentLength,
                      riverSegmentVar=0,
                      minRiverLength = None,
                      seaShoreOffsetMin = 0,
                      seaShoreOffsetMax = 0,
                      riverShoreOffsetMin = 0,
                      riverShoreOffsetMax = 0,
                      riverShoreSqueeze = 0,
                      selectWeights=None,
                      maxIterations=None,
                      retarget=True,
                      allowedByID=None):

        riverClipLevel  = None if minRiverLength is None else max(int(minRiverLength / self.separate), 1)
        seaShoreMin     = max(int(seaShoreOffsetMin / self.separate), 1)
        seaShoreMax     = max(int(seaShoreOffsetMax / self.separate), 1)
        riverShoreMin   = max(int(riverShoreOffsetMin / self.separate), 1)
        riverShoreMax   = max(int(riverShoreOffsetMax / self.separate), 1)
        squeezeLevels   = int(riverShoreSqueeze / self.separate)

        riverSepLevel = max(1, int(riverSeparation) / self.separate)

        ## River segment limits
        rsln = max(1,int((riverSegmentLength - riverSegmentVar) / self.separate))
        rslx = max(1,int((riverSegmentLength + riverSegmentVar) / self.separate))
        rsla = (rsln + rslx) / 2

        ## River selection choice level limits
        rscln = (riverSepLevel + rsln) // 2
        rsclx = (riverSepLevel + rslx) // 2
        
        probableIterations = math.ceil(sum(1 for pID, pLevel in self.enumerate_levels() if pLevel is not None and pLevel is not False and rsln <= pLevel and (retarget or pLevel <= rslx)) / riverSepLevel / rsla)
        if maxIterations is not None and maxIterations < probableIterations:
            probableIterations = maxIterations
            
        targets = list(pID for pID, pLevel in self.enumerate_levels()
                       if pLevel is not None and pLevel is not False and
                       (allowedByID is None or allowedByID[pID]) and
                       rscln <= pLevel <= rsclx)
        
        iteration = 0
        meter = tqdm.tqdm(total=probableIterations, desc=meterStr, leave=True)
        while(targets and (maxIterations is None or iteration < maxIterations)):
            iteration += 1
            
            if(selectWeights is None):
                tID = self.jr.choice(targets)
            else:
                localWeights = tuple(selectWeights[pID] for pID in targets)
                tID = self.jr.choices(targets, localWeights)[0]

            tLevel = self.level(tID)
            skipmin = max(0, tLevel - rslx)
            skipmax = max(0, tLevel - rsln)
            skip=self.jr.randint(skipmin, skipmax)

            self.add_river_source(tID, skip=skip)
            self.levelize()
            
            if retarget:
                targets = list(pID for pID, pLevel in self.enumerate_levels()
                               if pLevel is not None and pLevel is not False and
                               (allowedByID is None or allowedByID[pID]) and
                               rscln <= pLevel <= rsclx)
            else:
                targets = list(pID for pID in targets if rscln <= self.level(pID) <= rsclx)

            meter.update()

        if(riverClipLevel is not None):
            self.remove_river_stubs(riverClipLevel)

        minLevel = self.min_level()
        
        if minLevel is not None and squeezeLevels > 0:
            wtFn = jutil.make_linear_interp(minLevel, 0)
            squeezeFn = lambda i: int(wtFn(i)[1] * squeezeLevels + 0.5)
        else:
            squeezeFn = lambda i: 0


        self.levelize(seaShoreMin=seaShoreMin, seaShoreMax=seaShoreMax,
                      riverShoreMin=riverShoreMin, riverShoreMax=riverShoreMax, riverLiftFn=squeezeFn)

        meter.close()
        return self.max_level()

    def make_slope_fn(self, distGradeSeq, *,
                      drainMap, riverGradePower=2,
                      shoreGrade, wrinkGrade, riverMinGrade, sourceGrade, peakGrade):
        landSlopes = list()
        for distance, grade in distGradeSeq:
            level = int(distance / self.separate + 0.5)
            slope = grade / math.sqrt(1 - grade*grade)
            landSlopes.append((level, slope))

        shoreSlope = shoreGrade / math.sqrt(1 - shoreGrade*shoreGrade)
        wrinkSlope = wrinkGrade / math.sqrt(1 - wrinkGrade*wrinkGrade)
        sourceSlope = sourceGrade / math.sqrt(1 - sourceGrade*sourceGrade)
        peakSlope = peakGrade / math.sqrt(1 - peakGrade*peakGrade)

        def _slope_fn(pID):
            pLevel = self.level(pID)

            assert(pLevel is not None and pLevel is not False)

            if pLevel <= 0:
                if pID in drainMap:
                    dmag = riverGradePower ** max(0, drainMap[pID] - 1)
                    return(riverMinGrade, 2.0 * max(riverMinGrade, sourceSlope / dmag))
                
                else:
                    return(riverMinGrade, 2.0 * wrinkSlope)

            for landLevel, landSlope in landSlopes:
                if pLevel <= landLevel:
                    return(0, 2.0 * landSlope)

            return(0, 2.0 * peakSlope)
        
        return(_slope_fn)

    def make_genheight_meter_with_cb(self):
        lastTotal = self.point_count()
        meter = tqdm.tqdm(total=self.point_count(), leave=True)
        def meterCB(nS, nT):
            nonlocal lastTotal
            
            if lastTotal != nT:
                lastTotal = nT
                meter.reset(total=lastTotal)
            
            meter.n = nS
            meter.refresh()
            
        return meter, meterCB

    def gen_heights_really_hard(self, slopeFn):
        meter, meterCB = self.make_genheight_meter_with_cb()
        self.gen_heights(slopeFn, sea_height=-40, maxHeight=984,
                         selectRange=(0.45, 0.55),
                         completedCB=meterCB)
        
        meter.close()

    
    def punch_rivers(self, *, drainMap, widthatsource=13, meanDepth=18):
        widthatsource = 13
        riverbeds = dict()
        for rID, rSize in drainMap.items():
            wfactor = self.jr.uniform(0.8, 1.2)
            communityDist = widthatsource * max(1, (2 * rSize - 1)) * wfactor
            expectedCommunitySize = 2 * math.pi * communityDist * communityDist / self.separate / self.separate
            community = tuple(self.community(rID, widthatsource * max(1, (2 * rSize - 1)) * wfactor))
            if rID not in community:
                community = community + (rID,)

            dfactor = 1.0 / wfactor
            if(len(community) < expectedCommunitySize):
                dfactor *= expectedCommunitySize / len(community)
            
            for bID in community:
                if bID not in riverbeds or riverbeds[bID] < dfactor:
                    riverbeds[bID] = dfactor

        for rID, dfactor in riverbeds.items():
            rHeight = self._height[rID]
            rDepth = meanDepth * dfactor

            self._height[rID] -= (rDepth if rHeight >= 0 else
                                  (40 + rHeight) / 40 * rDepth if rHeight >= -40 else
                                  0)

    def draw_cs_heightmap(self):
        ## Let's write out the cs heightmap
        self.vp.set_view_size((1081, 1081))
        
        def hmc(pID):
            return (min(65535, max(0, int((self._height[pID] + 40) * 64))),)
        
        view=PIL.Image.new('I;16', self.vp.view_size())
        self.draw_simplices(view, view2grid_fn=self.vp.view2grid,
                            simplex_color_fn = lambda aID, bID, cID: ( hmc(aID),
                                                                       hmc(bID),
                                                                       hmc(cID)))
        return view


    def level_stats(self, combineRivers=False):
        ## returns {level:  {'height': (n, min, mean, max, stddev),
        ##                   'slope':  {level => (n, min, mean, max, stddev),
        ##                              ...}
        ##                   },
        ##          ...}
        retmap = dict()
        for pID, pPoint in self.enumerate_points():
            pLevel = self.level(pID)
            if combineRivers and pLevel is not None and pLevel < 0:
                pLevel = 0
            
            if pLevel not in retmap:
                retmap[pLevel] = {'height': [],
                                  'slope': {}}
            levelmap = retmap[pLevel]
            levelheights = levelmap['height']
            levelslopemap = levelmap['slope']
            
            pHeight = self.height(pID)
            levelheights.append(pHeight)

            for qID in self.neighbors(pID):
                qLevel = self.level(qID)
                if combineRivers and qLevel is not None and qLevel < 0:
                    qLevel = 0
                
                if qLevel not in levelslopemap:
                    levelslopemap[qLevel] = []

                qPoint = self.point(qID)
                pqDel = (qPoint[0] - pPoint[0],
                         qPoint[1] - pPoint[1])
                pqDist = math.sqrt(pqDel[0] * pqDel[0] +
                                   pqDel[1] * pqDel[1])

                qHeight = self.height(qID)
                pqSlope = (qHeight - pHeight) / pqDist

                levelslopemap[qLevel].append(pqSlope)

        def statistify(seqIter):
            cnt = 0
            summa = 0
            sumsq = 0
            n = None
            x = None

            for value in seqIter:
                cnt += 1
                summa += value
                sumsq += value * value
                if n is None or value < n:
                    n = value
                if x is None or value > x:
                    x = value

            mean = summa / cnt

            return (cnt, n, mean, x, math.sqrt(sumsq/cnt - mean*mean))

        for lv in retmap.values():
            lv['height'] = statistify(lv['height'])
            for svi in lv['slope'].keys():
                lv['slope'][svi] = statistify(lv['slope'][svi])

        return retmap

        
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description = "Idrisi Map Generator tuned for Cities Skylines Heightmaps")

    parser.add_argument('--separate', type=float, metavar='DIST', default=53,
                        help="mean separation between grid points")
    parser.add_argument('--river_separation', type=float, metavar='DIST', default=None,
                        help="minimum distance between flowing rivers")
    parser.add_argument('--river_segment_length', type=float, metavar='DIST', default=None,
                        help="mean length of river segments")
    parser.add_argument('--river_source_grade', type=float, metavar='PCT', default=None,
                        help="target grade of rivers at source")
    parser.add_argument('--river_min_grade', type=float, metavar='PCT', default=None,
                        help="minimum target grade for all rivers")
    parser.add_argument('--wrinkle_mean_length', type=float, metavar='DIST', default=None,
                        help="mean length of invisible feeder streams")
    parser.add_argument('--wrinkle_dev_length', type=float, metavar='DIST', default=None,
                        help="standard deviation from mean length of invisible feeder streams")
    parser.add_argument('--wrinkle_grade', type=float, metavar='PCT', default=None,
                        help="target grade of invisible feeder streams")
    parser.add_argument('--river_mean_depth', type=float, metavar='DEPTH', default=None,
                        help="target mean depth for rivers")
    parser.add_argument('--shore_grade', type=float, metavar='PCT', default=None,
                        help="target grade of underwater land adjacent to shore")
    parser.add_argument('--flood_plain_width', type=float, metavar='DIST', default=None,
                        help="target width of flood plain around river mouths")
    parser.add_argument('--flood_plain_grade', type=float, metavar='PCT', default=None,
                        help="target grade of flood plain")
    parser.add_argument('--foothill_width', type=float, metavar='DIST', default=None,
                        help="target width of foothill adjacent to flood plain")
    parser.add_argument('--foothill_grade', type=float, metavar='PCT', default=None,
                        help="target grade of foothill")
    parser.add_argument('--midhill_width', type=float, metavar='DIST', default=None,
                        help="target width of middle part of hill")
    parser.add_argument('--midhill_grade', type=float, metavar='PCT', default=None,
                        help="target grade of midhill")
    parser.add_argument('--shoulder_width', type=float, metavar='DIST', default=None,
                        help="target width of shoulder of peak above the midhill")
    parser.add_argument('--shoulder_grade', type=float, metavar='PCT', default=None,
                        help="target grade of shoulder")
    parser.add_argument('--peak_grade', type=float, metavar='PCT', default=None,
                        help="target grade of highest peaks")
    parser.add_argument('--show_stats', action='store_true')
    
    
    args=parser.parse_args()
    
    
    
    ## First, generate the base grid of points for the map.  A larger separate
    ## value generates a more blocky grid, yet much faster.
    csmap = CS(separate = args.separate)

    print(f"Grid of {csmap.point_count()} points generated with separation={csmap.separate:.2f}.")
    
    ## Before we generate the map, let's plan out the slopes and levels.
    
    ## First, we need the nearest distance that flowing rivers will be allowed.
    ## We probably don't want anything nearer than two large rivers per playing
    ## grid (2kmx2km), and we probably don't want anything less than having to
    ## go a grid away to get to a river.
    riverSeparation = (args.river_separation if args.river_separation is not None
                       else csmap.jr.uniform(1250, 2250))
    print(f"Flowing rivers shall be separated by no less than {riverSeparation:.2f}m")
    
    ## Rivers are made out of segments.  This gives them a lovely wriggly shape
    riverSegmentLength = (args.river_segment_length if args.river_segment_length is not None
                          else csmap.jr.uniform(200,500))
    print(f"Flowing rivers shall be constructed out of {riverSegmentLength:.2f}m segments")
    
    ## Rivers have a grade at their source.  They get less steep as they flow
    riverSourceGrade = (args.river_source_grade / 100 if args.river_source_grade is not None
                        else csmap.jr.uniform(0.05, 0.21))    
    riverMinGrade = (args.river_min_grade / 100 if args.river_min_grade is not None
                     else csmap.jr.uniform(0.001, 0.02))
    print(f"River source grade:  {100*riverMinGrade:.1f}% - {100*riverSourceGrade:.1f}%")
    
    ## Beyond the river are short invisible feeder streams.  They make the land more wrinkly.
    wrinkleMeanLength = (args.wrinkle_mean_length if args.wrinkle_mean_length is not None
                         else csmap.jr.uniform(0, riverSeparation / 4))
    wrinkleDevLength = (args.wrinkle_dev_length if args.wrinkle_dev_length is not None
                        else csmap.jr.uniform(0, wrinkleMeanLength))
    wrinkleGrade = (args.wrinkle_grade / 100 if args.wrinkle_grade is not None
                    else riverSourceGrade * csmap.jr.uniform(1, 1.5))
    print(f"Feeder streams will be from {wrinkleMeanLength-wrinkleDevLength:.2f}m to {wrinkleMeanLength+wrinkleDevLength:.2f}m long")
    print(f"Feeder streams have a target grade of {100*wrinkleGrade:.1f}%")
    
    meanDepth = (args.river_mean_depth if args.river_mean_depth is not None
                 else csmap.jr.uniform(10, 20))
    print(f"Rivers have a mean depth of {meanDepth:.2f}m")
    
    ## Width and grade of various land bits
    shoreGrade      = (args.shore_grade / 100 if args.shore_grade is not None
                       else csmap.jr.uniform(0.10, 0.50))
    floodPlainWidth = (args.flood_plain_width if args.flood_plain_width is not None
                       else csmap.jr.uniform(csmap.separate, riverSeparation * 3 / 4))
    floodPlainGrade = (args.flood_plain_grade / 100 if args.flood_plain_grade is not None
                       else csmap.jr.uniform(0.01, 0.03))

    ## Foothill + Midhill = riverSeparation / 2
    if args.foothill_width is None:
        if args.midhill_width is None:            
            footHillWidth, midHillWidth = csmap.jr.rand_sum_to_n(riverSeparation / 4, 2)
            footHillWidth += riverSeparation / 8
            midHillWidth += riverSeparation / 8
        else:
            footHillWidth = max(0, riverSeparation / 2 - args.midhill_width)
            midHillWidth = args.midhill_width
    elif args.midhill_width is None:
        footHillWidth = args.foothill_width
        midHillWidth = max(0, riverSeparation / 2 - args.foothill_width)
    else:
        footHillWidth = args.foothill_width
        midHillWidth = args.midhill_width
    
    footHillGrade   = (args.foothill_grade / 100 if args.foothill_grade is not None else
                       csmap.jr.uniform(0.01, 0.20))
    midHillGrade    = (args.midhill_grade / 100 if args.midhill_grade is not None else
                       csmap.jr.uniform(0.15, 0.45))
    shoulderWidth   = (args.shoulder_width if args.shoulder_width is not None else
                       csmap.jr.uniform(riverSeparation / 8, riverSeparation / 2))
    shoulderGrade   = (args.shoulder_grade / 100 if args.shoulder_grade is not None else
                       csmap.jr.uniform(0.30, 0.55))
    peakGrade       = (args.peak_grade / 100 if args.peak_grade is not None else
                       csmap.jr.uniform(0.45, 0.80))
    
    print( "        SeaShore FloodPln FootHill MidHill  Shoulder    Peak")
    print(f" Width:          {floodPlainWidth:8.2f} {footHillWidth:8.2f} {midHillWidth:8.2f} {shoulderWidth:8.2f}")
    print(f" Grade: {100*shoreGrade:7.1f}% {100*floodPlainGrade:7.1f}% {100*footHillGrade:7.1f}% {100*midHillGrade:7.1f}% {100*shoulderGrade:7.1f}% {100*peakGrade:7.1f}%")

    ## If this is not None, a river is generated from the nearest land point.
    ## All other rivers attempt to flow to this river.
    focalPossibilities = tuple((pID for pID, pPoint in csmap.enumerate_points() if csmap.level(pID) is not None and 4000 <= pPoint[0] <= 14000 and 4000 <= pPoint[1] <= 14000))
    forceRiver=csmap.jr.choice(focalPossibilities)
    
    ## Next, create shorelines.  There are a lot of ways to do this.  We will
    ## create base boundaries by creating polygons, generating koch curves, and
    ## using those as fill boundaries.

    ## For now we will generate a single path going clockwise around the landmass.
    ## There are four possible in paths:
    pathIn = list()
    pathIn.append( (((csmap.jr.uniform(9000, csmap.fillRect[2]), csmap.fillRect[1]), None),
                    ((csmap.jr.uniform(9000, 14000), 4000), None),
                    ((14000, 4000), None),
                    ((14000, 14000), None)))
    pathIn.append( (((csmap.fillRect[2], csmap.jr.uniform(csmap.fillRect[1], 9000)), None),
                    ((14000, csmap.jr.uniform(4000, 9000)), None),
                    ((14000, 14000), None)))
    pathIn.append( (((csmap.fillRect[2], csmap.jr.uniform(9000, csmap.fillRect[3])), None),
                    ((14000, csmap.jr.uniform(9000, 14000)), None),
                    ((14000, 14000), None)))
    pathIn.append( (((csmap.jr.uniform(14000, csmap.fillRect[2]), csmap.fillRect[3]), None),
                    ((csmap.jr.uniform(9000, 14000), 14000), False)))

    ## And there are four possible out paths
    pathOut = list()
    pathOut.append( (((csmap.jr.uniform(4000, 9000), 14000), False),
                     ((csmap.jr.uniform(csmap.fillRect[0], 4000), csmap.fillRect[3]), False)))
    pathOut.append( (((4000, 14000), False),
                     ((4000, csmap.jr.uniform(9000, 14000)), None),
                     ((csmap.fillRect[0], csmap.jr.uniform(9000, csmap.fillRect[3])), None)))
    pathOut.append( (((4000, 14000), False),
                     ((4000, csmap.jr.uniform(4000, 9000)), None),
                     ((csmap.fillRect[0], csmap.jr.uniform(csmap.fillRect[1], 9000)), None)))
    pathOut.append( (((4000, 14000), False),
                     ((4000, 4000), None),
                     ((csmap.jr.uniform(4000, 9000), 4000), None),
                     ((csmap.jr.uniform(csmap.fillRect[0], 9000), csmap.fillRect[1]), None)))
                   
    path = csmap.jr.choice(pathIn) + csmap.jr.choice(pathOut)
    #print(path)
    
    kp = list()
    for aleph, beth in zip(path, path[1:]):
        p0, dummy = aleph
        p1, leanLeft = beth
        kp.extend(csmap.jr.koch2_path(p0, p1, csmap.separate * csmap.separate / 100.0,
                                      #fixedR=1/4,
                                      canSkew=True,
                                      leanLeft = leanLeft))
    
    kp.append(path[-1][0])

    csmap.set_simplex_sea(kp)
    for pID, point in csmap.enumerate_points():
        if(8000 <= point[0] <= 10000 and
           16000 <= point[1]):
            csmap.set_fill_sea(pID)

    csmap.vp.reset_grid_sel()

    view = csmap.draw_levels(maxLevel=2)
    csmap.overlay_shore_edges(view, kp)
    csmap.overlay_parcels(view)
    csmap.quickview(view, fname="out/cs.edges.png", keep=True)
    
    csmap.vp.reset_grid_sel()
    
    csmap.levelize()

    if(forceRiver is not None and csmap.jr.randint(0,1) == 1):
        canRiver = tuple((pLevel is not None and pLevel is not False and
                          pLevel > riverSeparation / csmap.separate)
                         for pID, pLevel in csmap.enumerate_levels())
        
        csmap.add_river_source(forceRiver)
        csmap.levelize()
    else:
        canRiver = None
    
    if(0):
        view = csmap.draw_levels(maxLevel=csmap.max_level())
        csmap.overlay_parcels(view)
        csmap.quickview(view, fname="out/cs.shore.png", keep=True)
        
    ## Choose a random point within the center 3x3 as a good focus for river generation
    focusPoint = (csmap.jr.uniform(6000, 12000),
                   csmap.jr.uniform(6000, 12000))
    print(f"Focusing rivers toward {focusPoint}")
    selectWeights = csmap.gen_select_weights(focusPoint)

    
    for minRiverLength, viewStr, allowedIDs in ((riverSeparation * 3, "major", canRiver),
                                                (riverSeparation * 2, "minor", None),
                                                (riverSeparation * 1, "revis", None)):
        maxLevel = csmap.extend_rivers(meterStr=viewStr,
                                       riverSeparation=riverSeparation,
                                       riverSegmentLength=riverSegmentLength,
                                       minRiverLength=minRiverLength,
                                       maxIterations=600,
                                       selectWeights=selectWeights,
                                       allowedByID=allowedIDs)
        view = csmap.draw_levels(maxLevel=maxLevel)
        csmap.overlay_parcels(view)
        csmap.quickview(view, fname=f"out/cs.{viewStr}.png", keep=True)


    ## Our drainage information will be based on the unwrinkled river paths
    drains = csmap.gen_drain_levels()

    ## There are some problems with river generations that I haven't worked
    ## out.  Let's repair the drain information for now.
    for pID, pLevel in csmap.enumerate_levels():
        if pLevel is not None and pLevel is not False and pLevel == 0:
            for qID in csmap.neighbors(pID):
                qLevel = csmap.level(qID)
                if(qLevel is None or qLevel is not False and qLevel < 0):
                    break
            else:
                print(f"Warning!  Orphaned source!")
        
        if pID in drains:
            if pLevel is None or pLevel is False or pLevel > 0:
                print(f"Warning!  Drain point has a non-river level of {pLevel!r}")
        else:
            if pLevel is not None and pLevel is not False and pLevel <= 0:
                ## We know!  We know!  I wish I knew why this was happening!
                ## Stop yelling, just fix it!
                
                # print(f"Warning!  River point with level {pLevel!r} is not in the drain list")
                drains[pID] = 1
        
    print(f"Max draining:  {max(drains.values())}")

    ## Finally, make some wrinkly bits to represent unseen creek drainage
    maxLevel = csmap.extend_rivers(meterStr="wrink",
                                   riverSeparation=riverSeparation-wrinkleMeanLength-wrinkleDevLength,
                                   riverSegmentLength=wrinkleMeanLength,
                                   riverSegmentVar=wrinkleDevLength,
                                   seaShoreOffsetMin=floodPlainWidth / 2,
                                   seaShoreOffsetMax=floodPlainWidth + footHillWidth + midHillWidth,
                                   riverShoreOffsetMin=0,
                                   riverShoreOffsetMax=midHillWidth + shoulderWidth,
                                   riverShoreSqueeze=floodPlainWidth + footHillWidth,
                                   maxIterations=600,
                                   retarget=False)

    view = csmap.draw_levels(maxLevel=maxLevel)
    csmap.overlay_parcels(view)
    csmap.quickview(view, fname="out/cs.wrink.png", keep=True)


    slopeFn = csmap.make_slope_fn(((floodPlainWidth, floodPlainGrade),
                                   (floodPlainWidth + footHillWidth, footHillGrade),
                                   (floodPlainWidth + footHillWidth + midHillWidth, midHillGrade),
                                   (floodPlainWidth + footHillWidth + midHillWidth + shoulderWidth, shoulderGrade)),
                                  drainMap=drains,
                                  sourceGrade=riverSourceGrade,
                                  riverMinGrade=riverMinGrade,
                                  wrinkGrade=wrinkleGrade,
                                  shoreGrade=shoreGrade,
                                  peakGrade=peakGrade);

    csmap.gen_heights_really_hard(slopeFn)
    csmap.punch_rivers(drainMap=drains, meanDepth=meanDepth)

    print(f"Land runs from {csmap._lowest:.2f}m to {csmap._highest:.2f}m")

    view = csmap.draw_heights(drainMap=drains)
    csmap.overlay_parcels(view)
    csmap.quickview(view, fname="out/cs.heights.png", keep=True)
    
    view = csmap.draw_cs_heightmap()
    csmap.quickview(view, fname='out/cs.heightmap.png', keep=True)

    if(args.show_stats):
        levelstats = csmap.level_stats(combineRivers=True)

        def fmtlevel(level):
            return 'Sea' if level is None else f'{level:3d}'

        for level, bits in sorted(levelstats.items(), key=lambda e:-math.inf if e[0] is None else e[0]):
            cnt, n, m, x, d = bits['height']
            print(f'{fmtlevel(level)} : {cnt:5d} {n:6.1f} < {m:6.1f} < {x:6.1f} | {d:6.2f}')

            for tgt, bobs in sorted(bits['slope'].items(), key=lambda e:-math.inf if e[0] is None else e[0]):
                cnt, n, m, x, d = bobs
                ng = n / math.sqrt(1 + n * n)
                mg = m / math.sqrt(1 + m * m)
                xg = x / math.sqrt(1 + x * x)
                dg = d / math.sqrt(1 + d * d)

                print(f'      * {fmtlevel(tgt)} : {cnt:5d} {ng:6.2f} < {mg:6.2f} < {xg:6.2f} | {dg:6.2f}')
