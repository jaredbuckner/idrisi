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
                      retarget=True):

        riverClipLevel  = None if minRiverLength is None else max(int(minRiverLength / self.separate), 1)
        seaShoreMin     = max(int(seaShoreOffsetMin / self.separate), 1)
        seaShoreMax     = max(int(seaShoreOffsetMax / self.separate), 1)
        riverShoreMin   = max(int(riverShoreOffsetMin / self.separate), 1)
        riverShoreMax   = max(int(riverShoreOffsetMax / self.separate), 1)
        squeezeLevels   = int(riverShoreSqueeze / self.separate)
        
        rsln = max(1,int((riverSegmentLength - riverSegmentVar) / self.separate))
        rslx = max(1,int((riverSegmentLength + riverSegmentVar) / self.separate))
        rsla = (rsln + rslx) / 2
        
        probableIterations = math.ceil(sum(1 for pID, pLevel in self.enumerate_levels() if pLevel is not None and pLevel is not False and rsln <= pLevel and (retarget or pLevel <= rslx)) / rsla / rsla)
        if maxIterations is not None and maxIterations < probableIterations:
            probableIterations = maxIterations
            
        targets = list(pID for pID, pLevel in self.enumerate_levels()
                       if pLevel is not None and pLevel is not False and rsln <= pLevel <= rslx)
        
        iteration = 0
        meter = tqdm.tqdm(total=probableIterations, desc=meterStr, leave=True)
        while(targets and (maxIterations is None or iteration < maxIterations)):
            iteration += 1
            
            if(selectWeights is None):
                tID = self.jr.choice(targets)
            else:
                localWeights = tuple(selectWeights[pID] for pID in targets)
                tID = self.jr.choices(targets, localWeights)[0]

            self.add_river_source(tID)
            self.levelize()
            
            if retarget:
                targets = list(pID for pID, pLevel in self.enumerate_levels()
                               if pLevel is not None and pLevel is not False and rsln <= pLevel <= rslx)
            else:
                targets = list(pID for pID in targets if rsln <= self.level(pID) <= rslx)

            meter.update()

        if(riverClipLevel is not None):
            self.remove_river_stubs(riverClipLevel)

        minLevel = self.min_level()
        
        if squeezeLevels > 0:
            wtFn = jutil.make_linear_interp(minLevel, 0)
            squeezeFn = lambda i: int(wtFn(i)[1] * squeezeLevels + 0.5)
        else:
            squeezeFn = lambda i: 0


        self.levelize(seaShoreMin=seaShoreMin, seaShoreMax=seaShoreMax,
                      riverShoreMin=riverShoreMin, riverShoreMax=riverShoreMax, riverLiftFn=squeezeFn)

        meter.close()
        return self.max_level()

    def make_slope_fn(self, distGradeSeq, *,
                      drainMap,
                      shoreGrade, wrinkGrade, sourceGrade, peakGrade):
        landSlopes = list()
        for distance, grade in distGradeSeq:
            level = int(distance / self.separate + 0.5)
            slope = grade / math.sqrt(1 - grade*grade)
            landSlopes.append((level, slope))

        shoreSlope = shoreGrade / math.sqrt(1 - shoreGrade*shoreGrade)
        wrinkSlope = wrinkGrade / math.sqrt(1 - wrinkGrade*wrinkGrade)
        sourceSlope = sourceGrade / math.sqrt(1 - sourceGrade*sourceGrade)
        peakSlope = peakGrade / math.sqrt(1 - peakGrade*peakGrade)

        def _slope_fn(hmap, pID):
            pLevel = hmap.level(pID)

            if pLevel is None or pLevel is False:
                return(0.0, 2 * shoreSlope)

            if pLevel <= 0:
                if pID in drainMap:
                    dmag = 2 ** drainMap[pID] - 1
                    return(0.0, 2 * sourceSlope / dmag)

                else:
                    return(0.0, 2 * wrinkSlope)

            for landLevel, landSlope in landSlopes:
                if pLevel <= landLevel:
                    return(0.0, 2 *landSlope)

            return(0.0, 2 * peakSlope)

        return(_slope_fn)

    def make_genheight_meter_with_cb(self):
        lastTotal = self.point_count() * 2
        meter = tqdm.tqdm(total=lastTotal, leave=True)
        def meterCB(nI, nS, nT):
            nonlocal lastTotal
            
            if lastTotal != 2 * nT:
                lastTotal = 2 * nT
                meter.reset(total=lastTotal)

            meter.n = lastTotal-nI-nS
            meter.refresh()

        return meter, meterCB

    def gen_heights_really_hard(self, slopeFn):
        relax = 1
        while True:
            print(f"I am relaxed:  {relax}")

            def mms(*args):
                n, x = slopeFn(*args)
                return(n / relax, x)
            
            meter, meterCB = self.make_genheight_meter_with_cb()
            try:
                self.gen_heights(mms, sea_height=-40, maxHeight=984,
                                 selectRange=(0.4, 0.6),
                                 feedbackCB=meterCB, skooshWithin=100)
                
                print(f"Final relaxation:  {relax}")
                break
            except RuntimeError as e:
                print(e)
                relax *= 1.21
            except KeyboardInterrupt as e:
                print(e)
                relax *= 1.21                 

            meter.close()

    
    def punch_rivers(self, *, drainMap, widthatsource=13):
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
            rDepth = 18 * dfactor

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

        
if __name__ == '__main__':
    

    ## First, generate the base grid of points for the map.  A larger separate
    ## value generates a more blocky grid, yet much faster.
    csmap = CS(    
        #separate = 37
        #separate = 53
        #separate = 101
        separate = 233
        )

    print(f"Grid of {csmap.point_count()} points generated with separation={csmap.separate:.2f}.")
    
    ## Before we generate the map, let's plan out the slopes and levels.
    
    ## First, we need the nearest distance that flowing rivers will be allowed.
    ## We probably don't want anything nearer than two large rivers per playing
    ## grid (2kmx2km), and we probably don't want anything less than having to
    ## go a grid away to get to a river.
    riverSeparation = csmap.jr.uniform(750, 2250)
    print(f"Flowing rivers shall be separated by no less than {riverSeparation:.2f}m")

    ## NOTE:  This is really too long for "nice" river segments.  Later on
    ## we'll have to create a segment length and figure out how to make that
    ## work well.
    # riverSegmentLength = csmap.jr.uniform(250,750)
    # print(f"Flowing rivers shall be constructed out of {riverSegmentLength:.2f}m segments")

    ## Rivers have a grade at their source.  They get less steep as they flow
    riverSourceGrade = csmap.jr.uniform(0.1, 0.21)
    print(f"River source grade:  {100*riverSourceGrade:.1f}%")

    ## Beyond the river are short invisible feeder streams.  They make the land more wrinkly.
    wrinkleMeanLength = csmap.jr.uniform(0, riverSeparation / 4)
    wrinkleDevLength = csmap.jr.uniform(0, wrinkleMeanLength)
    wrinkleGrade = riverSourceGrade * csmap.jr.uniform(1, 1.5)
    print(f"Feeder streams will be from {wrinkleMeanLength-wrinkleDevLength:.2f}m to {wrinkleMeanLength+wrinkleDevLength:.2f}m long")
    print(f"Feeder streams have a target grade of {100*wrinkleGrade:.1f}%")

    
    
    ## Width and grade of various land bits
    shoreGrade      = csmap.jr.uniform(0.10, 0.50)
    floodPlainWidth = csmap.jr.uniform(csmap.separate, riverSeparation * 3 / 4)
    floodPlainGrade = csmap.jr.uniform(0.00, 0.02)

    ## Foothill + Midhill = riverSeparation / 2
    footHillWidth, midHillWidth = csmap.jr.rand_sum_to_n(riverSeparation / 4, 2)
    footHillWidth += riverSeparation / 8
    midHillWidth += riverSeparation / 8
    
    footHillGrade   = csmap.jr.uniform(0.01, 0.10)
    midHillGrade    = csmap.jr.uniform(0.05, 0.20)
    shoulderWidth   = csmap.jr.uniform(riverSeparation / 8, riverSeparation / 2)
    shoulderGrade   = csmap.jr.uniform(0.10, 0.50)
    peakGrade       = csmap.jr.uniform(0.30, 0.70)

    print( "        SeaShore FloodPln FootHill MidHill  Shoulder    Peak")
    print(f" Width:          {floodPlainWidth:8.2f} {footHillWidth:8.2f} {midHillWidth:8.2f} {shoulderWidth:8.2f}")
    print(f" Grade: {100*shoreGrade:7.1f}% {100*floodPlainGrade:7.1f}% {100*footHillGrade:7.1f}% {100*midHillGrade:7.1f}% {100*shoulderGrade:7.1f}% {100*peakGrade:7.1f}%")
    
    
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

    if(0):
        view = csmap.draw_levels(maxLevel=csmap.max_level())
        csmap.overlay_parcels(view)
        csmap.quickview(view, fname="out/cs.shore.png", keep=True)

    ## Choose a random point within the center 3x3 as a good focus for river generation
    focusPoint = (csmap.jr.uniform(6000, 12000),
                   csmap.jr.uniform(6000, 12000))
    print(f"Focusing rivers toward {focusPoint}")
    selectWeights = csmap.gen_select_weights(focusPoint)


    for minRiverLength, viewStr in ((riverSeparation * 3, "major"),
                                    (riverSeparation * 2, "minor"),
                                    (riverSeparation * 1, "revis")):
        maxLevel = csmap.extend_rivers(meterStr=viewStr,
                                       riverSegmentLength=riverSeparation,
                                       minRiverLength=minRiverLength,
                                       maxIterations=400,
                                       selectWeights=selectWeights)
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
                                   riverSegmentLength=wrinkleMeanLength,
                                   riverSegmentVar=wrinkleDevLength,
                                   seaShoreOffsetMin=floodPlainWidth / 2,
                                   seaShoreOffsetMax=floodPlainWidth + footHillWidth + midHillWidth/2,
                                   riverShoreOffsetMin=0,
                                   riverShoreOffsetMax=floodPlainWidth + footHillWidth,
                                   riverShoreSqueeze=floodPlainWidth,
                                   maxIterations=400,
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
                                  wrinkGrade=wrinkleGrade,
                                  shoreGrade=shoreGrade,
                                  peakGrade=peakGrade);

    csmap.gen_heights_really_hard(slopeFn)
    csmap.punch_rivers(drainMap=drains)

    print(f"Land runs from {csmap._lowest:.2f}m to {csmap._highest:.2f}m")

    view = csmap.draw_heights(drainMap=drains)
    csmap.overlay_parcels(view)
    csmap.quickview(view, fname="out/cs.heights.png", keep=True)

    view = csmap.draw_cs_heightmap()
    csmap.quickview(view, fname='out/cs.heightmap.png', keep=True)
