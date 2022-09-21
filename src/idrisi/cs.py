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

def quickview(view, fname="cs.png", keep=False):
    view.save(fname)
    proc = subprocess.Popen(("display", fname))
    if not keep:
        proc.wait()
        os.remove(fname)
    return

def gen_select_weights(hmap, centerPoint):
    sw = list()
    for pID, point in hmap.enumerate_points():
        delPoint = (point[0] - centerPoint[0],
                    point[1] - centerPoint[1])
        sw.append(delPoint[0] * delPoint[0] + delPoint[1] * delPoint[1])

    swmax = max(sw)
    for pID, weight in enumerate(sw):
        sw[pID] = swmax - weight + 1

    return sw

def river_extension(hmap, *, jr, vp, separate,
                    riverSegmentLength,
                    riverSegmentVar=0,
                    minRiverLength = None,
                    seaShoreOffsetMin = 0, seaShoreOffsetMax = 0,
                    riverShoreOffsetMin = 0, riverShoreOffsetMax = 0,
                    riverShoreSqueeze = 0,
                    selectWeights = None,
                    viewStr, maxIterations=None, retarget=True):
    riverClipLevel  = None if minRiverLength is None else max(int(minRiverLength / separate), 1)
    seaShoreMin     = max(int(seaShoreOffsetMin / separate), 1)
    seaShoreMax     = max(int(seaShoreOffsetMax / separate), 1)
    riverShoreMin   = max(int(riverShoreOffsetMin / separate), 1)
    riverShoreMax   = max(int(riverShoreOffsetMax / separate), 1)
    
    hmap.levelize()

    rsln = max(1,int((riverSegmentLength - riverSegmentVar) / separate))
    rslx = max(1,int((riverSegmentLength + riverSegmentVar) / separate))
    rsla = (rsln + rslx) / 2

    probableIterations = math.ceil(sum(1 for pID, pLevel in hmap.enumerate_levels() if pLevel is not None and pLevel is not False and rsln <= pLevel and (retarget or pLevel <= rslx)) / rsla / rsla)
    if maxIterations is not None and maxIterations < probableIterations:
        probableIterations = maxIterations
    
    targets = list(pID for pID, pLevel in hmap.enumerate_levels() if pLevel is not None and pLevel is not False and rsln <= pLevel <= rslx)
    iteration = 0
    
    meter = tqdm.tqdm(total=probableIterations, desc=viewStr, leave=True)
    
    while(targets and (maxIterations is None or iteration < maxIterations)):
        iteration += 1
        #print (f"{viewStr}{iteration}{mStr}")

        if(selectWeights is None):
            tID = jr.choice(targets)
        else:
            localWeights = tuple(selectWeights[pID] for pID in targets)
            tID = jr.choices(targets, localWeights)[0]
        
        hmap.add_river_source(tID)
        hmap.levelize()
        if retarget:
            targets = list(pID for pID, pLevel in hmap.enumerate_levels() if pLevel is not None and pLevel is not False and rsln <= pLevel <= rslx)
        else:
            targets = list(pID for pID in targets if rsln <= hmap.level(pID) <= rslx)

        meter.update()
    
    if(riverClipLevel is not None):
        pass
        hmap.remove_river_stubs(riverClipLevel)

    minLevel = hmap.min_level()
    squeezeLevels = int(riverShoreSqueeze / separate)
    if squeezeLevels > 0:
        wtFn = jutil.make_linear_interp(minLevel, 0)
        squeezeFn = lambda i: int(wtFn(i)[1] * squeezeLevels + 0.5)
    else:
        squeezeFn = lambda i: 0
    
    
    hmap.levelize(seaShoreMin=seaShoreMin, seaShoreMax=seaShoreMax,
                  riverShoreMin=riverShoreMin, riverShoreMax=riverShoreMax, riverLiftFn=squeezeFn)

    maxLevel = hmap.max_level()
    meter.close()
    
    ## DRAW MAJOR RIVER LEVELS
    view = PIL.Image.new('RGB', vp.view_size())
    hmap.draw_edges(view, grid2view_fn=vp.grid2view,
                    edge_color_fn = lambda pID, qID: (hmap.level_color(pID, maxLevel=maxLevel),
                                                      hmap.level_color(qID, maxLevel=maxLevel)))
    quickview(view, fname=f"out/cs.{viewStr}.png", keep=True)
    ### END DRAW MAJOR RIVER LEVELS

    for pID, pLevel in hmap.enumerate_levels():
        if pLevel is None or pLevel > 0:
            continue

        above = list()
        at = list()
        below = list()

        for qID in hmap.neighbors(pID):
            qLevel = hmap.level(qID)
            if qLevel is None or qLevel < pLevel:
                below.append(qLevel)
            elif qLevel > pLevel:
                above.append(qLevel)
            else:
                at.append(qLevel)

        if((pLevel != 0 and not above) or not below):
            print(f"{pLevel} => A:{above!r} @:{at!r} B:{below!r}")
            raise(RuntimeError("A river is shady!"))
    
    return maxLevel

if __name__ == '__main__':
    
    jr = jrandom.JRandom()
    vp = jutil.Viewport(gridSize = (18000, 18000),
                        viewSize = (1200, 1200))
    separate = 37
    #separate = 53
    #separate = 101
    #separate = 233
    
    # Create land beyond the outer rim by another 2km on each side
    vp.set_grid_sel((-2000, -2000), (20000, 20000))
    
    points = list(jr.punctillate_rect(pMin = vp.grid_sel_min(),
                                      pMax = vp.grid_sel_max(),
                                      distsq = separate * separate))
    hmap = heightmap.HeightMapper(points, jr=jr)

    hmap.forbid_long_edges(10*separate)

    for pID in range(hmap.point_count()):
        qCnt = sum(1 for qID in hmap.neighbors(pID))
        if(qCnt < 2):
            raise(RuntimeError(f"Somehow {pID=} has {qCnt} neighbors!"))
    
    pathIn = ((((14000, vp.grid_sel_max()[1]), None),),
              (((vp.grid_sel_max()[0], 14000), None),),
              (((vp.grid_sel_max()[0], 4000), None),
               ((14000, 4000), False)),
              (((14000, vp.grid_sel_min()[1]), None),
               ((14000, 4000), False)))

    pathMid = (((14000, 14000), False),)

    pathOut = ((((4000, 14000), None),  ((4000, vp.grid_sel_max()[1]), None)),
               (((4000, 14000), None),  ((vp.grid_sel_min()[0], 14000), None)),
               (((4000, 14000), False), ((4000, 4000), None), ((vp.grid_sel_min()[0], 4000), None)),
               (((4000, 14000), False), ((4000, 4000), None), ((4000, vp.grid_sel_min()[1]), None)))
               
    path = jr.choice(pathIn) + pathMid + jr.choice(pathOut)
    print(path)

    kp = list()
    for aleph, beth in zip(path, path[1:]):
        p0, leanLeft = aleph
        p1, dummy = beth
        kp.extend(jr.koch2_path(p0, p1, separate * separate / 100.0,
                                #fixedR=1/4,
                                canSkew=True,
                                leanLeft = leanLeft))

    kp.append(path[-1][0])

    hmap.set_simplex_sea(kp)
    for pID, point in hmap.enumerate_points():
        if(8000 <= point[0] <= 10000 and
           16000 <= point[1]):
            hmap.set_fill_sea(pID)

    ### DRAW SHORE EDGES
    view = PIL.Image.new('RGB', vp.view_size())
    hmap.draw_edges(view, grid2view_fn=vp.grid2view,
                    edge_color_fn = lambda pID, qID: (hmap.level_color(pID, maxLevel=2),
                                                      hmap.level_color(qID, maxLevel=2)))
    for kPoint in kp:
        kXY = tuple(int(p) for p in vp.grid2view(kPoint))
        if(0 <= kXY[0] < vp.view_size()[0] and 0 <= kXY[1] < vp.view_size()[1]):
            view.putpixel(kXY, levelmap.LevelMapper._errorColor)
        
    quickview(view, fname="out/cs.edges.png", keep=True)
    ### END DRAW SHORE EDGES
    
    vp.reset_grid_sel()
    
    hmap.levelize()
    maxshorelevel = hmap.max_level()
    
    ### DRAW SHORE LEVELS
    view = PIL.Image.new('RGB', vp.view_size())
    hmap.draw_edges(view, grid2view_fn=vp.grid2view,
                    edge_color_fn = lambda pID, qID: (hmap.level_color(pID, maxLevel=maxshorelevel),
                                                      hmap.level_color(qID, maxLevel=maxshorelevel)))

    quickview(view, fname="out/cs.shore.png", keep=True)
    ### END DRAW SHORE LEVELS
    
    #vp.reset_grid_sel()
    selectWeights = gen_select_weights(hmap, (9000, 13000))

    maxMajorLevel=river_extension(hmap, jr=jr, vp=vp, separate=separate, viewStr="major",
                                  riverSegmentLength=1100, minRiverLength=3600,
                                  maxIterations=400, selectWeights=selectWeights)
    maxMinorLevel=river_extension(hmap, jr=jr, vp=vp, separate=separate, viewStr="minor",
                                  riverSegmentLength=1100, minRiverLength=2400,
                                  maxIterations=400, selectWeights=selectWeights)
    maxRevisLevel=river_extension(hmap, jr=jr, vp=vp, separate=separate, viewStr="revis",
                                  riverSegmentLength=1100, minRiverLength=2400,
                                  maxIterations=400, selectWeights=selectWeights)


    drains = hmap.gen_drain_levels()
    
    for pID, pLevel in hmap.enumerate_levels():
        if pLevel is not None and pLevel is not False and pLevel == 0:
            for qID in hmap.neighbors(pID):
                qLevel = hmap.level(qID)
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

    maxWrinkleLevel=river_extension(hmap, jr=jr, vp=vp, separate=separate, viewStr="wrink",
                                    riverSegmentLength=500, riverSegmentVar=150,
                                    seaShoreOffsetMin=0, seaShoreOffsetMax=1200,
                                    riverShoreOffsetMax=1500,
                                    riverShoreSqueeze=500,
                                    maxIterations=400,
                                    retarget=False)
    
    seasinterp = jutil.make_array_interp(len(heightmap.HeightMapper._seacolors), -40, 0)
    landinterp = jutil.make_array_interp(len(heightmap.HeightMapper._landcolors), 0, 984)

    ## ( distance, minSlope, maxSlope )
    #landSlopes = ((20, 0.01, 0.07),
    #              (25, 0.01, 1.20),
    #              (30, 0.02, 0.07),
    #              (190, 0.02, 0.10),
    #              (200, 0.02, 1.20),
    #              (210, 0.03, 0.10),
    #              (700, 0.03, 0.15),
    #              (800, 0.05, 1.20),
    #              (1500, 0.65, 1.20))

    #landSlopes = ((175, 0.001, 0.07),
    #              (225, 0.01, 0.30),
    #              (275, 0.02, 0.07),
    #              (450, 0.02, 0.10),
    #              (600, 0.05, 0.60),
    #              (1500, 0.07, 1.20))
    landSlopes = ((   0, 0.01, 0.02),
                  ( 400, 0.01, 0.10),
                  ( 450, 0.05, 0.20),
                  ( 600, 0.35, 0.50),
                  ( 750, 0.35, 0.80),
                  (1150, 0.50, 1.20))
    
    landValues = []
    slopeIdx = 0
    lwf = lambda a: (0, 1)
    for level in range(maxWrinkleLevel + 1):
        levelDist = level * separate

        while slopeIdx < len(landSlopes) and levelDist > landSlopes[slopeIdx][0]:
            slopeIdx += 1
            if slopeIdx < len(landSlopes):
                print(f'{levelDist=} {landSlopes[slopeIdx-1][0]=} {landSlopes[slopeIdx][0]=}')
                lwf = jutil.make_linear_interp(landSlopes[slopeIdx-1][0],
                                               landSlopes[slopeIdx][0])
            else:
                lwf = lambda a: (1, 0)

        aW, bW = lwf(levelDist)
        landValues.append((landSlopes[max(0, slopeIdx-1)][1]*aW
                           + landSlopes[min(len(landSlopes) - 1, slopeIdx)][1]*bW,
                           landSlopes[max(0, slopeIdx-1)][2]*aW
                           + landSlopes[min(len(landSlopes) - 1, slopeIdx)][2]*bW))
    
    print(f'{landValues=!r}')
    
    def heightcolor(h):
        if(h <= 0):
            aIdx, aWt, bIdx, bWt = seasinterp(h)
            return tuple(a * aWt + b * bWt for a,b in zip(heightmap.HeightMapper._seacolors[aIdx],
                                                          heightmap.HeightMapper._seacolors[bIdx]))
        else:
            aIdx, aWt, bIdx, bWt = landinterp(h)
            return tuple(a * aWt + b * bWt for a,b in zip(heightmap.HeightMapper._landcolors[aIdx],
                                                          heightmap.HeightMapper._landcolors[bIdx]))    
        
    def minmaxslope(hmap, pID):
        pLevel = hmap.level(pID)
        if pLevel is None or pLevel is False:
            return(0.0, 0.55)

        if pLevel <= 0:            
            if pID in drains:
                dmag = 2 ** (drains[pID] - 1)
                return(0.00 / dmag, 0.10 / dmag)

            return(0.00, 0.30)

        return landValues[pLevel]

    def makeMeterWithCB():
        lastTotal = hmap.point_count() * 2
        meter = tqdm.tqdm(total=lastTotal, leave=True)
        def meterCB(nI, nS, nT):
            nonlocal lastTotal
            
            if lastTotal != 2 * nT:
                lastTotal = 2 * nT
                meter.reset(total=lastTotal)

            meter.n = lastTotal-nI-nS
            meter.refresh()

        return meter, meterCB
    
    relax = 1
    while True:
        print(f"I am relaxed:  {relax}")
        
        def mms(*args):
            n, x = minmaxslope(*args)
            return(n / relax, x)

        meter, meterCB = makeMeterWithCB()
        try:
            hmap.gen_heights(mms, sea_height=-40, maxHeight=984, selectRange=(0.5, 0.7), feedbackCB=meterCB, skooshWithin=100)
            print(f"Final relaxation:  {relax}")
            break
        except RuntimeError as e:
            print(e)
            relax *= 1.21
        except KeyboardInterrupt as e:
            print(e)
            relax *= 1.21                 

        meter.close()
        
    if(1):
        widthatsource = 13
        riverbeds = dict()
        for rID, rSize in drains.items():
            wfactor = jr.uniform(0.8, 1.2)
            communityDist = widthatsource * max(1, (2 * rSize - 1)) * wfactor
            expectedCommunitySize = 2 * math.pi * communityDist * communityDist / separate / separate
            community = tuple(hmap.community(rID, widthatsource * max(1, (2 * rSize - 1)) * wfactor))
            if rID not in community:
                community = community + (rID,)

            dfactor = 1.0 / wfactor
            if(len(community) < expectedCommunitySize):
                dfactor *= expectedCommunitySize / len(community)
            
            for bID in community:
                if bID not in riverbeds or riverbeds[bID] < dfactor:
                    riverbeds[bID] = dfactor

        for rID, dfactor in riverbeds.items():
            rHeight = hmap._height[rID]
            rDepth = 18 * dfactor

            hmap._height[rID] -= (rDepth if rHeight >= 0 else
                                  (40 + rHeight) / 40 * rDepth if rHeight >= -40 else
                                  0)
    
    print(f'({hmap._lowest} - {hmap._highest}')

    def edge_color_fn(pID, qID):
        pLevel = hmap._level[pID]
        qLevel = hmap._level[qID]
        pIsSea = pLevel is None
        pIsRiver = pLevel is not None and pLevel <=0 and pID in drains
        qIsSea = qLevel is None
        qIsRiver = qLevel is not None and qLevel <= 0 and qID in drains
        if (pIsRiver and (qIsRiver or qIsSea)) or (pIsSea and qIsRiver):
            return (levelmap.LevelMapper._riverColor,
                    levelmap.LevelMapper._riverColor)
        else:
            return (hmap.height_color(pID),
                    hmap.height_color(qID))

    view = PIL.Image.new('RGB', vp.view_size())
    hmap.draw_edges(view, grid2view_fn=vp.grid2view,
                    edge_color_fn = edge_color_fn)
    quickview(view, fname='out/cs.heights.png', keep=True)

    ## Let's write out the cs heightmap
    vp.set_view_size((1081, 1081))

    def hmc(pID):
        return (min(65535, max(0, int((hmap._height[pID] + 40) * 64))),)
        
        
    view=PIL.Image.new('I;16', vp.view_size())
    hmap.draw_simplices(view, view2grid_fn=vp.view2grid,
                        simplex_color_fn = lambda aID, bID, cID: ( hmc(aID),
                                                                   hmc(bID),
                                                                   hmc(cID)))
    quickview(view, fname='out/cs.heightmap.png', keep=True)
    
