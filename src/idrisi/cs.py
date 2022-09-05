## Stuff useful for Cities Skylines
import idrisi.levelmap as levelmap
import idrisi.heightmap as heightmap
import idrisi.jrandom as jrandom
import idrisi.jutil as jutil
import math
import os
import PIL.Image
import subprocess

def quickview(view, fname="cs.png", keep=False):
    view.save(fname)
    proc = subprocess.Popen(("display", fname))
    if not keep:
        proc.wait()
        os.remove(fname)
    return

def river_extension(hmap, *, jr, vp, separate,
                    riverSegmentLength,
                    minRiverLength = None,
                    seaShoreOffsetMin = 0, seaShoreOffsetMax = 0,
                    riverShoreOffsetMin = 0, riverShoreOffsetMax = 0,
                    viewStr, maxIterations=None):
    riverSearchLevel = max(int(riverSegmentLength / separate), 1)
    riverClipLevel  = None if minRiverLength is None else max(int(minRiverLength / separate), 1)
    seaShoreMin     = max(int(seaShoreOffsetMin / separate), 1)
    seaShoreMax     = max(int(seaShoreOffsetMax / separate), 1)
    riverShoreMin   = max(int(riverShoreOffsetMin / separate), 1)
    riverShoreMax   = max(int(riverShoreOffsetMax / separate), 1)
    
    mStr = '' if maxIterations is None else f' of {maxIterations}'
    hmap.levelize()
    
    targets = list(pID for pID, pLevel in hmap.enumerate_levels() if pLevel == riverSearchLevel)
    iteration = 0
    
    while(targets and (maxIterations is None or iteration < maxIterations)):
        iteration += 1
        print (f"{viewStr}{iteration}{mStr}")
        
        hmap.add_river_source(jr.choice(targets))
        hmap.levelize()
        targets = list(pID for pID, pLevel in hmap.enumerate_levels() if pLevel == riverSearchLevel)

    if(riverClipLevel is not None):
        hmap.remove_river_stubs(riverClipLevel)
        
    hmap.levelize(seaShoreMin=seaShoreMin, seaShoreMax=seaShoreMax,
                  riverShoreMin=riverShoreMin, riverShoreMax=riverShoreMax)
    
    maxLevel = hmap.max_level()
    
    ## DRAW MAJOR RIVER LEVELS
    view = PIL.Image.new('RGB', vp.view_size())
    hmap.draw_edges(view, grid2view_fn=vp.grid2view,
                    edge_color_fn = lambda pID, qID: (hmap.level_color(pID, maxLevel=maxLevel),
                                                      hmap.level_color(qID, maxLevel=maxLevel)))
    quickview(view, fname=f"out/cs.{viewStr}.png", keep=True)
    ### END DRAW MAJOR RIVER LEVELS

    return maxLevel

if __name__ == '__main__':
    
    jr = jrandom.JRandom()
    vp = jutil.Viewport(gridSize = (18000, 18000),
                        viewSize = (1200, 1200))
    separate = 53
    #separate = 533
    
    # Create land beyond the outer rim by another 2km on each side
    vp.set_grid_sel((-2000, -2000), (20000, 20000))
    
    points = list(jr.punctillate_rect(pMin = vp.grid_sel_min(),
                                      pMax = vp.grid_sel_max(),
                                      distsq = separate * separate))
    hmap = heightmap.HeightMapper(points, jr=jr)

    hmap.forbid_long_edges(10*separate)
    
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

    maxMajorLevel=river_extension(hmap, jr=jr, vp=vp, separate=separate, viewStr="major",
                                  riverSegmentLength=1100, minRiverLength=3600,
                                  maxIterations=100)
    maxMinorLevel=river_extension(hmap, jr=jr, vp=vp, separate=separate, viewStr="minor",
                                  riverSegmentLength=1100, minRiverLength=2400,
                                  maxIterations=100)
    maxRevisLevel=river_extension(hmap, jr=jr, vp=vp, separate=separate, viewStr="revis",
                                  riverSegmentLength=1100, minRiverLength=2400,
                                  seaShoreOffsetMin=600, seaShoreOffsetMax=1800,
                                  riverShoreOffsetMax=2400,
                                  maxIterations=100)


    drains = hmap.gen_drain_levels()
    print(f"Max draining:  {max(drains.values())}")
    
    seasinterp = jutil.make_array_interp(len(heightmap.HeightMapper._seacolors), -40, 0)
    landinterp = jutil.make_array_interp(len(heightmap.HeightMapper._landcolors), 0, 984)
    
    def heightcolor(h):
        if(h <= 0):
            aIdx, aWt, bIdx, bWt = seasinterp(h)
            return tuple(a * aWt + b * bWt for a,b in zip(heightmap.HeightMapper._seacolors[aIdx],
                                                          heightmap.HeightMapper._seacolors[bIdx]))
        else:
            aIdx, aWt, bIdx, bWt = landinterp(h)
            return tuple(a * aWt + b * bWt for a,b in zip(heightmap.HeightMapper._landcolors[aIdx],
                                                          heightmap.HeightMapper._landcolors[bIdx]))
    
    #landValues = [(None, 0.01, 0.03),
    #              (None, 0.02, 0.05),
    #              (None, 0.05, 0.20),
    #              (None, 0.10, 0.35),
    #              (None, 0.20, 0.70),
    #              (None, 0.35, 1.20),
    #              (None, 0.55, 1.20),
    #              (None, 0.75, 1.20),
    #              (None, 0.95, 1.20)]
    landValues = [(0.01, 0.55),
                  (0.01, 0.05),
                  (0.05, 0.20),
                  (0.15, 0.35),
                  (0.25, 0.70),
                  (0.35, 1.20),
                  (0.55, 1.20),

                  (0.75, 1.20),
                  (0.75, 1.20),
                  (0.75, 1.20),

                  (0.95, 1.20)
                  ]
    landWeights = jutil.make_array_interp(len(landValues), 1, maxRevisLevel)
    
        
    def minmaxslope(hmap, pID):
        pLevel = hmap.level(pID)
        if pLevel is None or pLevel is False:
            return(0.0, 0.55)

        if pLevel <= 0:            
            if pID in drains:
                dmag = drains[pID]
                return(0.02 / dmag, 0.10 / dmag)

            return(0.02, 0.10)

        aID, aW, bID, bW = landWeights(pLevel)
        return tuple(a * aW + b * bW if a is not None and b is not None else None for a, b in zip(landValues[aID], landValues[bID]))

    relax = 1
    while True:
        print(f"I am relaxed:  {relax}")
        
        def mms(*args):
            n, x = minmaxslope(*args)
            return(n / relax, x)
        
        try:            
            hmap.gen_heights(mms, sea_height=-40, maxHeight=984, selectRange=(0.5, 0.7))
            break
        except RuntimeError as e:
            print(e)
            relax *= 1.5
    
    
    if(1):
        widthatsource = 10
        riverbeds = dict()
        for rID, rSize in drains.items():
            wfactor = jr.uniform(0.5, 1.5)
            dfactor = 1.0 / wfactor
            for bID in (rID, *hmap.community(rID, widthatsource * rSize * wfactor)):
                if bID not in riverbeds or riverbeds[bID] < dfactor:
                    riverbeds[bID] = dfactor

        for rID, dfactor in riverbeds.items():
            rHeight = hmap._height[rID]
            rDepth = 15 * dfactor

            hmap._height[rID] -= (rDepth if rHeight >= 0 else
                                  (40 + rHeight) / 40 * rDepth if rHeight >= -40 else
                                  0)
    
    print(f'({hmap._lowest} - {hmap._highest}')

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
    
