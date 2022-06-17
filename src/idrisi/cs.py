## Stuff useful for Cities Skylines
import idrisi.levelmap as levelmap
import idrisi.heightmap as heightmap
import idrisi.jrandom as jrandom
import idrisi.jutil as jutil
import math
import os
import PIL.Image
import subprocess

if __name__ == '__main__':
    def quickview(view, fname="cs.png", keep=False):
        view.save(fname)
        proc = subprocess.Popen(("display", fname))
        if not keep:
            proc.wait()
            os.remove(fname)
        return
    
    jr = jrandom.JRandom()
    vp = jutil.Viewport(gridSize = (18000, 18000),
                        viewSize = (1200, 1200))
    separate = 101

    vp.zoom_grid_sel(9/11)  # Create land beyond the outer rim by another 2km on each side
    
    points = list(jr.punctillate_rect(pMin = vp.grid_sel_min(),
                                      pMax = vp.grid_sel_max(),
                                      distsq = separate * separate))
    hmap = heightmap.HeightMapper(points, jr=jr)
    hmap.forbid_long_edges(5 * separate)

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

    vp.reset_grid_sel()

    hmap.levelize()
    levelfromshore = dict(hmap._level)
    maxshorelevel = hmap.max_level()

    ### DRAW SHORE LEVELS
    view = PIL.Image.new('RGB', vp.view_size())
    hmap.draw_edges(view, grid2view_fn=vp.grid2view,
                    edge_color_fn = lambda pID, qID: (hmap.level_color(pID, maxLevel=maxshorelevel),
                                                      hmap.level_color(qID, maxLevel=maxshorelevel)))

    if(0):
        for kPoint in kp:
            kXY = tuple(int(p) for p in vp.grid2view(kPoint))
            if(0 <= kXY[0] < vp.view_size()[0] and 0 <= kXY[1] < vp.view_size()[1]):
                view.putpixel(kXY, levelmap.LevelMapper._errorColor)
        
    quickview(view, fname="out/cs.shore.png", keep=True)
    ### END DRAW SHORE LEVELS

    major_length = 3000
    major_level = max(int(major_length / separate),1)
    influence_length = 250
    influence_level = max(int(influence_length / separate),1)

    targets = list(pID for pID, pLevel in hmap._level.items() if pLevel == major_level)
    while(targets):
        print ("M")
        hmap.add_river_source(jr.choice(targets))
        hmap.levelize()
        targets = list(pID for pID, pLevel in hmap._level.items() if pLevel == major_level)

    hmap.remove_river_stubs(major_level - 1)
    hmap.levelize(ignoreSea=True, maxIterations=maxshorelevel)
    levelfrommajor = dict(hmap._level)

    ## DRAW MAJOR RIVER LEVELS
    view = PIL.Image.new('RGB', vp.view_size())
    hmap.draw_edges(view, grid2view_fn=vp.grid2view,
                    edge_color_fn = lambda pID, qID: (hmap.level_color(pID, maxLevel=influence_level),
                                                      hmap.level_color(qID, maxLevel=influence_level)))
    quickview(view, fname="out/cs.major.png", keep=True)
    ### END DRAW MAJOR RIVER LEVELS

    hmap.levelize()

    minor_length = 1000
    minor_level = max(int(minor_length / separate),1)
    targets = list(pID for pID, pLevel in hmap._level.items() if pLevel == minor_level)
    iterations = 0
    while(targets):
        iterations += 1        
        print(f"m{iterations}")
        hmap.add_river_source(jr.choice(targets))
        hmap.levelize()
        targets = list(pID for pID, pLevel in hmap._level.items() if pLevel == minor_level)

    hmap.remove_river_stubs(minor_level - 1)
    hmap.levelize()
    levelfromminor = dict(hmap._level)
    maxminorlevel = hmap.max_level()
    
    ## DRAW MINOR RIVER LEVELS
    view = PIL.Image.new('RGB', vp.view_size())
    hmap.draw_edges(view, grid2view_fn=vp.grid2view,
                    edge_color_fn = lambda pID, qID: (hmap.level_color(pID, maxLevel=maxminorlevel),
                                                      hmap.level_color(qID, maxLevel=maxminorlevel)))
    quickview(view, fname="out/cs.minor.png", keep=True)
    ### END DRAW MINOR RIVER LEVELS
    
    targetHeight = dict()
    maxTargetHeight = None
    for pID, pLevel in levelfromshore.items():
        if pLevel is None:
            targetHeight[pID] = -40
        else:
            pHeight = 984 * pLevel / maxshorelevel
            pHeight = pHeight * math.sqrt(pHeight / 984)
            qHeight = 0
            qLevel = levelfromminor.get(pID, None)
            if qLevel is not None and qLevel > 0:
                qHeight = 984 * qLevel / maxminorlevel

            height = (pHeight + qHeight) / 2
            if maxTargetHeight is None or maxTargetHeight < height:
                maxTargetHeight = height
                
            targetHeight[pID] = height

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
    
    ## DRAW H LEVELS
    view = PIL.Image.new('RGB', vp.view_size())
    hmap.draw_edges(view, grid2view_fn=vp.grid2view,
                    edge_color_fn = lambda pID, qID: (heightcolor(targetHeight[pID]),
                                                      heightcolor(targetHeight[qID])))
    quickview(view, fname="out/cs.minor.png", keep=True)
    ### END DRAW H LEVELS
    
    def minmaxslope(pID):
        levelmin = levelfromminor.get(pID, None)

        if levelmin is None:
            return (0.0, 2.0)

        if levelmin <= -major_level:
            return (0.001, 0.005)

        if levelmin <= 0:
            return (0.01, 0.03)

        levelmaj = levelfrommajor.get(pID, None)

        if levelmaj is not None and levelmaj < influence_level:
            return (0.01, 0.02)
        
        return (0.02, 0.60)

    hmap.gen_heights(lambda pLevel, pID: (targetHeight[pID], *minmaxslope(pID)), -40)

    widthatsource = 10
    riverbeds = set()
    riversize = hmap.gen_drain_levels()
    for rID, rSize in riversize.items():
        riverbeds.update(hmap.community(rID, widthatsource * rSize).keys())

    for rID in riverbeds:
        rHeight = hmap._height[rID]
        rDepth = 6
        
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
    quickview(view, fname='cs.heights.png', keep=True)

    
