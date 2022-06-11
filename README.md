== IDRISI ==

Idrisi is a set of algorithms useful for generating simulated terrains suitable
for importing into games.

An overview of the flow steps:

 * Select a map size in world units and populate it with a random smattering of
   points.  Use Delaunay triangulation to connect the points into a grid of
   triangles.  Forbid any extra long connection lines, as these create highly
   warped terrain.

 * Select a portion of these points to act as the sea.  The generated terrain
   will lay such that virtual rainfall would flow into this virtual sea without
   obstruction.

 * Levelize.  This marks the non-sea grid nodes with a number representing the
   minimum number of steps along the grid edges to reach the sea.

 * Create river segments from non-sea points.  The segments will be constructed
   such that the rivers will follow the levelization paths and flow into the
   sea.  Re-levelize after each segment creation to renumber the grid -- now
   the numbers give a minimum distance to either a sea or a river node.  Repeat
   until a suitable number of rivers have been created.

 * Utilizing the sea, river, and land node levels, construct terrain heights.
   Heights are constructed from lowest levels (sea) to highest levels (largest
   distance from a river node).  With a function converting locations into
   desired elevation and minimum/maximum inter-level slopes, a suitable set of
   land features can be created.

 * Providing a set of conversion factors -- how to go from map units and height
   values to pixel units and greyscale shades -- write out a file suitable for
   reading into your favorite game's map editor.  Viola!

