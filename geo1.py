## fiona
'''
Fiona is a minimalist python package for reading (and writing) vector data in python. Fiona provides python objects (e.g. a dictionary for each record) to geospatial data in various formats.

More pythonic:
OGR → Fiona
GDAL → Rasterio

'''

import os
import fiona # Fiona requires Python versions 2.7 or 3.4+ and GDAL version 1.11-2.4. GDAL version 3 is not yet supported.
from matplotlib import pyplot as plt
# %matplotlib inline

os.chdir(r'C:\Users\James\OneDrive\PythonFiles\SciPy-Tutorial-2015\examples')

input_file = os.path.join(os.path.abspath('..'), 'examples', 'nybb_15b', 'nybb.shp')
if os.path.exists(input_file):
    print('Input file:', input_file)
else:
    print('Please download the tutorial data or fix the path!')






with fiona.open(input_file,'r') as src:
    # f = src.next() # next() derprecated
    f = next(src)
    # python syntax: next(iterator, default) default (optional) - this value is returned if the iterator is exhausted (no items left)

    print(len(src))
    print(f.keys())
    print(f['type'])
    print(f['id'])
    print(f['geometry']['type'])
    print(f['geometry']['coordinates'])
    print(f['properties'])



## shapely
'''
Shapely is a python library for geometric operations using the GEOS library.

Shapely can perform:

geometry validation
geometry creation (e.g. collections)
geometry operations
'''
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import GeometryCollection

line = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
dilated = line.buffer(0.5) # polygon
eroded = dilated.buffer(-0.3)

coords = np.array(line.coords)
plt.plot(coords[:,0],coords[:,1])
plt.show()

# trick
x,y = zip(*line.coords)
plt.plot(x,y)
plt.show()

# plot
plt.plot(line.xy[0],line.xy[1])
plt.plot(dilated.exterior.xy[0],dilated.exterior.xy[1])
plt.plot(eroded.exterior.xy[0],eroded.exterior.xy[1])
plt.show()



from shapely.geometry import Point, LineString, Polygon, MultiPoint, GeometryCollection

xs = [0, 1, 2, 0, 1, 2]
ys = [0, 0, 0, 1, 1, 1]
line = LineString([(x,y) for x,y in zip(xs,ys)])

poly = Polygon(line)
poly.is_valid # False, crosses itself

square = Polygon(LineString([Point(0,0),Point(1,0),Point(1,1),Point(0,1)]))
square.is_valid


plt.plot(square.exterior.xy[0],square.exterior.xy[1])
a = plt.gca()
a.set_aspect('equal')
plt.show()


## pyproj
'''
EPSG:4326 latitude, longitude in WGS-84 coordinate system
EPSG:900913 and EPSG:3857 Google spherical Mercator
ESRI:102718 NAD 1983 StatePlane New York Long Island FIPS 3104 Feet
'''

from pyproj import Proj
from pyproj import transform

p = Proj(init='epsg:3857')
p.srs

# Transforming from longitude, latitude to map coordinates is a call on a Proj instance that we have initiallized by its EPSG code:
epsg32618 = Proj(init='epsg:32618')
lat, lon = 40.78, -73.97
x, y = epsg32618(lon, lat) # Note the order of lon, lat
print(x, y)

# We can initialize a UTM projection by its zone. Zone 18 is equivalent to EPSG above, so we should ge the same result:
epsg32618(x, y, inverse=True)


# We can initialize a UTM projection by its zone. Zone 18 is equivalent to EPSG above, so we should ge the same result:
utm18 = Proj(proj="utm", zone="18")
print(utm18(lon, lat))


# A string like "+proj=utm +zone=18" can be used as well (this is compatible with PROJ.4 command line arguments).
utm18 = Proj("+proj=utm +zone=18")
print(utm18(lon, lat))


# We can do datum transformations as well as map projections. To create a projection in the NAD27 datum:
nad27 = Proj(proj="utm", zone="18", ellps="clrk66", datum="NAD27")
print(nad27(lon, lat))
# a few meters off wrt WGS84, so be careful with old maps


# We could also do this transfromation between the two reference frames directly:
old_x, old_y = transform(utm18, nad27, x, y)
print(old_x, old_y)


# Libraries including Fiona, rasterio, and GeoPandas use a lightweight python dictionary of parameters that can be passed to the Proj constructor when it is needed for a transformation. Here we use the explicit parameters equivalent to EPSG:2263 (NY state plane Long Island, US feet):
crs = {'lon_0': -74, 'datum': 'NAD83', 'y_0': 0, 'no_defs': True, 'proj': 'lcc',
       'x_0': 300000, 'units': 'us-ft', 'lat_2': 41.03333333333333,
       'lat_1': 40.66666666666666, 'lat_0': 40.16666666666666}
nyc = Proj(**crs)
print(nyc(lon, lat))



# Transform the locations of the Texas state capitol (97.74035° W, 30.27467° N) and the AT&T Center (97.74034° W, 30.28240° N) to UTM zone 14 and calculate the distance between them in meters.
lat1,lon1 = 30.27467, -97.74035
lat2, lon2 = 30.28240, -97.74034

utm14 = Proj(proj='utm',zone='14')
print(utm14(lon1,lat1))
print(utm14(lon2,lat2))



## Rasterio

import rasterio
import os, inspect
loc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(loc)

with rasterio.open('map_shot.png') as f:
    img = f.read(1)
plt.imshow(img, cmap='gray')
plt.show()

##

# http://radar.weather.gov/Conus/RadarImg/latest.gif


# Unfortunately, the GIF image does not have geographic metadata, so we will need to set the input projection. We get the transform parameters from the world file at http://radar.weather.gov/Conus/RadarImg/latest_radaronly.gfw
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pyproj import Proj
import requests
import imageio
import cv2 as cv
from matplotlib import pyplot as plt

src_crs = {'init': 'EPSG:4326'}
west = -127.620375523875420
north = 50.406626367301044
dx = 0.017971305190311
dy = -dx
# Desired width of the output image (height will be calculated):
width = 1600


# Now we need to set the output projection. We'll use a Lambert Equal-Area projection, specifically EPSG 2163, which is used by the National Map to display the continental US.

# Here's an example that displays average annual precipication for the CONUS:
# Image(url='http://nationalmap.gov/small_scale/printable/images/preview/precip/pageprecip_us3.gif')

# We'll use pyproj to transform the corners of the radar image to get the bounds to use for our output grid. First we'll compute the south and east boundaries of the map in lat/lon space. To do that, we'll open the remote file with rasterio to get the image size.

# Next, we'll compute the corners in projected coordinates.

image_url = 'http://radar.weather.gov/Conus/RadarImg/latest.gif'
image_location = os.path.join(os.getcwd(),'radar_img.gif')

r = requests.get(image_url)
img = r.content
with open('radar_img.gif','wb') as f:
    f.write(img)
img_gif = imageio.imread('radar_img.gif')
cv.imwrite('radar_img2.png',img_gif)


# with rasterio.drivers():
with rasterio.open(image_location) as src:
    south = north + src.height * dy
    east = west + src.width * dx
    src_transform = rasterio.transform.from_bounds(west, south, east, north, src.width, src.height)

dst_crs = {'init': 'EPSG:2163'}
# us_equal_area = Proj(**dst_crs)
us_equal_area = Proj(init='epsg:2163')
left, bottom = us_equal_area(west, south)
right, _ = us_equal_area(east, south)
_, top = us_equal_area(east, north)
height = width * (top - bottom) / (right - left)
dst_transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)


# We'll initialize a NumPy array of bytes to transform the image data into.

dst_shape = (round(height), width)
destination = np.zeros(dst_shape, np.uint8)


# Now for the actual transformation. We open the radar file with rasterio, and use rasterio.warp.reproject to transform it to our new coordinate system. It's important to use resampling=RESAMPLING.nearest in this case, because we don't want to interpolate values from the GIF image. For continuous data, other resampling methods may be appropriate.

# with rasterio.drivers():
with rasterio.open(image_location) as src:
    data = src.read(1) # starts with band 1
    cmap = src.colormap(1)

reproject(data, destination,
            src_transform=src_transform, src_crs=src_crs,
            dst_crs=dst_crs, dst_transform=dst_transform,
            resampling=Resampling.nearest)


# Now the result is in a NumPy array. The values in the array are the pixel values from the GIF image, which will not make sense without that file's colormap.

# We'll use rasterio to write the output to a new local GIF file.

with rasterio.open('warped.gif', 'w', driver='GIF',
                   width=width, height=height,
                   count=1, dtype=np.uint8) as dst:
    dst.write_band(1, destination)
    dst.write_colormap(1, cmap)



## Rasterize

from rasterio.features import rasterize
mask = rasterize([poly], transform=src.transform, out_shape=src.shape)



# Rasterio and Cartopy

"""
Plot a raster image with Cartopy axes
Kelsey Jordahl
SciPy tutorial 2015
"""

import os
import rasterio
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

utm18n = ccrs.UTM(18)
ax = plt.axes(projection=utm18n)
plt.title('UTM zone 18N')


here = os.path.dirname(os.path.abspath('__file__'))
data_dir = os.path.join(here, '..', 'data')
raster_file = os.path.join(data_dir, 'manhattan.tif')


with rasterio.open(raster_file) as src:
    left, bottom, right, top = src.bounds
    ax.imshow(src.read(1), origin='upper',
              extent=(left, right, bottom, top), cmap='gray')
    x = [left, right, right, left, left]
    y = [bottom, bottom, top, top, bottom]
    ax.coastlines(resolution='10m', linewidth=4, color='red')
    ax.gridlines(linewidth=2, color='lightblue', alpha=0.5, linestyle='--')

plt.savefig('rasterio_cartopy.png', dpi=300)
plt.show()




"""
Plot a raster image with Cartopy axes on a Mercator map.
This example reprojects the UTM image to Mercator.
Kelsey Jordahl
SciPy tutorial 2015
"""

import os
import rasterio
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

utm18n = ccrs.UTM(18)
merc = ccrs.Mercator()
ax = plt.axes(projection=merc)


here = os.path.dirname(os.path.abspath('__file__'))
data_dir = os.path.join(here, '..', 'data')
raster_file = os.path.join(data_dir, 'manhattan.tif')


with rasterio.open(raster_file) as src:
    left, bottom, right, top = src.bounds
    ax.set_extent((left, right, bottom, top), utm18n)
    ax.imshow(src.read(1), origin='upper', transform=utm18n,
              extent=(left, right, bottom, top), cmap='gray',
              interpolation='nearest')
    x = [left, right, right, left, left]
    y = [bottom, bottom, top, top, bottom]
    ax.coastlines(resolution='10m', linewidth=4, color='red')
    ax.gridlines(linewidth=2, color='lightblue', alpha=0.5, linestyle='--')

plt.savefig('rasterio_cartopy.png', dpi=300)
plt.show()










