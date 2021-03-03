import matplotlib
import matplotlib.pyplot as plt
import shapely.geometry
import cartopy
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import ascend
from ascend import shape
import numpy as np
import glob

# regions=['0N100W','0N130W','0N15W','0N160E','0N160W','0N30W','0N50E','0N70E','0N88E','10N100W','10N120W','10N140W','10N145E','10N160E','10N170W','10N30W','10N50W','10N60E','10N88E','10S120W','10S140W','10S15W','10S170E','10S170W','10S30W','10S5E','10S60E','10S88E','10S90W','20N135E','20N145W','20N170E','20N170W','20N30W','20N55W','20N65E','20S0E','20S100W','20S105E','20S130W','20S160W','20S30W','20S55E','20S80E','21N115W','29N65W','30N130W','30N145E','30N150W','30N170E','30N170W','30N25W','30N45W','30S100W','30S10E','30S130W','30S15W','30S160W','30S40W','30S60E','30S88E','40N140W','40N150E','40N160W','40N170E','40N25W','40N45W','40N65W','40S0E','40S100E','40S100W','40S130W','40S160W','40S50E','40S50W','50N140W','50N149E','50N160W','50N170E','50N25W','50N45W','50S150E','50S150W','50S30E','50S30W','50S88E','50S90W','60N15W','60N35W','60S0E','60S140E','60S140W','60S70E','60S70W','70N0E','70S160W','70S40W','80N150W']

# point_list = []
# for p in regions:
#     if 'N' in p:
#         psplit = p.split('N')
#         lat = int(psplit[0])
#         if 'W' in psplit[1]:
#             lon = int(psplit[1][:-1])
#         else:
#             lon = 1.*int(psplit[1][:-1])
#     elif 'S' in p:
#         psplit = p.split('S')
#         lat = -1*int(psplit[0])
#         if 'W' in psplit[1]:
#             lon = int(psplit[1][:-1])
#         else:
#             lon = -1*int(psplit[1][:-1])
#     point=(lon,lat)
#     point_list.append(point)
# def regions():
#     reg_lat=np.empty(0, int)
#     reg_lon=np.empty(0, int)
#     with open('ml_lams_latlon_aqua_only.dat', 'r') as filestream:
#         print(filestream)
#         for line in filestream:
#             currentline = line.split(",")
#             reg_lat=np.append(reg_lat,int(currentline[1]))
#             reg_lon=np.append(reg_lon,int(currentline[2]))
#     # End definition of lat/lon for all the regions.


#     for region_number in np.arange(0,len(reg_lat),1):
#             # Loop over the various LAMs (this could be all 98 of them).
#             if reg_lat[region_number]>=0:
#                 lat_letter='N'
#             else:
#                 lat_letter='S'
#             if reg_lon[region_number]<0:
#                 lon_letter='W'
#             else:
#                 lon_letter='E'
#             region=str(np.abs(reg_lat[region_number]))+lat_letter+str(np.abs(reg_lon[region_number]))+lon_letter
#             print('\'{0}\''.format(region),end=',')
# regions()

data = np.loadtxt('ml_lams_latlon_aqua_only.dat',delimiter=',')
point_list = []
for p in data:
    point_list.append(tuple(p[::-1]))

matplotlib.rcParams['figure.figsize'] = (13,8)

# coords = [(0,50)]
# point = shape.create(coords, {'shape':'point'}, 'Point')
point = shape.create(point_list, {'shape':'point'}, 'MultiPoint')

bounds=False
projection=ccrs.PlateCarree()
# projection=ccrs.Robinson()
scale='110m'
facecolor='blue'
shapelist = [point]
ax = shape.plot(shapelist, point_buffer=2., facecolor=facecolor, projection=projection)
if scale == '110m':
    ax.add_feature(cartopy.feature.BORDERS)
    ax.add_feature(cartopy.feature.COASTLINE)
else:
    borders = NaturalEarthFeature(category='cultural',
                                    name='admin_0_countries',
                                    scale=scale, facecolor='none')
    ax.add_feature(borders)

if bounds:
    if hasattr(bounds, '__len__'):
        ax.set_extent(bounds, crs=projection)
    else:
        bounds = np.array([a_shape.data.bounds for a_shape in shapelist])
        x_min = bounds.min(axis=0)[0]
        x_max = bounds.max(axis=0)[2]
        y_min = bounds.min(axis=0)[1]
        y_max = bounds.max(axis=0)[3]
        crs = shapelist[0].coord_system.as_cartopy_projection()
        ax.set_extent([x_min, x_max, y_min, y_max], crs=crs)
else:
    ax.set_global()

plt.show()
plt.close()