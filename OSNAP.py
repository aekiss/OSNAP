#!/usr/bin/env python
# coding: utf-8

# # OSNAP data extraction
# run with
# qsub -P x77 -q normalbw -l ncpus=27 -l walltime=10:00:00,mem=108GB -l wd -l storage=gdata/hh5+gdata/ik11+gdata/cj50 -V -N OSNAP -- ./OSNAP.py


# In[1]:


import cosima_cookbook as cc
import numpy as np
import pandas as pd
import xarray as xr
import flox  # for faster groupby in xarray with dask
from dask.distributed import Client
from datetime import timedelta, date
import calendar
import os
from collections import OrderedDict
import logging
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)
logging.getLogger('distributed.utils_perf').setLevel(logging.ERROR)


# In[2]:


import climtas.nci
climtas.nci.GadiClient(malloc_trim_threshold='64kib')


# In[3]:


session = cc.database.create_session()


# ## Initialise data structure and define helper functions

# In[4]:


# WARNING! FORGETS ALL LOADED DATA!
data = OrderedDict() # init nested dict of experiments and their analyses


# In[5]:


def addexpt(k, d):
    if k in data:
        print('skipped {}: already exists'.format(k))
    else:
        data[k] = d


# In[6]:


def dictget(d, l):
    """
    Get item in nested dict using a list of keys

    d: nested dict
    l: list of keys
    """
    if len(l) == 1:
        return d[l[0]]
    return dictget(d[l[0]], l[1:])


# In[7]:


def dictknown(d, l):
    """
    Return true if list of keys is valid in nested dict

    d: nested dict
    l: list of keys
    """    
    while len(l)>0 and l[0] in d:
        d = d[l[0]]
        l = l[1:]
    return len(l) == 0


# In[8]:


def dictput(d, l, item):
    """
    Put item in nested dict using a list of keys

    d: nested dict
    l: list of keys
    item: item to be put
    """
    while l[0] in d and len(l)>1:  # transerse existing keys
        d = d[l[0]]
        l = l[1:]
    while len(l)>1:  # add new keys as needed
        d[l[0]] = dict()
        d = d[l[0]]
        l = l[1:]
    d[l[0]] = item
    return


# In[9]:


# convenience functions
def dget(l):
    return dictget(data, l)
def dknown(l):
    return dictknown(data, l)
def dput(l, item):
    return dictput(data, l, item)


# In[10]:


def showdata():
    """
    Display structure of data
    """
    for k, d in data.items():
        print(k)
        for k2, d2 in d.items():
            print('  ', k2)
            try:
                for k3, d3 in d2.items():
                    print('    ', k3)
                    try:
                        for k4, d4 in d3.items():
                            print('      ', k4)
                            try:
                                for k5, d5 in d4.items():
                                    print('        ', k5)
                                    try:
                                        for k6, d6 in d5.items():
                                            print('          ', k6)
                                    except:
                                        pass
                            except:
                                pass
                    except:
                        pass
            except:
                pass


# ## Set experiments, regions, date ranges, variables, frequencies etc
# 1deg_jra55_iaf_omip2_cycle6
# 
# 1deg_jra55_iaf_omip2_cycle6_jra55v150_extension
# 
# 025deg_jra55_iaf_omip2_cycle6
# 
# 025deg_jra55_iaf_omip2_cycle6_jra55v150_extension
# 
# 01deg_jra55v140_iaf_cycle4
# 
# 01deg_jra55v140_iaf_cycle4_jra55v150_extension

# In[11]:


addexpt('1', {'model':'access-om2-025',
              'expts': ['1deg_jra55_iaf_omip2_cycle6',
                        '1deg_jra55_iaf_omip2_cycle6_jra55v150_extension'],
              'gridpaths': ['/g/data/ik11/grids/ocean_grid_10.nc']})


# In[12]:


addexpt('025', {'model':'access-om2-025',
                'expts': ['025deg_jra55_iaf_omip2_cycle6',
                          '025deg_jra55_iaf_omip2_cycle6_jra55v150_extension'],
                'gridpaths': ['/g/data/ik11/grids/ocean_grid_025.nc']})


# In[13]:


addexpt('01', {'model':'access-om2-01',
               'expts': ['01deg_jra55v140_iaf_cycle4',
                         '01deg_jra55v140_iaf_cycle4_jra55v150_extension'],
               'gridpaths': ['/g/data/ik11/grids/ocean_grid_01.nc', 
                             '/g/data/cj50/access-om2/raw-output/access-om2-01/01deg_jra55v140_iaf/output000/ocean/ocean-2d-area_t.nc',
                             '/g/data/cj50/access-om2/raw-output/access-om2-01/01deg_jra55v140_iaf/output000/ocean/ocean-2d-area_u.nc']
              })


# In[14]:


showdata()


# In[15]:


# set date range

tstart = pd.to_datetime('1958', format='%Y')
# tend = pd.to_datetime('2023-01-01', format='%Y-%m-%d')
tend = pd.to_datetime('2023', format='%Y')
# tend = tstart + pd.DateOffset(years=30)
timerange = slice(tstart, tend)
firstyear = pd.to_datetime(tstart).year  # assumes tstart is 1 January!
lastyear = pd.to_datetime(tend).year-1  # assumes tend is 1 January!
yearrange = str(firstyear)+'-'+str(lastyear)
print('yearrange =', yearrange, 'complete years')
print('tstart =', tstart)
print('tend =', tend)


# In[16]:


varnames = [
            'u', 'v',
            'pot_temp',
            'salt',
            'pot_rho_0', 'pot_rho_2',
            'sea_level',
            'net_sfc_heating', 'frazil_3d_int_z',  # heat: https://forum.access-hive.org.au/t/net-surface-heat-and-freshwater-flux-variables/993/2
            'pme_river',  # water
            'sfc_salt_flux_ice', 'sfc_salt_flux_restore',  # salt
            # 'mh_flux',  # sea ice melt
            # 'sfc_hflux_coupler',
            # 'sfc_hflux_from_runoff',
            # 'sfc_hflux_pme',
            # 'net_sfc_heating', 'frazil_3d_int_z',  # Net surface heat flux into ocean is net_sfc_heating + frazil_3d_int_z: https://github.com/COSIMA/access-om2/issues/139#issuecomment-639278547
            # 'swflx',
            # 'lw_heat',
            # 'sens_heat',
            # 'evap_heat',
            # 'fprec_melt_heat',
           ]


# In[17]:


frequencies = ['1 monthly']


# In[18]:


# for the North Atlantic: 70W-0E, 40N-70N
regions = OrderedDict([
    ('NA', {'lon': slice(-70, 0), 'lat': slice(40, 70)}),
])


# ## Calculations

# ### Load data

# In[19]:


def loadalldata(data, regions, freqs, varnames, timerange=timerange, ncfiles=None):
    region = 'global'
    reduction = 'unreduced'

    varnames = list(set(varnames))

    if not isinstance(ncfiles, list):
        ncfiles = [ncfiles]*len(varnames)  # use the same ncfile for all variables

    for expt in data.keys():
        print(expt)
        for freq in freqs:
            kkey = [expt, region, freq, reduction]
            for varname, ncfile in zip(varnames, ncfiles):
                if not dknown(kkey+[varname]):
                    if ncfile is None:
                        print('loading', varname)
                    else:
                        print('loading', varname, 'from', ncfile)
                    dput(kkey+[varname],
                            xr.concat([
                                    cc.querying.getvar(dget([expt, 'expts'])[0], varname, session, frequency=freq, ncfile=ncfile, decode_coords=False, start_time=str(timerange.start)),
                                    cc.querying.getvar(dget([expt, 'expts'])[1], varname, session, frequency=freq, ncfile=ncfile, decode_coords=False, end_time=str(timerange.stop)),
                                                        ], 'time').sel(time=timerange))

        freq = 'static'
    
        grids = [(p, xr.open_dataset(p, chunks='auto')) for p in dget([expt, 'gridpaths'])]
        for k in ['xt_ocean', 'yt_ocean', 'geolon_t', 'geolat_t', 'area_t',
                  'xu_ocean', 'yu_ocean', 'geolon_c', 'geolat_c', 'area_u']:
            kkey = [expt, region, freq, k]
            if not dknown(kkey):
                for (p, g) in grids:
                    try:
                        dput(kkey, g[k])
                        da = g[k]
                        print(k, 'loaded from', p)
                        break
                    except:
                        continue
                try:
                    da = da.rename({'grid_x_T': 'xt_ocean', 'grid_y_T': 'yt_ocean'}) # fix for 01deg
                    da.coords['xt_ocean'] = dget(kkey[0:-1]+['xt_ocean']).values
                    da.coords['yt_ocean'] = dget(kkey[0:-1]+['yt_ocean']).values
                    dput(kkey, da)
                except:
                    pass
                try:
                    da = da.rename({'grid_x_C': 'xu_ocean', 'grid_y_C': 'yu_ocean'}) # fix for 01deg
                    da.coords['xu_ocean'] = dget(kkey[0:-1]+['xu_ocean']).values
                    da.coords['yu_ocean'] = dget(kkey[0:-1]+['yu_ocean']).values
                    dput(kkey, da)
                except:
                    pass


# In[20]:


loadalldata(data, regions, frequencies, varnames, timerange=timerange, ncfiles=None)


# In[21]:


showdata()


# ### Select data for each region

# In[22]:


def slicexy(da, r):
    try:
        da = da.sel(xt_ocean=r['lon'])
    except:
        pass
    try:
        da = da.sel(xu_ocean=r['lon'])
    except:
        pass
    try:
        da = da.sel(yt_ocean=r['lat'])
    except:
        pass
    try:
        da = da.sel(yu_ocean=r['lat'])
    except:
        pass
    return da


# In[23]:


# select data for each region
reduction = 'unreduced'
for expt in data.keys():
    for region, region_data in regions.items():
        for varname, vardata in dget([expt, 'global', 'static']).items():
            if not dknown([expt, region, 'static', varname]):
                dput([expt, region, 'static', varname], slicexy(vardata, region_data))
        for freq in frequencies:
            kkey = [expt, region, freq, reduction]
            for varname, vardata in dget([expt, 'global', freq, reduction]).items():
                if not dknown(kkey+[varname]):
                    d = slicexy(vardata, region_data)
                    d.attrs['subset'] = 'Subset extracted by https://github.com/aekiss/OSNAP/blob/9a2b951/OSNAP.py'
                    dput(kkey+[varname], d)


# ## Save files

# In[24]:


basedir = '/g/data/v45/aek156/notebooks/github/aekiss/OSNAP/data/'


# In[ ]:


reduction = 'unreduced'
for expt in data.keys():
    dpath = os.path.join(basedir, 'access-om2-'+expt)
    os.makedirs(dpath, exist_ok=True )
    for region, region_data in regions.items():
        print(region)
        if region == 'global':
            continue
        for varname, vardata in dget([expt, 'global', 'static']).items():
            fn = '_'.join(['access-om2-'+expt, 'grid', varname])+'.nc'
            fpath = os.path.join(dpath, fn)
            if os.path.exists(fpath) or os.path.exists(fpath+'PARTIAL'):
                print('--- skipped', fn)
            else:
                print('saving', fn)
                dget([expt, region, 'static', varname]).to_netcdf(fpath+'-PARTIAL')
                os.rename(fpath+'-PARTIAL', fpath)
        for freq in frequencies:
            kkey = [expt, region, freq, reduction]
            for varname, vardata in dget(kkey).items():
                fn = '_'.join(['access-om2-'+expt, 'var', varname])+'.nc'
                fpath = os.path.join(dpath, fn)
                if os.path.exists(fpath) or os.path.exists(fpath+'PARTIAL'):
                    print('--- skipped', fn)
                else:
                    print('saving', fn)
                    dget(kkey+[varname]).to_netcdf(fpath+'-PARTIAL')
                    os.rename(fpath+'-PARTIAL', fpath)

