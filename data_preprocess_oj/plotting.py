import datetime
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import gc
import iris
import tendencies


def plot_tseries(qfile:str,qincrfile:str,tfile:str,tincrfile:str,subdomain:int):
    """
    Simple plotting for error checking
    """
    qdata = Dataset(qfile)
    qincrdata = Dataset(qincrfile)
    tdata = Dataset(tfile)
    tincrdata = Dataset(tincrfile)
    q = qdata.variables['specific_humidity'][:]
    q = np.ma.masked_where(q < 0.,q)
    qincr = qincrdata.variables['change_over_time_in_specific_humidity_due_to_advection'][:]
    qincr = np.ma.masked_where(qincr == 0., qincr)
    t = tdata.variables['air_temperature'][:]
    t = np.ma.masked_where(t < 0.,t)
    tincr = tincrdata.variables['change_over_time_in_air_temperature_due_to_advection'][:]
    tincr = np.ma.masked_where(tincr == 0., tincr)
    print("Plotting ... ")
    fig, axs = plt.subplots(2,2,figsize=(14, 10))
    ax = axs[0, 0]
    c = ax.pcolor(q.T)
    ax.set_title('Q')
    fig.colorbar(c,ax=ax)

    ax = axs[0, 1]
    c = ax.pcolor(t.T)
    ax.set_title('T')
    fig.colorbar(c,ax=ax)

    ax = axs[1, 0]
    c = ax.pcolor(qincr.T)
    ax.set_title('Q_incr')
    fig.colorbar(c,ax=ax)

    ax = axs[1, 1]
    c = ax.pcolor(tincr.T)
    ax.set_title('T_incr')
    fig.colorbar(c,ax=ax)
    figname="qt_subdomain_{0}.png".format(str(subdomain).zfill(3))
    print("Saving figure {0}".format(figname))
    plt.savefig(figname)
    plt.close(fig)
    # plt.show()
    gc.collect()

def plot_tend_tseries(qfile:str, tfile:str, subdomain:int):
    """
    Simple plotting for error checking
    """
    qdata = Dataset(qfile)
    tdata = Dataset(tfile)
    q = qdata.variables['q_phys'][:]
    # q = np.ma.masked_where(q < 0.,q)
    t = tdata.variables['t_phys'][:]
    # t = np.ma.masked_where(t < 0.,t)
    print("Plotting ... ")
    fig, axs = plt.subplots(2,1,figsize=(14, 10))
    ax = axs[0]
    c = ax.pcolor(q.T)
    ax.set_title('Q Phys')
    fig.colorbar(c,ax=ax)

    ax = axs[1]
    c = ax.pcolor(t.T)
    ax.set_title('T Phys')
    fig.colorbar(c,ax=ax)

    figname="qt_phys_subdomain_{0}.png".format(str(subdomain).zfill(3))
    print("Saving figure {0}".format(figname))
    plt.savefig(figname)
    plt.close(fig)
    # plt.show()
    gc.collect()

def error_check_plot(region):
    # region='10N160E'
    for subdomain in range(6,64):
        qfile = "/project/spice/radiation/ML/CRM/data/u-bj775/{0}/concat_stash_00010/30_days_{0}_km1p5_ra1m_30x30_subdomain_{1}_00010.nc".format(region, str(subdomain).zfill(3))
        qincrfile = "/project/spice/radiation/ML/CRM/data/u-bj775/{0}/concat_stash_12182/30_days_{0}_km1p5_ra1m_30x30_subdomain_{1}_12182.nc".format(region, str(subdomain).zfill(3))
        tfile = "/project/spice/radiation/ML/CRM/data/u-bj775/{0}/concat_stash_16004/30_days_{0}_km1p5_ra1m_30x30_subdomain_{1}_16004.nc".format(region, str(subdomain).zfill(3))
        tincrfile = "/project/spice/radiation/ML/CRM/data/u-bj775/{0}/concat_stash_12181/30_days_{0}_km1p5_ra1m_30x30_subdomain_{1}_12181.nc".format(region, str(subdomain).zfill(3))
        plot_tseries(qfile,qincrfile,tfile,tincrfile,subdomain)

def tendencies_plot(region):
    for subdomain in range(64):
        qfile = "/project/spice/radiation/ML/CRM/data/u-bj775/{0}/tend_q/30_days_{0}_km1p5_ra1m_30x30_subdomain_{1}_q_phys.nc".format(region, str(subdomain).zfill(3))
        tfile = "/project/spice/radiation/ML/CRM/data/u-bj775/{0}/tend_t/30_days_{0}_km1p5_ra1m_30x30_subdomain_{1}_t_phys.nc".format(region, str(subdomain).zfill(3))
        plot_tend_tseries(qfile, tfile, subdomain)

