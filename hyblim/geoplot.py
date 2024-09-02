''' Plot geospatial data.

@Author  :   Jakob Schlör 
@Time    :   2022/08/18 10:56:26
@Contact :   jakob.schloer@uni-tuebingen.de
'''

import sys, os, string
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy as ctp

PATH = os.path.dirname(os.path.abspath(__file__))
plt.style.use(PATH + "/../paper.mplstyle")


def loss(loss, n_batches=None, ax=None, yscale='linear', ylabel='mse', **pargs):
    """Plot training and validation loss.

    Args:
    -----
    loss (np.array): (num_batches) Training loss of each batch.
    n_batches (int): Number of batches in the dataset.
    ax (plt.axes): Matplotlib axes, default: None
    y_scale (str): Y scale of plotting.

    Return:
    -------
    ax (plt.axes): Return axes.

    """
    # plot loss
    if ax is None:
        fig, ax = plt.subplots()

    xlabel = '# of batch'
    if n_batches is not None:
        loss = np.average(loss.reshape(-1, n_batches), axis=1)
        xlabel = 'epochs'

    ax.plot(loss, **pargs)

    ax.set_xscale('linear')
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if 'label' in pargs.keys():
        ax.legend()

    return ax


def create_map_plot(ax=None, ctp_projection='PlateCarrree',
                    central_longitude=0,
                    gridlines_kw=dict(draw_labels=True, dms=True, x_inline=False, 
                                      y_inline=False, linewidth=0.0)):
    """Generate cartopy figure for plotting.

    Args:
        ax ([type], optional): [description]. Defaults to None.
        ctp_projection (str, optional): [description]. Defaults to 'PlateCarrree'.
        central_longitude (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: [description]

    Returns:
        ax (plt.axes): Matplotplib axes object.
    """
    if ax is None:
        # set projection
        if ctp_projection == 'Mollweide':
            proj = ctp.crs.Mollweide(central_longitude=central_longitude)
        elif ctp_projection == 'EqualEarth':
            proj = ctp.crs.EqualEarth(central_longitude=central_longitude)
        elif ctp_projection == 'Robinson':
            proj = ctp.crs.Robinson(central_longitude=central_longitude)
        elif ctp_projection == 'PlateCarree':
            proj = ctp.crs.PlateCarree(central_longitude=central_longitude)
        else:
            raise ValueError(
                f'This projection {ctp_projection} is not available yet!')

        fig, ax = plt.subplots()
        ax = plt.axes(projection=proj)

    # axes properties
    ax.coastlines()
    gl = ax.gridlines(**gridlines_kw)
    gl.top_labels = False
    gl.right_labels = False
    ax.add_feature(ctp.feature.RIVERS)
    ax.add_feature(ctp.feature.BORDERS, linestyle=':')
    ax.add_feature(ctp.feature.LAND, facecolor='grey')
    return ax, gl


def plot_map(dmap, ax=None, vmin=None, vmax=None, eps=0.1,   
             cmap='RdBu_r', centercolor=None, bar='discrete', add_bar=True, 
             ctp_projection='PlateCarree', transform=None, central_longitude=0,
             kwargs_pl=None,
             kwargs_cb=dict(orientation='horizontal', shrink=0.8, extend='both'),
             kwargs_gl=dict(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                            linewidth=0.0)
             ):
    """Simple map plotting using xArray.

    Args:
        dmap ([type]): [description]
        central_longitude (int, optional): [description]. Defaults to 0.
        vmin ([type], optional): [description]. Defaults to None.
        vmax ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        fig ([type], optional): [description]. Defaults to None.
        color (str, optional): [description]. Defaults to 'RdBu_r'.
        bar (bool, optional): [description]. Defaults to True.
        ctp_projection (str, optional): [description]. Defaults to 'PlateCarree'.
        label ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if kwargs_pl is None:
        kwargs_pl = dict()

    # create figure
    ax, gl = create_map_plot(ax=ax, ctp_projection=ctp_projection,
                         central_longitude=central_longitude,
                         gridlines_kw=kwargs_gl)


    # choose symmetric vmin and vmax
    if vmin is None and vmax is None:
         vmin = dmap.min(skipna=True)
         vmax = dmap.max(skipna=True)
         vmax = vmax if vmax > (-1*vmin) else (-1*vmin)
         vmin = -1*vmax

    # Select colormap
    if bar == 'continuous':
        cmap = plt.get_cmap(cmap)
        kwargs_pl['vmin'] = vmin 
        kwargs_pl['vmax'] = vmax
    elif bar == 'discrete':
        if 'norm' not in kwargs_pl:
            eps = (vmax-vmin)/10 if eps is None else eps
            bounds = np.arange(vmin, vmax+eps-1e-5, eps)
            # Create colormap
            n_colors = len(bounds)+1
            cmap = plt.get_cmap(cmap, n_colors)
            colors = np.array([mpl.colors.rgb2hex(cmap(i)) for i in range(n_colors)])
            # Set center of colormap to specific color
            if centercolor is not None:
                idx = [len(colors) // 2 - 1, len(colors) // 2]
                colors[idx] = centercolor 
            cmap = mpl.colors.ListedColormap(colors)
            kwargs_pl['norm'] = mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend='both')
        else:
            cmap = plt.get_cmap(cmap)
    else:
        raise ValueError(f"Specified bar={bar} is not defined!")

    if transform is None:
        transform = ctp.crs.PlateCarree(central_longitude=0)

    # plot map
    im = ax.pcolormesh(
        dmap.coords['lon'], dmap.coords['lat'], dmap.data,
        cmap=cmap, transform=transform,
        **kwargs_pl
    )

    # set colorbar
    if add_bar:
        if 'label' not in list(kwargs_cb.keys()):
            kwargs_cb['label'] = dmap.name
        cbar = plt.colorbar(im, ax=ax, **kwargs_cb)
    else:
        cbar = None

    return {'ax': ax, "im": im, 'gl': gl, 'cb': cbar}


def plot_contourf(dmap, ax=None, label=None, 
                    cmap='RdBu_r', vmin=None, vmax=None, eps=0.1, 
                    add_bar=True, centercolor="#FFFFFF",
                    ctp_projection='PlateCarree', transform=None, central_longitude=0,
                    kwargs_pl=dict(extend='both'),
                    kwargs_cb=dict(orientation='horizontal', shrink=0.8),
                    kwargs_gl=dict(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                                   linewidth=0.0)
                    ):
    """Plot map with discrete shading.

    Args:
        dmap ([type]): [description]
        central_longitude (int, optional): [description]. Defaults to 0.
        vmin ([type], optional): [description]. Defaults to None.
        vmax ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        fig ([type], optional): [description]. Defaults to None.
        color (str, optional): [description]. Defaults to 'RdBu_r'.
        bar (bool, optional): [description]. Defaults to True.
        ctp_projection (str, optional): [description]. Defaults to 'PlateCarree'.
        label ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # create figure
    ax, gl = create_map_plot(ax=ax, ctp_projection=ctp_projection,
                         central_longitude=central_longitude,
                         gridlines_kw=kwargs_gl)


    # choose symmetric vmin and vmax
    if vmin is None and vmax is None:
         vmin = dmap.min(skipna=True)
         vmax = dmap.max(skipna=True)
         vmax = vmax if vmax > (-1*vmin) else (-1*vmin)
         vmin = -1*vmax
        
    # Contour levels
    if 'levels' not in kwargs_pl:
        eps = (vmax-vmin)/10 if eps is None else eps
        kwargs_pl['levels'] = np.arange(vmin, vmax+eps-1e-5, eps)
    
    # Color levels
    n_colors = len(kwargs_pl['levels'])+1
    cmap = plt.get_cmap(cmap, n_colors)
    colors = np.array([mpl.colors.rgb2hex(cmap(i)) for i in range(n_colors)])

    # Set center to white
    if centercolor is not None:
        idx = [len(colors) // 2 - 1, len(colors) // 2]
        colors[idx] = centercolor 

    if transform is None:
        transform = ctp.crs.PlateCarree(central_longitude=0)

    # plot map
    im = ax.contourf(
        dmap.coords['lon'], dmap.coords['lat'], dmap.data, colors=colors,
        transform=transform, **kwargs_pl
    )

    # set colorbar
    if add_bar:
        label = dmap.name if label is None else label
        cbar = plt.colorbar(im, label=label, ax=ax, **kwargs_cb)
    else:
        cbar = None

    return {'ax': ax, "im": im, 'gl': gl, 'cb': cbar}


def plot_contour(dmap, ax=None, 
                vmin=None, vmax=None, eps=0.1, 
                add_inline_labels=True, zerolinecolor='white',
                ctp_projection='PlateCarree', transform=None, central_longitude=0,
                kwargs_pl=dict(colors='k'),
                kwargs_labels=dict(fontsize=9, fmt='%1.1f'),
                kwargs_gl=dict(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                               linewidth=0.0),
                 ):
    """Plot map with discrete shading.

    Args:
        dmap ([type]): [description]
        central_longitude (int, optional): [description]. Defaults to 0.
        vmin ([type], optional): [description]. Defaults to None.
        vmax ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        fig ([type], optional): [description]. Defaults to None.
        color (str, optional): [description]. Defaults to 'RdBu_r'.
        bar (bool, optional): [description]. Defaults to True.
        ctp_projection (str, optional): [description]. Defaults to 'PlateCarree'.
        label ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # create figure
    ax, gl = create_map_plot(ax=ax, ctp_projection=ctp_projection,
                         central_longitude=central_longitude,
                         gridlines_kw=kwargs_gl)

        
    # Contour levels
    if 'levels' not in kwargs_pl.keys():
        # choose symmetric vmin and vmax
        vmin = dmap.min(skipna=True) if vmin is None else vmin
        vmax = dmap.max(skipna=True) if vmax is None else vmax
        eps = (vmax-vmin)/10 if eps is None else eps
        kwargs_pl['levels'] = np.arange(vmin, vmax+eps-1e-5, eps)
    
    if transform is None:
        transform = ctp.crs.PlateCarree(central_longitude=0)
    

    # plot map
    im = ax.contour(
        dmap.coords['lon'], dmap.coords['lat'], dmap.data,
        transform=transform, **kwargs_pl
    )

    # set labels 
    if add_inline_labels:
        ax.clabel(im, inline=True, **kwargs_labels)

    if zerolinecolor is not None:
        kwargs = kwargs_pl.copy()
        if 'colors' in kwargs_pl:
            kwargs.pop('colors')
        kwargs.pop('levels')
        
        im = ax.contour(
            dmap.coords['lon'], dmap.coords['lat'], dmap.data,
            levels=[0], transform=transform, **kwargs, colors=zerolinecolor
        )

    return {'ax': ax, "im": im, 'gl': gl}


def significance_mask(mask, ax=None, ctp_projection='PlateCarree', hatch='..',
                      transform=None, central_longitude=0):
    """Plot significante areas using the mask map with True for significant
    and False for non-significant.

    Args:
        mask (xr.Dataarray): Significance mask. 
        ax (plt.Axes, optional): Axes object. Defaults to None.
        ctp_projection (str, optional): Cartopy projection. Defaults to 'PlateCarree'.
        transform (cartopy.Transform, optional): Transform object.
            Defaults to None meaning 'PlateCarree'.
        central_longitude (int, optional): Central longitude. Defaults to 0.

    Returns:
        {'ax': ax, "im": im} (dict): Dictionary with Axes and plot object.
    """
    if ax is None:
        ax = create_map_plot(ax=ax, ctp_projection=ctp_projection,
                             central_longitude=central_longitude)
    if transform is None:
        transform = ctp.crs.PlateCarree(central_longitude=central_longitude)

    # Convert True/False map into 1/NaN
    msk = xr.where(mask == True, 1, np.nan)
    im = ax.pcolor(
        msk['lon'],
        msk['lat'],
        msk.data,
        hatch=hatch,
        alpha=0.0,
        transform=transform,
    )

    return {'ax': ax, "im": im}


def plot_rectangle(ax, lon_range, lat_range, central_longitude, **kwargs):
    """Plots a rectangle on a cartopy map

    Args:
        ax (geoaxis): Axis of cartopy object
        lon_range (list): list of min and max longitude
        lat_range (list): list of min and max lat

    Returns:
        geoaxis: axis with rectangle plotted
    """
    from shapely.geometry.polygon import LinearRing

    shortest = kwargs.pop("shortest", True)
    #if (max(lon_range) - min(lon_range) <= 180) or shortest is False:
    #    cl = 0
    #    lons = [max(lon_range), min(lon_range), min(lon_range), max(lon_range)]
    #else:
    #    cl = 180
    #    lons = [
    #        max(lon_range) - 180,
    #        180 + min(lon_range),
    #        180 + min(lon_range),
    #        max(lon_range) - 180,
    #    ]
    lons = [lon_range[0], lon_range[1], lon_range[1], lon_range[0]]
    lats = [lat_range[0], lat_range[0], lat_range[1], lat_range[1]]

    ring = LinearRing(list(zip(lons, lats)))
    lw = kwargs.pop("lw", 1)
    color = kwargs.pop("color", "k")
    fill = kwargs.pop("fill", False)
    facecolor = color if fill else "none"
    zorder = kwargs.pop('zorder', 11)
    ax.add_geometries(
        [ring],
        ctp.crs.PlateCarree(central_longitude=central_longitude),
        facecolor=facecolor,
        edgecolor=color,
        linewidth=lw,
        zorder=zorder,
    )

    return ax


#Plot matrix colormap using pcolormesh.
def plot_matrix(da: xr.DataArray, xcoord:str, ycoord:str, ax: plt.axes=None,
                vmin: float = None, vmax: float = None, eps: float = 0.1,   
                cmap: str ='RdBu_r', centercolor: str = None,
                bar: str = 'discrete', add_bar: bool=True, 
                kwargs_pl: dict = None,
                kwargs_cb: dict = dict(orientation='horizontal', shrink=0.8, extend='both'),
             ) -> dict:
    """Plot matrix colormap using pcolormesh.

    Args:
        da (xr.DataArray): Dataarray 
        xcoord (str): Coordinates of x.
        ycoord (str): Coordinates of y.
        ax (plt.axes, optional): _description_. Defaults to None.
        vmin (float, optional): _description_. Defaults to None.
        vmax (float, optional): _description_. Defaults to None.
        eps (float, optional): _description_. Defaults to 0.1.
        cmap (str, optional): _description_. Defaults to 'RdBu_r'.
        centercolor (str, optional): _description_. Defaults to None.
        bar (str, optional): _description_. Defaults to 'discrete'.
        add_bar (bool, optional): _description_. Defaults to True.
        kwargs_pl (dict, optional): _description_. Defaults to dict().
        kwargs_cb (dict, optional): _description_. Defaults to dict(orientation='horizontal', shrink=0.8, extend='both').

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    
    if kwargs_pl is None:
        kwargs_pl = dict()
        
    # Select colormap
    if bar == 'continuous':
        cmap = plt.get_cmap(cmap)
        kwargs_pl['vmin'] = vmin 
        kwargs_pl['vmax'] = vmax

    elif bar == 'discrete':
        if 'norm' not in kwargs_pl:
            eps = (vmax-vmin)/10 if eps is None else eps
            bounds = np.arange(vmin, vmax+eps-1e-5, eps)
            # Create colormap
            n_colors = len(bounds)+1
            cmap = plt.get_cmap(cmap, n_colors)
            colors = np.array([mpl.colors.rgb2hex(cmap(i)) for i in range(n_colors)])
            # Set center of colormap to specific color
            if centercolor is not None:
                idx = [len(colors) // 2 - 1, len(colors) // 2]
                colors[idx] = centercolor 
            cmap = mpl.colors.ListedColormap(colors)
            kwargs_pl['norm'] = mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend='both')
        else:
            cmap = plt.get_cmap(cmap)
    else:
        raise ValueError(f"Specified bar={bar} is not defined!")

    # plot map
    im = ax.pcolormesh(
        da[xcoord], da[ycoord], da.data,
        cmap=cmap, **kwargs_pl
    )

    # set colorbar
    if add_bar:
        if 'label' not in list(kwargs_cb.keys()):
            kwargs_cb['label'] = da.name
        cbar = plt.colorbar(im, ax=ax, **kwargs_cb)
    else:
        cbar = None

    return {'ax': ax, "im": im, 'cb': cbar}


def contourf(da: xr.DataArray, xcoord:str, ycoord:str, ax: plt.axes=None,
             vmin: float = None, vmax: float = None, eps: float = 0.1, norm: np.ndarray = None,   
             cmap: str ='RdBu_r', centercolor: str = None, 
             bar: str='discrete', add_bar: bool=True, 
             kwargs_pl: dict = None,
             kwargs_cb: dict = dict(orientation='horizontal', shrink=0.8, extend='both'), 
            ) -> dict:
    """Plot matrix colormap using pcolormesh.

    Args:
        da (xr.DataArray): Dataarray 
        xcoord (str): Coordinates of x.
        ycoord (str): Coordinates of y.
        ax (plt.axes, optional): _description_. Defaults to None.
        vmin (float, optional): _description_. Defaults to None.
        vmax (float, optional): _description_. Defaults to None.
        eps (float, optional): _description_. Defaults to 0.1.
        cmap (str, optional): _description_. Defaults to 'RdBu_r'.
        centercolor (str, optional): _description_. Defaults to None.
        bar (str, optional): _description_. Defaults to 'discrete'.
        add_bar (bool, optional): _description_. Defaults to True.
        kwargs_pl (dict, optional): _description_. Defaults to dict().
        kwargs_cb (dict, optional): _description_. Defaults to dict(orientation='horizontal', shrink=0.8, extend='both').

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    
    if kwargs_pl is None:
        kwargs_pl = dict()

    if bar == 'continuous':
        cmap = plt.get_cmap(cmap)
        kwargs_pl['vmin'] = vmin 
        kwargs_pl['vmax'] = vmax

    elif bar == 'discrete':
        if norm is None:
            eps = (vmax-vmin)/10 if eps is None else eps
            levels = np.arange(vmin, vmax+eps-1e-5, eps)
            # Create colormap
            n_colors = len(levels)+1
            cmap = plt.get_cmap(cmap, n_colors)
            colors = np.array([mpl.colors.rgb2hex(cmap(i)) for i in range(n_colors)])
            # Set center of colormap to specific color
            if centercolor is not None:
                idx = [len(colors) // 2 - 1, len(colors) // 2]
                colors[idx] = centercolor 
            cmap = mpl.colors.ListedColormap(colors)
            kwargs_pl['norm'] = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend='both')
        else:
            cmap = plt.get_cmap(cmap)
            eps = (vmax-vmin)/10 if eps is None else eps
            levels = np.arange(vmin, vmax+eps, eps)
            
    else:
        raise ValueError(f"Specified bar={bar} is not defined!")

        
    # plot map
    im = ax.contourf(
        da[xcoord], da[ycoord], da.data, levels=levels,
        cmap=cmap, **kwargs_pl
    )

    # set colorbar
    if add_bar:
        if 'label' not in list(kwargs_cb.keys()):
            kwargs_cb['label'] = da.name
        cbar = plt.colorbar(im, ax=ax, **kwargs_cb)
    else:
        cbar = None

    return {'ax': ax, "im": im, 'cb': cbar}


def enumerate_axes(axs, pos_x=-0.07, pos_y=1.04, fontsize=None):
    """Adds letters to subplots of a figure.

    Args:
        axs (list): List of plt.axes.
        pos_x (float, optional): x position of label. Defaults to 0.02.
        pos_y (float, optional): y position of label. Defaults to 0.85.
        fontsize (int, optional): Defaults to 18.

    Returns:
        axs (list): List of plt.axes.
    """
    axs = np.array(axs)
    if type(pos_x) == float:
        pos_x = [pos_x] * len(axs.flatten())
    if type(pos_y) == float:
        pos_y = [pos_y] * len(axs.flatten())

    for n, ax in enumerate(axs.flatten()):
        ax.text(
            pos_x[n],
            pos_y[n],
            f"{string.ascii_lowercase[n]}" if n < 26 else f"{string.ascii_lowercase[n-26]}{string.ascii_lowercase[n-26]}.",
            transform=ax.transAxes,
            size=fontsize,
            weight="bold",
            va='top', ha='left',
        )
    return axs