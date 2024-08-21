"""Util functions for analyzing ENSO types."""
import os, cftime
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats
from sklearn.decomposition import PCA
import hyblim.data.preproc as utpp
from hyblim.data.eof import EmpiricalOrthogonalFunctionAnalysis

PATH = os.path.dirname(os.path.abspath(__file__))

def get_nino_indices(ssta, time_range=None, monthly=False, antimeridian=False):
    """Returns the time series of the Nino 1+2, 3, 3.4, 4, 5

    Args:
        ssta (xr.dataarray): Sea surface temperature anomalies.
        time_range (list, optional): Select only a certain time range.
        monthly (boolean): Averages time dimensions to monthly. 
                            Default to True.

    Returns:
        [type]: [description]
    """
    if ssta.lon.max() > 180:
        fn_transform = lambda lons: utpp.lon_to_360(np.array(lons))
    elif antimeridian is True:
        fn_transform = utpp.get_antimeridian_coord
    else:
        fn_transform = lambda lons: lons
        
    lon_range = fn_transform([-90, -80]) 
    nino12, nino12_std = utpp.get_mean_time_series(
        ssta, lon_range=lon_range,
        lat_range=[-10, 0], time_roll=0
    )
    nino12.name = 'nino12'

    lon_range = fn_transform([-150, -90]) 
    nino3, nino3_std = utpp.get_mean_time_series(
        ssta, lon_range=lon_range,
        lat_range=[-5, 5], time_roll=0
    )
    nino3.name = 'nino3'

    lon_range = fn_transform([-170, -120])
    nino34, nino34_std = utpp.get_mean_time_series(
        ssta, lon_range=lon_range,
        lat_range=[-5, 5], time_roll=0
    )
    nino34.name = 'nino34'

    lon_range = fn_transform([160, -150])
    nino4, nino4_std = utpp.get_mean_time_series(
        ssta, lon_range=lon_range,
        lat_range=[-5, 5], time_roll=0
    )
    nino4.name = 'nino4'

    lon_range = fn_transform([130, 160])
    nino5, nino5_std = utpp.get_mean_time_series(
        ssta, lon_range=lon_range,
        lat_range=[-5, 5], time_roll=0
    )
    nino5.name = 'nino5'

    nino_idx = xr.merge([nino12, nino3, nino34, nino4, nino5])

    if monthly:
        nino_idx = nino_idx.resample(time='M', label='left' ).mean()
        nino_idx = nino_idx.assign_coords(
            dict(time=nino_idx['time'].data + np.timedelta64(1, 'D'))
        )

    # Cut time period
    if time_range is not None:
        nino_idx = nino_idx.sel(time=slice(
            np.datetime64(time_range[0], "M"), 
            np.datetime64(time_range[1], "M")
        ))
    return nino_idx


def get_tni_index(ssta):
    """ TNI index (Trenberth & Stepaniak, 2001)"""
    nino_idx = get_nino_indices(ssta)
    n12 = nino_idx['nino12']/ nino_idx['nino12'].std()
    n4 = nino_idx['nino4']/ nino_idx['nino4'].std()
    tni = n12 - n4
    tni.name = 'tni'
    return tni


def get_emi_index(ssta):
    """El Niño Modoki index (EMI; Ashok et al., 2007)."""
    central, central_std = utpp.get_mean_time_series(
        ssta, lon_range=[-165, 140],
        lat_range=[-10, 10]
    )
    eastern, eastern_std =  utpp.get_mean_time_series(
        ssta, lon_range=[-110, -70],
        lat_range=[-15, 5]
    )
    western, western_std =  utpp.get_mean_time_series(
        ssta, lon_range=[125, 145],
        lat_range=[-10, 20]
    )
    emi = central - 0.5 * (eastern + western)
    emi.name = 'emi'
    return emi


def get_epcp_index(ssta):
    """EPnew–CPnew indices (Sullivan et al., 2016)."""
    nino_idx = get_nino_indices(ssta)
    n3 = nino_idx['nino3']/ nino_idx['nino3'].std()
    n4 = nino_idx['nino4']/ nino_idx['nino4'].std()

    ep_idx = n3 - 0.5 * n4
    cp_idx = n4 - 0.5 * n3
    ep_idx.name = 'EPnew'
    cp_idx.name = 'CPnew'
    return ep_idx, cp_idx


def EC_indices(ssta, pc_sign=[1, 1], time_range=None):
    """E and C indices (Takahashi et al., 2011).

    Args:
        ssta (xr.DataArray): Dataarray of SSTA in the region
            lat=[-10,10] and lon=[120E, 70W].
        pc_sign (list, optional): Sign of principal components which can be switched
            for consistency with e.g. Nino-indices. See ambigouosy of sign of PCA.
            For:
                ERA5 data set to [1,-1].
                CMIP6 models set to[-1,-1]
            Defaults to [1,1].

    Returns:
        e_index (xr.Dataarray)
        c_index (xr.Dataarray)
    """

    # Flatten and remove NaNs
    buff = ssta.stack(z=('lat', 'lon'))
    ids = ~np.isnan(buff.isel(time=0).data)
    X = buff.isel(z=ids)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(X.data)

    # Modes
    ts_modes = []
    for i, comp in enumerate(pca.components_):
        ts = stats.zscore(X.data @ comp, axis=0)
        # Flip sign of mode due to ambiguousy of sign
        ts = pc_sign[i] * ts
        ts_modes.append(
            xr.DataArray(data=ts,
                         name=f'eof{i+1}',
                         coords={"time": X.time},
                         dims=["time"])
        )
    ts_mode = xr.merge(ts_modes)

    # Compute E and C index
    # Changed sign of eof2 due to sign flip of it
    e_index = (ts_mode['eof1'] - ts_mode['eof2']) / np.sqrt(2)
    e_index.name = 'E'

    c_index = (ts_mode['eof1'] + ts_mode['eof2']) / np.sqrt(2)
    c_index.name = 'C'

    # Cut time period
    if time_range is not None:
        e_index = e_index.sel(time=slice(
            np.datetime64(time_range[0], "M"), np.datetime64(
                time_range[1], "M")
        ))
        c_index = c_index.sel(time=slice(
            np.datetime64(time_range[0], "M"), np.datetime64(
                time_range[1], "M")
        ))

    return e_index, c_index


#########################################################################################
# ENSO classification
#########################################################################################
def get_enso_flavors_N3N4(nino_indices,
                          month_range=[12, 2],
                          mean=True, threshold=0.5,
                          offset=0.0,
                          min_diff=0.0,
                          drop_volcano_year=False):
    """Get nino flavors from Niño‐3–Niño‐4 approach (Kug et al., 2009; Yeh et al.,2009).

    Parameters:
    -----------
        min_diff (float): min_diff between nino3 and nino4 to get only the
                            extreme EP or CP
        threshold (float, str): Threshold to define winter as El Nino or La Nina,
                                A float or 'std' are possible.
                                Default: 0.5.
    """
    if offset > 0.0:
        print("Warning! A new category of El Nino and La Ninas are introduced." )

    if threshold == 'std':
        threshold_nino3 = float(nino_indices['nino3'].std(skipna=True))
        threshold_nino4 = float(nino_indices['nino4'].std(skipna=True))
    else:
        threshold_nino3 = float(threshold)
        threshold_nino4 = float(threshold)
    
    def is_datetime360(time):
        return isinstance(time, cftime._cftime.Datetime360Day)

    def is_datetime(time):
        return isinstance(time, cftime._cftime.DatetimeNoLeap)

    # Identify El Nino and La Nina types
    enso_classes = []
    sd, ed = np.array([nino_indices.time.data.min(), nino_indices.time.data.max()])
    if is_datetime360(nino_indices.time.data[0]) or is_datetime(nino_indices.time.data[0]):
        times = xr.cftime_range(start=sd,
                                end=ed,
                                freq='Y')
    else:
        times = np.arange(
            np.array(sd, dtype='datetime64[Y]'),
            np.array(ed, dtype='datetime64[Y]')
        )
    for y in times:
        if is_datetime360(nino_indices.time.data[0]):
            y = y.year
            y_end = y+1 if month_range[1] < month_range[0] else y
            time_range = [cftime.Datetime360Day(y, month_range[0], 1),
                          cftime.Datetime360Day(y_end, month_range[1]+1, 1)]
        elif is_datetime(nino_indices.time.data[0]):
            y = y.year
            y_end = y+1 if month_range[1] < month_range[0] else y
            time_range = [cftime.DatetimeNoLeap(y, month_range[0], 1),
                          cftime.DatetimeNoLeap(y_end, month_range[1]+1, 1)]
        else:
            y_end = y+1 if month_range[1] < month_range[0] else y
            time_range = [np.datetime64(f"{y}-{month_range[0]:02d}-01", "D"),
                          np.datetime64(f"{y_end}-{month_range[1]+1:02d}-01", "D")-1]

        # Select time window
        nino34 = nino_indices['nino34'].sel(time=slice(time_range[0], time_range[1]))
        nino3 = nino_indices['nino3'].sel(time=slice(time_range[0], time_range[1]))
        nino4 = nino_indices['nino4'].sel(time=slice(time_range[0], time_range[1]))

        # Choose mean or min
        if mean:
            nino34 = nino34.mean(dim='time', skipna=True)
            nino3 = nino3.mean(dim='time', skipna=True)
            nino4 = nino4.mean(dim='time', skipna=True)
        else:
            nino34 = nino34.min(dim='time', skipna=True)
            nino3 = nino3.min(dim='time', skipna=True)
            nino4 = nino4.min(dim='time', skipna=True)

        # El Nino years
        if ((nino3.data >= threshold_nino3) or (nino4.data >= threshold_nino4)):
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nino', 'N3': float(nino3), 'N4': float(nino4)}
            buff_dic['N3-N4'] = nino3.data - nino4.data

            Nino_EP_label = 'Nino_EP_weak' if offset > 0 else 'Nino_EP'
            Nino_CP_label = 'Nino_CP_weak' if offset > 0 else 'Nino_CP'

            # EP type if DJF nino3 > 0.5 and nino3 > nino4
            if (nino3.data - min_diff) > nino4.data:
                buff_dic['type'] = Nino_EP_label
            # CP type if DJF nino4 > 0.5 and nino3 < nino4
            elif (nino4.data - min_diff) > nino3.data:
                buff_dic['type'] = Nino_CP_label

            # Strong El Ninos
            if offset > 0.0:
                if (nino3.data >= threshold_nino3 + offset) and (nino3.data - min_diff) > nino4.data:
                    buff_dic['type'] = "Nino_EP_strong"
                elif (nino4.data >= threshold_nino4 + offset) and (nino4.data - min_diff) > nino3.data:
                    buff_dic['type'] = 'Nino_CP_strong'

            enso_classes.append(buff_dic)

        # La Nina years
        elif ((nino3.data <= -threshold_nino3) or (nino4.data <= -threshold_nino4)):
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nina', 'N3': float(nino3), 'N4': float(nino4)}
            buff_dic['N3-N4'] = nino3.data - nino4.data

            Nina_EP_label = 'Nina_EP_weak' if offset > 0 else 'Nina_EP'
            Nina_CP_label = 'Nina_CP_weak' if offset > 0 else 'Nina_CP'

            # EP type if DJF nino3 < -0.5 and nino3 < nino4
            if (nino3.data + min_diff) < nino4.data:
                buff_dic['type'] = Nina_EP_label
            # CP type if DJF nino4 < -0.5 and nino3 > nino4
            elif (nino4.data + min_diff) < nino3.data:
                buff_dic['type'] = Nina_CP_label

            # Strong La Nina
            if offset > 0.0:
                if (nino3.data <= -threshold_nino3 - offset) and (nino3.data + min_diff) < nino4.data:
                    buff_dic['type'] = "Nina_EP_strong"
                elif (nino4.data <= -threshold_nino4 - offset) and (nino4.data + min_diff) < nino3.data:
                    buff_dic['type'] = 'Nina_CP_strong'

            enso_classes.append(buff_dic)

        # standard years
        else:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Normal', 'N3': float(nino3), 'N4': float(nino4)}
            buff_dic['N3-N4'] = nino3.data - nino4.data
            enso_classes.append(buff_dic)

    enso_classes = pd.DataFrame(enso_classes)

    # Years of strong volcanic erruptions followed by an El Nino
    if drop_volcano_year:
        volcano_years_idx = enso_classes.loc[
            (enso_classes['start'] == '1955-12-01') |
            (enso_classes['start'] == '1956-12-01') |
            (enso_classes['start'] == '1957-12-01') |
            (enso_classes['start'] == '1963-12-01') |
            (enso_classes['start'] == '1980-12-01') |
            (enso_classes['start'] == '1982-12-01') |
            (enso_classes['start'] == '1991-12-01')
        ].index
        enso_classes = enso_classes.drop(index=volcano_years_idx)

    return enso_classes


def get_enso_flavor_EC(e_index, c_index, month_range=[12, 2],
                       offset=0.0, mean=True, nino_indices=None):
    """Classify winters into their ENSO flavors based-on the E- and C-index.

    The E- and C-index was introduced by Takahashi et al. (2011).
    The following criterias are used:


    Args:
        e_index (xr.DataArray): E-index.
        c_index (xr.DataArray): C-index.
        month_range (list, optional): Month range where to consider the criteria.
            Defaults to [12,2].
        offset (float, optional): Offset to identify only extremes of the flavors.
            Defaults to 0.0.
        mean (boolean, optional): If True the mean of the range must exceed the threshold.
            Otherwise all months within the range must exceed the threshold.
            Defaults to True.

    Returns:
        enso_classes (pd.DataFrame): Dataframe containing the classification.
    """
    e_threshold = e_index.std(dim='time', skipna=True) + offset
    c_threshold = c_index.std(dim='time', skipna=True) + offset

    years = np.arange(
        np.array(e_index.time.min(), dtype='datetime64[Y]'),
        np.array(e_index.time.max(), dtype='datetime64[Y]')
    )
    enso_classes = []
    for y in years:
        time_range = [np.datetime64(f"{y}-{month_range[0]:02d}-01", "D"),
                      np.datetime64(f"{y+1}-{month_range[1]+1:02d}-01", "D")-1]
        # Either mean or min of DJF must exceed threshold
        e_range = e_index.sel(time=slice(*time_range))
        c_range = c_index.sel(time=slice(*time_range))
        if mean:
            e_range = e_range.mean(dim='time', skipna=True)
            c_range = c_range.mean(dim='time', skipna=True)

        # Preselect EN and LN conditions based on Nino34
        if nino_indices is not None:
            nino34 = nino_indices['nino34'].sel(
                time=slice(time_range[0], time_range[1]))
            nino3 = nino_indices['nino3'].sel(
                time=slice(time_range[0], time_range[1]))
            nino4 = nino_indices['nino4'].sel(
                time=slice(time_range[0], time_range[1]))

            # Normal conditions
            if ((nino34.min() >= -0.5 and nino34.max() <= 0.5)
                or (nino3.min() >= -0.5 and nino3.max() <= 0.5)
                    or (nino4.min() >= -0.5 and nino4.max() <= 0.5)):
                buff_dic = {'start': time_range[0], 'end': time_range[1],
                            'type': 'Normal'}
                enso_classes.append(buff_dic)
                continue

        # EPEN
        if e_range.min() >= e_threshold:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nino_EP'}
        # CPEN
        elif c_range.min() >= c_threshold:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nino_CP'}
        # EPLN
        elif e_range.max() <= -e_threshold:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nina_EP'}
        # CPLN
        elif c_range.max() <= -c_threshold:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nina_CP'}
        # Normal
        else:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Normal'}

        enso_classes.append(buff_dic)

    return pd.DataFrame(enso_classes)


def get_enso_flavors_obs(definition='N3N4', fname=None,
                         vname='sst', climatology='month',
                         month_range=[12, 2],
                         time_range=None, offset=0.0):
    """Classifies given month range into ENSO flavors.

    Args:
        definition (str, optional): Definition used for classification.
            Defaults to 'N3N4'.
        fname (str, optional): Each definition might require information of other
            datasets, i.e.:
                'N3N4' requires the global SST dataset.
                'EC' requires the global SST dataset for the EOF analysis.
                'N3N4_NOAA' requires the nino-indices by NOAA
                'Cons' requires a table of classifications.
            Defaults to None which uses the preset paths.
        vname (str): Varname of SST only required for 'N3N4' and 'EC'. Defaults to 'sst'.
        land_area_mask (xr.Dataarray): Land area fraction mask, i.e. 0 over oceans.
            Defaults to None.
        climatology (str, optional): Climatology to compute anomalies.
            Only required for 'N3N4' and 'EC'. Defaults to 'month'.
        month_range (list, optional): Month range. Defaults to [11,1].
        time_range (list, optional): Time period of interest.
            Defaults to None.
        offset (float, optional): Offset for 'extreme' events.
            Defaults to 0.0.

    Raises:
        ValueError: If wrong definition is defined.

    Returns:
        (pd.Dataframe) Containing the classification including the time-period.
    """
    if definition in ['N3N4', 'EC']:
        if fname is None:
            raise ValueError(f"Attribute fname must be set if definition={definition}!")
        # Process global SST data
        da_sst = xr.open_dataset(fname)[vname]

        # Check dimensions
        # TODO: remove sorting here, this takes forever
        da_sst = utpp.check_dimensions(da_sst, sort=True)
        # Detrend data
        da_sst = utpp.detrend_dim(da_sst, dim='time', startyear=1950)
        # Anomalies
        ssta = utpp.compute_anomalies(da_sst, group=climatology)

    if definition == 'N3N4':
        nino_indices = get_nino_indices(ssta, time_range=time_range)
        enso_classes = get_enso_flavors_N3N4(
            nino_indices, month_range=month_range, mean=True,
            threshold=0.5, offset=offset,
            min_diff=0.1, drop_volcano_year=False
        )
    elif definition == 'EC':
        # Cut area for EOFs
        lon_range = [120, -80]
        lat_range = [-10, 10]
        ssta = utpp.cut_map(
            ds=ssta, lon_range=lon_range, lat_range=lat_range, shortest=True
        )

        # EC-index based-on EOFs
        e_index, c_index = EC_indices(
            ssta, pc_sign=[1, -1], time_range=time_range
        )
        enso_classes = get_enso_flavor_EC(
            e_index, c_index, month_range=month_range, mean=True,
            offset=offset
        )
    elif definition == 'N3N4_NOAA':
        # N3N4 approach
        nino_indices = get_nino_indices(ssta, time_range=time_range)
        enso_classes = get_enso_flavors_N3N4(
            nino_indices, month_range=month_range, mean=True, threshold=0.5,
            offset=offset, min_diff=0.1, drop_volcano_year=False
        )
    else:
        raise ValueError(f"Specified ENSO definition type {definition} is not defined! "
                         + "The following are defined: 'N3N4', 'Cons', 'EC'")

    return enso_classes


def get_enso_flavors_cmip(fname_sst, vname='ts', land_area_mask=None, climatology='month',
                          definition='N3N4', month_range=[12, 2],
                          time_range=None, offset=0.0, detrend_from=1950):
    """Classifies CMIP data into ENSO flavors.

    Args:
        fname_sst (str): Path to global SST dataset.
        vname (str): Varname of SST. Defaults to 'ts'.
        land_area_mask (xr.Dataarray): Land area fraction mask, i.e. 0 over oceans.
            Defaults to None.
        climatology (str, optional): Climatology to compute anomalies. Defaults to 'month'.
        definition (str, optional): Definition used for classification.
            Defaults to 'N3N4'.
        month_range (list, optional): Month range. Defaults to [11,1].
        time_range (list, optional): Time period of interest.
            Defaults to None.
        offset (float, optional): Offset for 'extreme' events.
            Defaults to 0.0.

    Raises:
        ValueError: If wrong definition is defined.

    Returns:
        (pd.Dataframe) Containing the classification including the time-period.
    """
    # Process global SST data
    da_sst = xr.open_dataset(fname_sst)[vname]
    # Mask only oceans
    if land_area_mask is not None or definition != 'N3N4':
        da_sst = da_sst.where(land_area_mask == 0.0)

    # Check dimensions
    da_sst = utpp.check_dimensions(da_sst, sort=True)
    # Detrend data
    if detrend_from is not None:
        da_sst = utpp.detrend_dim(da_sst, startyear=detrend_from)
    # Anomalies
    ssta = utpp.compute_anomalies(da_sst, group=climatology)

    # ENSO event classification 
    if definition == 'N3N4':
        nino_indices = get_nino_indices(ssta, time_range=time_range)
        enso_classes = get_enso_flavors_N3N4(
            nino_indices, month_range=month_range, mean=True, threshold=0.5,
            offset=offset, min_diff=0.1, drop_volcano_year=False
        )
    elif definition == 'EC':
        # Cut area for EOFs
        lon_range = [120, -80]
        lat_range = [-10, 10]
        ssta = utpp.cut_map(
            ds=ssta, lon_range=lon_range, lat_range=lat_range, shortest=True
        )

        # EC-index based-on EOFs
        e_index, c_index = EC_indices(
            ssta, pc_sign=[-1, -1], time_range=time_range
        )
        enso_classes = get_enso_flavor_EC(
            e_index, c_index, month_range=month_range, mean=True,
            offset=offset
        )
    else:
        raise ValueError(f"Specified ENSO definition type {definition} is not defined! "
                         + "The following are defined: 'N3N4', 'Cons', 'EC'")
    return enso_classes


