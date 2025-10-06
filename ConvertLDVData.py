
# %% [imports]
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay

from vtk import vtkDoubleArray, vtkPoints, vtkDelaunay3D, vtkUnstructuredGrid, vtkXMLUnstructuredGridWriter, vtkMultiBlockDataSet, vtkXMLMultiBlockDataWriter

# Import pandas
import pandas as pd

# from tqdm.notebook import tqdm
from tqdm import tqdm

from IPython.display import display
from icecream import ic
from textwrap import dedent

pd.set_option('display.float_format', '{:.6e}'.format)
pd.set_option('expand_frame_repr', False)
pd.set_option('colheader_justify', 'right')


def Display(data):
    """
    Displays the entire DataFrame in a Jupyter Notebook.

    Parameters:
    data (pd.DataFrame): The DataFrame to be displayed. If the number of rows is less than 100,
                         all rows will be shown. All columns will be shown regardless of the number of columns.

    Returns:
    None
    """
    pd.set_option('display.width', 1000)
    if len(data) < 100:
        pd.set_option("display.max_rows", len(data))
    pd.set_option('display.max_columns', len(data.columns))
    display(data)
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")


# %% [Setup functions]
def SetupLDV(f):
    """
    Writes the setup content for LDV.py to a CSV file.
    """

    content = """# Setup file for LDV.py

    Parameter;Value;Comment

    ### General input
    RootPath;'.';Root path
    DataPath;'Data/Exports/CRP1_LDV';Data path
    OutputPath;'Analysis/CRP1_LDV';Output path
    Case;'unified-data';Case
    Rref;109.355;Reference radius in mm

    ### Generation of database
    GenerateDatabase;False;True to generate database
    ExternalChannels;[1,2];List of external channels to load
    AxisScaleFactor;[1.0, -1.33, -1.0];Axis scale factor (X,Y,Z)
    ExportCsv;True;Export to .csv files
    ExportMat;False;Export to .mat files

    ### Phase analysis setup
    PhaseAnalysis;True;True to run phase analysis
    RadiusRange;[0,220];Radius range for analysis (in mm)
    PlaneRange;[1];Plane range for analysis (-1 for all planes)
    SkipPlanes;[-1];Skip these planes (default: -1 for no exclusion)
    nStd;4.0;Number of std to remove spurious data
    Period;360.0;Modulo
    Step;2.0;Step between slots
    Wleft;1.0;Slot width to the left
    Wright;1.0;Slot width to the right
    Overwrite;False;Overwrite existing VField files, otherwise skip

    ### Plot generation setup
    GeneratePolarPlots;True;True to generate polar plots
    RotationSign;-1;Rotation sign (-1,+1)
    RefractiveIndexCorrection;0.8;Refractive index correction (1.0 for no correction)
    PolarPlotRadiusLimits;[0.2, 1.25];Radius limits for polar plots
    VerticalUpPhaseOffset;0.0;Phase offset for vertical up axis
    VerticalDownPhaseOffset;0.0;Phase offset for vertical down axis
    HorizontalLeftPhaseOffset;0.0;Phase offset for horizontal left axis
    HorizontalRightPhaseOffset;0.0;Phase offset for horizontal right axis
    Autoscale;True;Set autoscale for polar plots
    Va_samp_range;[0.0, 200.0];Polar plot range (Va samples)
    Va_mean_range;[3, 4.0];Polar plot range (Va mean)
    Va_sdev_range;[0.0, 0.5];Polar plot range (Va rms)
    Vr_samp_range;[0.0, 500.0];Polar plot range (Vr samples)
    Vr_mean_range;[-0.2, 0.2];Polar plot range (Vr mean)
    Vr_sdev_range;[0.0, 0.5];Polar plot range (Vr rms)
    Vt_samp_range;[0.0, 1000.0];Polar plot range (Vt samples)
    Vt_mean_range;[-1.5, 1.5];Polar plot range (Vt mean)
    Vt_sdev_range;[0.0, 0.5];Polar plot range (Vt rms)
    Interpolation;thin_plate_spline;linear/thin_plate_spline/cubic/quintic/gaussian/none

    ### Execution output setup
    Verbose;False;Verbose output
    ShowPhasePlots;False;Show phase plots
    ShowPolarPlots;False;Show polar plots
    """

    with open(f, 'w') as file:
        file.write(dedent(content))


def GetSettings(f):
    """
    Retrieves or initializes settings for LDV (Laser Doppler Velocimetry) analysis.
    If the specified file does not exist, it initializes the settings by calling Init_LDV(),
    creates the necessary directories, and saves the settings to a CSV file. If the file exists,
    it loads the settings from the CSV file, processes them to remove comments and duplicates,
    and cleans up the data.
    Args:
        f (pathlib.Path): The file path where the settings are stored or will be stored.
    Returns:
        pandas.DataFrame: The settings DataFrame with 'Value' and 'Comment' columns.
    """

    if not f.exists():
        print(f'Creating {f}')
        f.parent.mkdir(parents=True, exist_ok=True)
        SetupLDV(f)
    else:
        # Load the settings
        print(f'Loading {f}')

    Settings = pd.read_csv(f, delimiter=';',
                           comment='#',
                           index_col=0,
                           na_filter=False,
                           dtype=str)

    # print(Settings)

    Settings = Settings[~Settings.index.str.startswith('#')]
    Settings = Settings[~Settings.index.duplicated(keep='last')]

    # def split_name(name):
    #    return pd.Series(name.split("#", 1))
    # Settings[['Value', 'Comment']] = Settings['Value'].apply(split_name)

    Settings = Settings.replace(np.nan, '-', regex=True)
    Settings.index = [x.lstrip().rstrip() for x in Settings.index]
    for col in ['Value', 'Comment']:
        Settings[col] = Settings[col].apply(lambda x: str(x).lstrip().rstrip().strip("'"))

    return Settings


def RunSettings(filename):
    """
    Configures and initializes settings for LDV analysis from a CSV file.
    This function reads settings from a CSV file named 'SettingsLDV.csv' located in the current directory.
    It processes and converts the settings into appropriate data types, and then stores them in global variables
    for further use in the analysis.
    Args:
        verbose (bool, optional): If True, displays the settings and their types. Defaults to False.
    Global Variables:
        settings (dict): A dictionary containing the processed settings.
        DataFolder (Path): Path object pointing to the data folder based on the settings.
        OutFolder (Path): Path object pointing to the output folder based on the settings.
        Overlap (float): The overlap percentage between adjacent slots.
    Notes:
        - The function expects the CSV file to have specific columns and values.
        - The settings are categorized and processed based on their expected data types.
        - The function also calculates additional settings like 'Wslot' and 'IntervalClosed'.
        - If 'verbose' is True, the settings and their types are displayed for debugging purposes.
    """

    Settings = GetSettings(Path(filename))
    # display(Settings['Value'].apply(type))

    Values = ['Root', 'PropModel', 'Case', 'Interpolation']

    Values = ['Rref', 'nStd', 'Period', 'Step', 'Wleft', 'Wright',
              'VerticalUpPhaseOffset', 'VerticalDownPhaseOffset',
              'HorizontalLeftPhaseOffset', 'HorizontalRightPhaseOffset',
              'RefractiveIndexCorrection']
    for value in Values:
        Settings.at[value, 'Value'] = float(Settings.at[value, 'Value'])

    Values = ['RotationSign']
    for value in Values:
        Settings.at[value, 'Value'] = int(Settings.at[value, 'Value'])

    Values = ['ExportCsv', 'ExportMat',
              'GenerateDatabase', 'PhaseAnalysis',
              'GeneratePolarPlots', 'Autoscale',
              'Verbose', 'Overwrite',
              'ShowPhasePlots', 'ShowPolarPlots']
    for value in Values:
        Settings.at[value, 'Value'] = True if Settings.at[value, 'Value'] == 'True' else False

    Values = ['AxisScaleFactor',
              'RadiusRange', 'PolarPlotRadiusLimits',
              'Va_samp_range', 'Va_mean_range', 'Va_sdev_range',
              'Vr_samp_range', 'Vr_mean_range', 'Vr_sdev_range',
              'Vt_samp_range', 'Vt_mean_range', 'Vt_sdev_range']
    for value in Values:
        lst = Settings.at[value, 'Value']
        lst = list(lst.strip('[]').split(','))
        if lst == ['']:
            raise ValueError(f"The list for {value} is empty.")
        Settings.at[value, 'Value'] = [float(a) for a in lst]

    Values = ['ExternalChannels', 'PlaneRange', 'SkipPlanes']
    for value in Values:
        lst = Settings.at[value, 'Value']
        lst = list(lst.strip('[]').split(','))
        if lst == ['']:
            lst = ['-1']
        Settings.at[value, 'Value'] = [int(a) for a in lst]

    settings = Settings['Value'].to_dict()
    settings['Wslot'] = settings['Wleft']+settings['Wright']
    settings['IntervalClosed'] = 'left'

    settings['DataFolder'] = Path(settings['RootPath'], settings['DataPath'], settings['Case'])
    settings['OutFolder'] = Path(settings['RootPath'], settings['OutputPath'], settings['Case'])

    if settings['Verbose']:
        display(Settings)

    return settings

# print(RunSettings('test.csv'))


# %% [Phase analysis]
def SetIntervals(period, step, width_left, width_right):
    """
    Generates intervals and their centers based on the given period, step, and widths.

    Args:
        period (float): The total period over which intervals are generated.
        step (float): The step size between interval centers.
        width_left (float): The left width from the interval center.
        width_right (float): The right width from the interval center.

    Returns:
        tuple: A tuple containing:
            - intervals (pd.arrays.IntervalArray): The generated intervals.
            - centers (np.ndarray): The centers of the intervals.
    """

    centers = np.arange(0, period, step)
    left_edges = centers - width_left
    right_edges = centers + width_right

    intervals = pd.arrays.IntervalArray.from_arrays(left_edges, right_edges, closed=settings['IntervalClosed'])
    # display(intervals)

    return (intervals, centers)


def calculate_vorticity_cylindrical(df, u_col, v_col, w_col):
    """
    Calculates the vorticity vector in cylindrical coordinates.

    Args:
        df (pd.DataFrame): DataFrame with columns 'R', 'Angular position (deg)', 'X', u_col, v_col, w_col.
        u_col (str): Column name for the radial velocity component (u).
        v_col (str): Column name for the tangential velocity component (v).
        w_col (str): Column name for the axial velocity component (w).

    Returns:
        pd.DataFrame: DataFrame with added columns 'vorticity_r', 'vorticity_theta', 'vorticity_z'.
                      Returns None if input is invalid.
    """

    if not all(col in df.columns for col in ['R', 'Angular position (deg)', 'X', u_col, v_col, w_col]):
        print("Error: DataFrame must contain columns 'R', 'Angular position (deg)', 'X', u_col, v_col, w_col.")
        return None

    r = df['R'].values
    theta = np.deg2rad(df['Angular position (deg)'].values)  # Convert degrees to radians
    x = df['X'].values
    u = df[u_col].values
    v = df[v_col].values
    w = df[w_col].values

    ic(r, theta, x)
    # Calculate partial derivatives using NumPy's gradient function.  This assumes a reasonably structured grid.
    # For highly unstructured data, consider more sophisticated interpolation methods.
    dr = np.gradient(r)
    dtheta = np.gradient(theta)
    dx = np.gradient(x)

    ic(dr, dtheta, dx)

    du_dtheta = np.gradient(u, dtheta)
    du_dx = np.gradient(u, dx)
    dv_dx = np.gradient(v, dx)
    dw_dr = np.gradient(w, dr)

    # Vorticity components
    vorticity_r = (1/r) * (np.gradient(r*w, dtheta) - dv_dx)
    vorticity_theta = (du_dx - dw_dr)
    vorticity_x = (1/r) * (np.gradient(r*v, dr) - du_dtheta)

    return vorticity_r, vorticity_theta, vorticity_x


def calculate_magnitude(axis0, axis1, axis2):

    return np.sqrt(axis0**2 + axis1**2 + axis2**2)


def calculate_tke(axial, radial, tangential):
    """
    Calculates the turbulent kinetic energy (TKE) from the velocity components.
    Args:
        axial (np.ndarray): Axial velocity component.
        radial (np.ndarray): Radial velocity component.
        tangential (np.ndarray): Tangential velocity component.
    Returns:
        np.ndarray: The calculated TKE.
    """
    return 0.5 * (axial**2 + radial**2 + tangential**2)


    # %% [Slice functions
def Slice(Planes, dir, sw, datafolder, outfolder, nx=10, nr=10, smooth=0.0, verbose=False):

    # Statfile = [item for item in datafolder.iterdir()]
    # print(Statfile)

    Intervals, Ctrs = SetIntervals(settings['Period'], settings['Step'],
                                   settings['Wleft'], settings['Wright'])
    ic(Ctrs)
    reps = int(360/settings['Period'])
    ic(reps)

    with tqdm(total=len(Planes), dynamic_ncols=True) as pbar:
        vstats = {}
        for orientation in ['Vu', 'Vd', 'Hl', 'Hr']:

            orient = ('Up' if orientation == 'Vu' else
                      'Down' if orientation == 'Vd' else
                      'Left' if orientation == 'Hl' else 'Right')
            vstats[orient] = pd.DataFrame([], columns=['Slot', 'Angular position (deg)',
                                                       'R', 'X',
                                                       'Ch. 1 samples', 'Ch. 1 mean', 'Ch. 1 sdev',
                                                       'Ch. 2 samples', 'Ch. 2 mean', 'Ch. 2 sdev'])

            offset = 0
            match orient:
                case 'Up':
                    offset = settings['VerticalUpPhaseOffset']
                case 'Down':
                    offset = settings['VerticalDownPhaseOffset']
                case 'Left':
                    offset = settings['HorizontalLeftPhaseOffset']
                case 'Right':
                    offset = settings['HorizontalRightPhaseOffset']

            cond0 = (Data['Orientation'] == orientation)
            cond1 = (Data['R (mm)'] > 0.0)
            for plane in Planes['id'][::]:

                cond2 = (Data['Plane'] == plane)
                data = Data[cond0 & cond1 & cond2].copy()
                if len(data) == 0:
                    continue
                # ic(orient, len(Data[cond0 & cond1]), len(Data[cond0 & cond1 & cond2]))

                count = 0
                data.reset_index(drop=True, inplace=True)
                for irow, row in data.iterrows():
                    # R = row['R (mm)'] / settings['Rref']
                    R = row['R (mm)'] * 1e-3
                    if R < 1e-6:
                        continue
                    # Xp = row['X (mm)'] / settings['Rref']
                    Xp = row['X (mm)'] * 1e-3

                    statfile = Path(datafolder, '%s_Stats_%s_P%06d' % (settings['Case'], sw, row['Point']))
                    statfile = statfile.with_suffix('.csv')

                    if statfile.exists():
                        # vstat = pd.DataFrame()
                        try:
                            vstat = pd.read_csv(statfile)

                            vstat['R'] = R * settings['RefractiveIndexCorrection']
                            vstat['X'] = -Xp # minus sign to convert to right-handed system

                            vstat['Angular position (deg)'] += offset
                            # ic(irow, len(vstat), len(vstats))
                            if count == 0:
                                vstatp = vstat.copy()
                            else:
                                vstatp = pd.concat([vstatp, vstat], ignore_index=True)
                            count += 1

                        except pd.errors.ParserError:
                            print('ParserError: %s' % statfile)
                            continue

                        except pd.errors.EmptyDataError:
                            print('EmptyDataError: %s' % statfile)
                            continue

                        # if verbose:
                        #     print('Missing %s (file index %d)' % (statfile, irow))
                        # if irow == 0:
                        #     vstatp = vstat0.copy()
                        # else:
                        #     vstatp = pd.concat([vstatp, vstat0], ignore_index=True)

                    # ic(count)
                    # ic(vstatp, len(vstatp))

                if count < len(data):
                    print('%d of %d points missing for %s (%s) in plane %d' % (len(data)-count, len(data), orient, orientation, plane))
                ic(dir.name, orient, plane, len(data), count)

                cos = np.cos(np.deg2rad(vstatp['Angular position (deg)']))
                sin = np.sin(np.deg2rad(vstatp['Angular position (deg)']))
                vstatp['Y'] = vstatp['R'] * cos
                vstatp['Z'] = vstatp['R'] * sin
                vstatp.sort_values(by=['Angular position (deg)', 'R', 'X'], inplace=True)
                vstatp.reset_index(drop=True, inplace=True)

                rad = vstatp['R'].to_numpy()
                theta = vstatp['Angular position (deg)'].to_numpy()
                pts = np.vstack([rad, theta]).T

                for i in [1, 2]:
                    if orient in ['Up', 'Down']:
                        comp = 'radial' if i == 1 else 'axial'
                    else:
                        comp = 'tangential' if i == 1 else 'axial'

                    # ic(plane, comp)
                    vn = 'Count %s velocity (%s)' % (comp, orient)
                    vm = 'Mean %s velocity (%s)' % (comp, orient)
                    vs = 'RMS %s velocity (%s)' % (comp, orient)
                    Var = [[f'Ch. {i} samples', vn], [f'Ch. {i} mean', vm], [f'Ch. {i} sdev', vs]]
                    for var in Var:
                        chi, cho = var
                        ic(chi)

                        v = vstatp[chi].to_numpy()
                        # np.nan_to_num(v, copy=False)
                        cond = ~np.isnan(v)
                        # ic(cho, np.count_nonzero(cond))
                        # ic(ch, len(v))

                        interp = RBFInterpolator(pts[cond], v[cond],
                                                 smoothing=smooth,
                                                 kernel=settings['Interpolation'])
                        # ic(interp.kernel)

                        V = interp(pts[cond])
                        # ic(V.shape)

                        vstatp.loc[cond, cho] = V.T
                        # ic(vstatp[name])

                if plane == 0:
                    vstats[orient] = vstatp.copy()
                else:
                    vstats[orient] = pd.concat([vstats[orient], vstatp], ignore_index=True)

                slicefile = Path(outfolder, 'Slice_%s_P%03d' % (orient, plane))
                vstatp.to_csv(slicefile.with_suffix('.csv'), index=False)

            if len(vstats[orient]) == 0:
                vstats.pop(orient)
                continue
            ic(vstats[orient], vstats.keys())

        xmin = ymin = 1e6
        xmax = ymax = -1e6
        for orient in vstats.keys():
            # ic(orient, len(vstats[orient]))
            if xmin > vstats[orient]['X'].min():
                xmin = vstats[orient]['X'].min()
            if xmax < vstats[orient]['X'].max():
                xmax = vstats[orient]['X'].max()
            if ymin > vstats[orient]['R'].min():
                ymin = vstats[orient]['R'].min()
            if ymax < vstats[orient]['R'].max():
                ymax = vstats[orient]['R'].max()
        X = np.linspace(xmin, xmax, nx)
        Y = np.linspace(ymin, ymax, nr)
        ic(X, Y)
        dx = np.abs(X[1]-X[0])
        dr = np.abs(Y[1]-Y[0])
        da = np.abs(Intervals[1].right - Intervals[0].right)
        ic(dx, dr, da)
        rda = np.deg2rad(da) * Y
        ic(rda)
        dX = np.full((nr, nx), dx)
        dR = np.full((nr, nx), dr)
        rdA = np.tile(np.array([rda]).T, (1, nx))
        ic(dX, dX[:, 0])
        ic(dR, dR[0, :])
        ic(rdA, rdA[0, :], rdA[:, 0])

        Xi, Yi = np.meshgrid(X, Y)
        xi = Xi.flatten()
        yi = Yi.flatten()

        Vol = pd.DataFrame()
        for interval in Intervals[::]:
            new_slice = True
            Slice = pd.DataFrame()
            ic(interval)

            for orient in vstats.keys():
                # ic('key:', orient)

                slice = vstats[orient].loc[vstats[orient]['Slot'] == str(interval)].copy()

                angle = np.median(slice['Angular position (deg)'].to_numpy())
                angle_s = np.std(slice['Angular position (deg)'].to_numpy())
                if angle_s > 1.0:
                    print('Angle std > 1.0 for %s (%s)' % (orient, interval))
                    continue

                cos = np.cos(np.deg2rad(angle))
                sin = np.sin(np.deg2rad(angle))
                x = slice['X'].to_numpy()
                r = slice['R'].to_numpy()
                pts = np.vstack([x, r]).T

                if interval == Intervals[0]:
                    slice.to_csv(Path(outfolder, 'Slice_%s.csv' % orient), index=False)

                # xi = x
                # yi = r
                Pts = np.vstack([xi, yi]).T

                if new_slice:
                    Slice = pd.DataFrame([xi, yi*cos, yi*sin]).T
                    Slice.columns = ['X', 'Y', 'Z']
                    Slice['Angular position (deg)'] = angle
                    Slice['R'] = np.sqrt(Slice['Y']**2 + Slice['Z']**2)
                    new_slice = False
                    # ic(len(Slice), len(slice), len(Pts))

                for i in [1, 2]:
                    if orient in ['Up', 'Down']:
                        comp = 'radial' if i == 1 else 'axial'
                    else:
                        comp = 'tangential' if i == 1 else 'axial'

                    vn = 'Count %s velocity (%s)' % (comp, orient)
                    vm = 'Mean %s velocity (%s)' % (comp, orient)
                    vs = 'RMS %s velocity (%s)' % (comp, orient)
                    Var = [vn, vm, vs]
                    for var in Var:
                        # ic(var)
                        
                        v = slice[var].to_numpy()
                        # np.nan_to_num(v, copy=False)
                        cond = ~np.isnan(v)
                        # ic(var, np.count_nonzero(cond))

                        interp = RBFInterpolator(pts[cond], v[cond],
                                                 smoothing=smooth,
                                                 kernel=settings['Interpolation'])
                        # ic(interp.kernel)

                        V = interp(Pts)
                        # ic(Pts.shape, V.shape)

                        Slice = pd.concat([Slice, pd.DataFrame({var: V.T})], axis=1)
                        # ic(Slice)

            Slice['Velocity magnitude (Up)'] = calculate_magnitude(Slice['Mean axial velocity (Up)'],
                                                                   Slice['Mean radial velocity (Up)'],
                                                                   Slice['Mean tangential velocity (Left)'])
            Slice['Velocity magnitude (Left)'] = calculate_magnitude(Slice['Mean axial velocity (Left)'],
                                                                     Slice['Mean radial velocity (Up)'],
                                                                     Slice['Mean tangential velocity (Left)'])

            Slice['TKE partial (Up)'] = calculate_tke(Slice['RMS axial velocity (Up)'],
                                                      Slice['RMS radial velocity (Up)'], 0)
            Slice['TKE partial (Left)'] = calculate_tke(Slice['RMS axial velocity (Left)'],
                                                        Slice['RMS tangential velocity (Left)'], 0)
            Slice['TKE total (Up)'] = calculate_tke(Slice['RMS axial velocity (Up)'],
                                                    Slice['RMS radial velocity (Up)'],
                                                    Slice['RMS tangential velocity (Left)'])
            Slice['TKE total (Left)'] = calculate_tke(Slice['RMS axial velocity (Left)'],
                                                      Slice['RMS radial velocity (Up)'],
                                                      Slice['RMS tangential velocity (Left)'])

            Vort = {'Mean axial vorticity': ['Mean tangential velocity (Left)', 'Mean radial velocity (Up)', dR, rdA],
                    'Mean radial vorticity (Up)': ['Mean axial velocity (Up)', 'Mean tangential velocity (Left)', rdA, dX],
                    'Mean radial vorticity (Left)': ['Mean axial velocity (Left)', 'Mean tangential velocity (Left)', rdA, dX],
                    'Mean tangential vorticity (Up)': ['Mean radial velocity (Up)', 'Mean axial velocity (Up)', dX, dR],
                    'Mean tangential vorticity (Left)': ['Mean radial velocity (Up)', 'Mean axial velocity (Left)', dX, dR]}

            for vort in Vort.keys():
                # ic(vort, Vort[vort], Vort[vort][0], Vort[vort][1], dX, dY)
                u = Slice[Vort[vort][0]].to_numpy()
                v = Slice[Vort[vort][1]].to_numpy()
                dy = Vort[vort][2]
                dx = Vort[vort][3]
                # ic(u.shape, v.shape)
                u = np.reshape(u, (nr, nx))
                v = np.reshape(v, (nr, nx))
                # ic(u.shape, v.shape)
                # ic(dx, len(dx), dy, len(dy))
                du = np.gradient(u/dy, 1.0, axis=0, edge_order=2)
                dv = np.gradient(v/dx, 1.0, axis=1, edge_order=2)
                Slice[vort] = (du - dv).flatten()

            Slice['Vorticity magnitude (Up)'] = calculate_magnitude(Slice['Mean axial vorticity'],
                                                                    Slice['Mean radial vorticity (Up)'],
                                                                    Slice['Mean tangential vorticity (Up)'])
            Slice['Vorticity magnitude (Left)'] = calculate_magnitude(Slice['Mean axial vorticity'],
                                                                      Slice['Mean radial vorticity (Left)'],
                                                                      Slice['Mean tangential vorticity (Left)'])

            if interval == Intervals[0]:
                Vol = Slice.copy()

                # Add periodic slice to close 3d gap
                cos = np.cos(np.deg2rad(settings['Period']))
                sin = np.sin(np.deg2rad(settings['Period']))
                r = Slice['R'].to_numpy()
                Slice['Y'] = r * cos
                Slice['Z'] = r * sin
                
            Vol = pd.concat([Vol, Slice], ignore_index=True)

        Vol = Vol.reset_index(drop=True)

        return Vol


# %% [Velocity gradient tensor]
def calculate_velocity_gradient_from_dataframe_cylindrical(df, u, v, w):
    """
    Calculates the velocity gradient tensor from a Pandas DataFrame
    containing cylindrical coordinates and velocity components.

    Args:
        df (pd.DataFrame): DataFrame with columns 'r', 'theta', 'z', 'u', 'v', 'w'.
                           It is assumed that the data points represent a structured
                           grid in cylindrical coordinates, although the spacing
                           may be non-uniform. The DataFrame should be sortable
                           in a way that reflects the grid structure (e.g.,
                           sorted by z, then theta, then r).

    Returns:
        numpy.ndarray: 5D array representing the velocity gradient tensor Dv_ij
                       with shape (nr, ntheta, nz, 3, 3), where nr, ntheta, nz
                       are the number of unique values in 'r', 'theta', and 'z'
                       respectively. The last two dimensions correspond to the
                       (i, j) components in the cylindrical basis (r, theta, z).
                       Returns None if the input DataFrame is not suitable.
    """
    # if not all(col in df.columns for col in ['r', 'theta', 'z', 'u', 'v', 'w']):
    #     print("Error: DataFrame must contain columns 'r', 'theta', 'z', 'u', 'v', 'w'.")
    #     return None


    df['R'] = np.trunc(df['R']*1e6)/1e6
    df['Angular position (deg)'] = np.trunc(df['Angular position (deg)']*1e6)/1e6
    df['X'] = np.trunc(df['X']*1e6)/1e6

    r_unique = np.unique(df['R'].values)
    theta_unique = np.unique(df['Angular position (deg)'].values)
    theta_unique = np.append(theta_unique, settings['Period'])  # Close the loop
    x_unique = np.unique(df['X'].values)
    nr, ntheta, nx = len(r_unique), len(theta_unique), len(x_unique)
    ic(nr, ntheta, nx)
    ic(r_unique, theta_unique, x_unique)
    ic(len(r_unique), len(theta_unique), len(x_unique))

    Dv = np.zeros((nr, ntheta, nx, 3, 3))

    r_indices = np.searchsorted(r_unique, df['R'].values)
    theta_indices = np.searchsorted(theta_unique, df['Angular position (deg)'].values)
    x_indices = np.searchsorted(x_unique, df['X'].values)
    ic(r_indices, theta_indices, x_indices)

    u_grid = np.zeros((nr, ntheta, nx))
    v_grid = np.zeros((nr, ntheta, nx))
    w_grid = np.zeros((nr, ntheta, nx))

    u_grid[r_indices, theta_indices, x_indices] = df[u].values
    v_grid[r_indices, theta_indices, x_indices] = df[v].values
    w_grid[r_indices, theta_indices, x_indices] = df[w].values

    # Handle periodicity in theta by copying the first slice to the last
    u_grid[r_indices, -1, x_indices] = u_grid[r_indices, 0, x_indices]
    v_grid[r_indices, -1, x_indices] = v_grid[r_indices, 0, x_indices]
    w_grid[r_indices, -1, x_indices] = w_grid[r_indices, 0, x_indices]

    # Derivatives with respect to r
    du_dr = np.zeros_like(u_grid, dtype=float)
    dv_dr = np.zeros_like(v_grid, dtype=float)
    dw_dr = np.zeros_like(w_grid, dtype=float)
    for j in range(ntheta):
        for k in range(nx):
            du_dr[:, j, k] = np.gradient(u_grid[:, j, k], r_unique)
            dv_dr[:, j, k] = np.gradient(v_grid[:, j, k], r_unique)
            dw_dr[:, j, k] = np.gradient(w_grid[:, j, k], r_unique)

    ic(du_dr)

    # Derivatives with respect to theta
    du_dtheta = np.zeros_like(u_grid, dtype=float)
    dv_dtheta = np.zeros_like(v_grid, dtype=float)
    dw_dtheta = np.zeros_like(w_grid, dtype=float)
    theta_unique_rad = np.deg2rad(theta_unique)
    for i in range(nr):
        for k in range(nx):
            du_dtheta[i, :, k] = np.gradient(u_grid[i, :, k], theta_unique_rad)
            dv_dtheta[i, :, k] = np.gradient(v_grid[i, :, k], theta_unique_rad)
            dw_dtheta[i, :, k] = np.gradient(w_grid[i, :, k], theta_unique_rad)

    # Derivatives with respect to z
    du_dx = np.zeros_like(u_grid, dtype=float)
    dv_dx = np.zeros_like(v_grid, dtype=float)
    dw_dx = np.zeros_like(w_grid, dtype=float)
    for i in range(nr):
        for j in range(ntheta):
            du_dx[i, j, :] = np.gradient(u_grid[i, j, :], x_unique)
            dv_dx[i, j, :] = np.gradient(v_grid[i, j, :], x_unique)
            dw_dx[i, j, :] = np.gradient(w_grid[i, j, :], x_unique)

    # Construct the velocity gradient tensor components
    # Extract velocity gradient components
    Dv[..., 0, 0] = du_dr
    Dv[..., 0, 1] = du_dtheta
    Dv[..., 0, 2] = du_dx
    Dv[..., 1, 0] = dv_dr
    Dv[..., 1, 1] = dv_dtheta
    Dv[..., 1, 2] = dv_dx
    Dv[..., 2, 0] = dw_dr
    Dv[..., 2, 1] = dw_dtheta
    Dv[..., 2, 2] = dw_dx

    R_grid, theta_grid, x_grid = np.meshgrid(r_unique, theta_unique, x_unique, indexing='ij')
    Coord = np.stack((R_grid, theta_grid, x_grid), axis=-1)

    velocity = np.stack((u_grid, v_grid, w_grid), axis=-1)
    vel_norm = np.linalg.norm(velocity, axis=-1)
    Vel = np.stack((u_grid, v_grid, w_grid, vel_norm), axis=-1)

    Div = (1 / R_grid) * (np.gradient((R_grid * u_grid), r_unique, axis=0)[0]) + (1 / R_grid) * dv_dtheta + dw_dx

    return Vel, Dv, Coord, Div


# %% [Symmetric and antisymmetric tensors]


def decompose_velocity_gradient(Vel, Dv, Coord):
    """
    Decomposes the velocity gradient tensor into its symmetric (strain rate)
    and antisymmetric (vorticity) parts.

    Args:
        Dv (numpy.ndarray): 5D array representing the velocity gradient tensor
                           with shape (nr, ntheta, nz, 3, 3).

    Returns:
        tuple: A tuple containing two numpy.ndarrays:
               - S (numpy.ndarray): Symmetric part (strain rate tensor) with the same shape as Dv.
               - Omega (numpy.ndarray): Antisymmetric part (vorticity tensor) with the same shape as Dv.
    """

    nr, ntheta, nx, _, _ = Dv.shape
    r = Coord[..., 0]
    Grad = np.zeros((nr, ntheta, nx, 3, 3))

    u = Vel[..., 0]
    v = Vel[..., 1]

    du_dr = Dv[..., 0, 0]
    du_dtheta = Dv[..., 0, 1]
    du_dx = Dv[..., 0, 2]
    dv_dr = Dv[..., 1, 0]
    dv_dtheta = Dv[..., 1, 1]
    dv_dx = Dv[..., 1, 2]
    dw_dr = Dv[..., 2, 0]
    dw_dtheta = Dv[..., 2, 1]
    dw_dx = Dv[..., 2, 2]

    Grad[..., 0, 0] = du_dr
    Grad[..., 0, 1] = (1 / r) * (du_dtheta - v)
    Grad[..., 0, 2] = du_dx
    Grad[..., 1, 0] = dv_dr
    Grad[..., 1, 1] = (1 / r) * (dv_dtheta + u)
    Grad[..., 1, 2] = dv_dx
    Grad[..., 2, 0] = dw_dr
    Grad[..., 2, 1] = (1 / r) * dw_dtheta
    Grad[..., 2, 2] = dw_dx

    Omega = np.zeros((nr, ntheta, nx, 3, 3))
    S = np.zeros((nr, ntheta, nx, 3, 3))
    ic(Omega.shape, S.shape)

    # Calculate vorticity tensor components
    Omega[..., 0, 1] = 0.5 * (Grad[..., 0, 1] - Grad[..., 1, 0])
    Omega[..., 0, 2] = 0.5 * (Grad[..., 0, 2] - Grad[..., 2, 0])
    Omega[..., 1, 0] = -Omega[..., 0, 1]
    Omega[..., 1, 2] = 0.5 * (Grad[..., 1, 1] - Grad[..., 2, 1])
    Omega[..., 2, 0] = -Omega[..., 0, 2]
    Omega[..., 2, 1] = -Omega[..., 1, 2]

    # Calculate strain rate tensor components
    S[..., 0, 0] = 0.5 * (Grad[..., 0, 0] + Grad[..., 0, 0])
    S[..., 0, 1] = 0.5 * (Grad[..., 0, 1] + Grad[..., 1, 0])
    S[..., 0, 2] = 0.5 * (Grad[..., 0, 2] + Grad[..., 2, 0])
    S[..., 1, 0] = S[..., 0, 1]  # Symmetry
    S[..., 1, 1] = 0.5 * (Grad[..., 1, 1] + Grad[..., 1, 1])
    S[..., 1, 2] = 0.5 * (Grad[..., 1, 2] + Grad[..., 2, 1])
    S[..., 2, 0] = S[..., 0, 2]  # Symmetry
    S[..., 2, 1] = S[..., 1, 2]  # Symmetry
    S[..., 2, 2] = 0.5 * (Grad[..., 2, 2] + Grad[..., 2, 2])

    return S, Omega


def calculate_frobenius_norm(tensor):
    """
    Calculates the Frobenius norm of a tensor.

    Args:
        tensor (numpy.ndarray): A tensor with shape (..., M, N), where the Frobenius norm is calculated over the last two dimensions.

    Returns:
        numpy.ndarray: The Frobenius norm of the tensor, with shape (...).
    """
    return np.sqrt(np.sum(tensor**2, axis=(-2, -1)))


def calculate_q_criterion(S, Omega):
    """Calculates the Q-criterion from the strain rate and vorticity tensors.

    Args:
        S (numpy.ndarray): Strain rate tensor with shape (..., 3, 3).
        Omega (numpy.ndarray): Vorticity tensor with shape (..., 3, 3).

    Returns:
        numpy.ndarray: Q-criterion with shape (...).
    """

    S_norm = calculate_frobenius_norm(S)
    Omega_norm = calculate_frobenius_norm(Omega)
    Q = 0.5 * (Omega_norm**2 - S_norm**2)

    return Q


def calculate_vorticity_vector(Dv, Coord):
    """
    Calculates the vorticity vector from the velocity gradient tensor in cylindrical coordinates.

    Args:
        Dv (numpy.ndarray): Velocity gradient tensor with shape (nr, ntheta, nx, 3, 3).
        r (numpy.ndarray): Radial coordinates with shape (nr, ntheta, nx).
        theta (numpy.ndarray): Azimuthal coordinates with shape (nr, ntheta, nx) in radians.
        x (numpy.ndarray): Axial coordinates with shape (nr, ntheta, nx).

    Returns:
        numpy.ndarray: Vorticity vector with shape (nr, ntheta, nx, 3).
    """

    r = Coord[..., 0]

    du_dr = Dv[..., 0, 0]
    du_dtheta = Dv[..., 0, 1]
    du_dx = Dv[..., 0, 2]
    dv_dr = Dv[..., 1, 0]
    dv_dtheta = Dv[..., 1, 1]
    dv_dx = Dv[..., 1, 2]
    dw_dr = Dv[..., 2, 0]
    dw_dtheta = Dv[..., 2, 1]
    dw_dx = Dv[..., 2, 2]

    # Calculate vorticity components
    vorticity_r = (1 / r) * dw_dtheta - dv_dx
    vorticity_theta = du_dx - dw_dr
    vorticity_x = (1 / r) * (dv_dr - du_dtheta)

    # Combine vorticity components into a vector
    vorticity = np.stack([vorticity_r, vorticity_theta, vorticity_x], axis=-1)
    vorticity_norm = np.linalg.norm(vorticity, axis=-1)
    Vort = np.stack([vorticity_r, vorticity_theta, vorticity_x, vorticity_norm], axis=-1)
    ic(Vort.shape)

    return Vort


# %% [VTU export]


def AddScalarArray(vtk_dataset, name, data, cnames):

    npoints = data.shape[0]
    narrays = 1
    if len(data.shape) > 1:
        narrays = data.shape[1]

    array = vtkDoubleArray()
    array.SetName(name)
    array.SetNumberOfComponents(narrays)
    array.SetNumberOfTuples(npoints)

    ic(data)
    if narrays > 1:
        for i, cname in enumerate(cnames):
            array.SetComponentName(i, cname)
        for i, val in enumerate(data):
            array.SetTuple(i, val)
    else:
        for i, val in enumerate(data):
            array.SetTuple1(i, val)
    vtk_dataset.GetPointData().AddArray(array)

    return vtk_dataset


def AddTensorArray(vtk_dataset, name, cname, data):
    array = vtkDoubleArray()
    array.SetName(name)
    array.SetNumberOfComponents(9)
    for i in range(3):
        for j in range(3):
            array.SetComponentName(3*i + j, f'{cname}_{i}{j}')
    array.SetNumberOfTuples(len(data))
    for i, tensor in enumerate(data):
        array.SetTuple(i, tensor)
    vtk_dataset.GetPointData().AddArray(array)

    return vtk_dataset


def ExportDatasetToVTK(Vol, outfile):
    """
    Creates a VTK unstructured grid dataset from a Delaunay triangulation and a specified z-coordinate.
    Parameters:
    tri (scipy.spatial.Delaunay): A Delaunay triangulation object containing points and simplices.
    z (float): The z-coordinate to be assigned to all points in the dataset.
    Returns:
    vtkUnstructuredGrid: A VTK unstructured grid dataset with the specified points and triangles.
    """

    X = Vol['X'].to_numpy().flatten()
    x = np.unique(X)
    zeros = np.zeros(len(x))
    ic(x, zeros, len(x))

    Zero = pd.DataFrame([x, zeros, zeros]).T
    Zero.columns = ['X', 'Y', 'Z']
    Vol = pd.concat([Vol, Zero])

    Vol.sort_values(by=['R', 'Angular position (deg)', 'X'], inplace=True)
    # Vol = Vol.sample(frac=1.0, random_state=1, ignore_index=True)
    Vol.reset_index(drop=True, inplace=True)

    X = Vol['X'].to_numpy().flatten()
    Y = Vol['Y'].to_numpy().flatten()
    Z = Vol['Z'].to_numpy().flatten()
    pts = list(np.vstack([X, Y, Z]).T)

    points = vtkPoints()
    rng = np.random.default_rng()
    for id, pt in enumerate(pts):
        x, y, z = pt #/ (settings['Rref'] * 1e-3)

        x = x + (2*rng.random() - 1)/1e6
        y = y + (2*rng.random() - 1)/1e6
        z = z + (2*rng.random() - 1)/1e6

        # ic(x, y, z)
        points.InsertPoint(id, [x, y, z])

    vtk_dataset = vtkUnstructuredGrid()
    vtk_dataset.SetPoints(points)

    axial = [var for var in Vol.columns if 'axial velocity' in var.lower()]
    radial = [var for var in Vol.columns if 'radial velocity' in var.lower()]
    tangential = [var for var in Vol.columns if 'tangential velocity' in var.lower()]
    magnitude = [var for var in Vol.columns if 'velocity magnitude' in var.lower()]
    vorticity = [var for var in Vol.columns if 'vorticity' in var.lower()]
    turbulence = [var for var in Vol.columns if 'tke' in var.lower()]
    Var = [[axial, 'Axial velocity'],
           [radial, 'Radial velocity'],
           [tangential, 'Tangential velocity'],
           [magnitude, 'Velocity magnitude'],
           [vorticity, 'Vorticity'],
           [turbulence, 'TKE']]
    for var in Var:
        cnames, lbl = var
        # ic(len(Vol[cnames[0]]))
        V = []
        for cname in cnames:
            ic(cname)
            vol = Vol[cname].to_numpy().flatten()
            # ic(len(vol), vol)
            if cname == cnames[0]:
                V = [vol]
            else:
                V.append(vol)
            # ic(len(V), V)
        V = np.vstack(V).T
        vtk_dataset = AddScalarArray(vtk_dataset, lbl, V, cnames)

    # tri = Delaunay(pts, qhull_options='QJ')
    # ic(tri)
    # vtk_dataset.Allocate(tri.nsimplex)
    # for point_ids in tri.simplices:
    #     vtk_dataset.InsertNextCell(VTK_TRIANGLE, 4, point_ids)

    ic(settings['Interpolation'], settings['Smoothing'])
    print('Running Delaunay...')
    delaunay = vtkDelaunay3D()
    # delaunay.SetTolerance(0.01)
    delaunay.SetInputData(vtk_dataset)
    delaunay.Update()

    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(outfile)
    writer.SetInputConnection(delaunay.GetOutputPort())
    # writer.SetInputData(vtk_dataset)
    writer.Write()


def ExportTensorsToVTK(Dv, coordinates, Vel, Div, Vort, S, Omega, Q, outfile):
    """Exports strain rate, vorticity, and Q-criterion tensors to a VTK file."""

    # Create points from coordinates
    ic(coordinates.shape)
    pts = coordinates.reshape(-1, 3)
    x_unique = np.unique(pts[:, 2])
    ic(x_unique, len(x_unique))
    zero = np.zeros((len(x_unique), 3))
    zero[:, 2] = x_unique
    pts = np.vstack([pts, zero])
    ic(pts, pts.shape)

    points = vtkPoints()
    rng = np.random.default_rng()
    for id, pt in enumerate(pts):
        x, y, z = pt #/ (settings['Rref'] * 1e-3)

        x = x + (2*rng.random() - 1)/1e6
        y = y + (2*rng.random() - 1)/1e6
        z = z + (2*rng.random() - 1)/1e6

        cos = np.cos(np.deg2rad(y))
        sin = np.sin(np.deg2rad(y))
        points.InsertPoint(id, [z, x*cos, x*sin])

    vtk_dataset = vtkUnstructuredGrid()
    vtk_dataset.SetPoints(points)

    # Reshape tensors to match the number of points
    Div = Div.reshape(-1)
    Q = Q.reshape(-1)
    Vel = Vel.reshape(-1, 4)
    Vort = Vort.reshape(-1, 4)
    S = S.reshape(-1, 9)
    Omega = Omega.reshape(-1, 9)

    ic(Div.shape, Q.shape, Vel.shape, Vort.shape, S.shape, Omega.shape)

    # Add NaN entries for the added zero points
    Div = np.hstack([Div, np.full(len(x_unique), np.nan)])
    Q = np.hstack([Q, np.full(len(x_unique), np.nan)])
    nand = np.full((1, 4), np.nan).flatten()
    Vel = np.vstack([Vel, np.tile(nand, (len(x_unique), 1))])
    Vort = np.vstack([Vort, np.tile(nand, (len(x_unique), 1))])
    nand = np.full((3, 3), np.nan).flatten()
    S = np.vstack([S, np.tile(nand, (len(x_unique), 1))])
    Omega = np.vstack([Omega, np.tile(nand, (len(x_unique), 1))])

    ic(Div.shape, Q.shape, Vel.shape, Vort.shape, S.shape, Omega.shape)
    # Add tensors to VTK dataset
    AddScalarArray(vtk_dataset, 'Velocity', Vel[:, :3], ['Radial', 'Tangential', 'Axial']) # no magnitude
    AddScalarArray(vtk_dataset, 'Vorticity', Vort[:, :3], ['Radial', 'Tangential', 'Axial']) # no magnitude
    AddScalarArray(vtk_dataset, "Divergence", Div, None)

    AddTensorArray(vtk_dataset, "Strain tensor", 'S', S)
    AddTensorArray(vtk_dataset, "Vorticity tensor", 'Omega', Omega)
    AddScalarArray(vtk_dataset, "Q", Q, None)

    ic(settings['Interpolation'], settings['Smoothing'])
    print('Running Delaunay...')
    delaunay = vtkDelaunay3D()
    # delaunay.SetTolerance(0.01)
    delaunay.SetInputData(vtk_dataset)
    delaunay.Update()

    # Write VTK file
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(outfile)
    writer.SetInputConnection(delaunay.GetOutputPort())
    writer.Write()


def ExportTensorsToCSV(Dv, coordinates, Vel, Div, Vort, S, Omega, Q, outfile):
    """Exports strain rate, vorticity, and Q-criterion tensors to a CSV file."""
    nr, ntheta, nx, _, _ = Dv.shape
    ic(nr, ntheta, nx)
    R = coordinates[..., 0].flatten()
    Theta = coordinates[..., 1].flatten()
    X = coordinates[..., 2].flatten()
    Y = R * np.cos(np.deg2rad(Theta))
    Z = R * np.sin(np.deg2rad(Theta))
    ic(R, Theta, X)
    data = {
        'R': R,
        'Angular position (deg)': Theta,
        'X': X,
        'Y': Y,
        'Z': Z,
    }
    axis = ['radial', 'tangential', 'axial', 'magnitude']
    for i in range(4):
        data[f'Velocity ({axis[i]})'] = Vel[..., i].flatten()
    data['Divergence'] = Div.flatten()
    for i in range(4):
        data[f'Vorticity ({axis[i]})'] = Vel[..., i].flatten()

    for i in range(3):
        for j in range(3):
            data[f'S_{i}{j}'] = S[..., i, j].flatten()
    for i in range(3):
        for j in range(3):
            data[f'Omega_{i}{j}'] = Omega[..., i, j].flatten()
    data['Q'] = Q.flatten()

    df = pd.DataFrame(data)
    df.to_csv(outfile, index=False)


def Derivatives(vol, Qcsv, Qvtu):

    Vel, Dv, Coord, Div = calculate_velocity_gradient_from_dataframe_cylindrical(vol,
                                                                                 'Mean radial velocity (Up)',
                                                                                 'Mean tangential velocity (Left)',
                                                                                 'Mean axial velocity (Up)')
    ic(Dv.shape)
    S, Omega = decompose_velocity_gradient(Vel, Dv, Coord)
    Q = calculate_q_criterion(S, Omega)
    ic(S.shape, Omega.shape, Q.shape)

    Vort = calculate_vorticity_vector(Dv, Coord)

    ExportTensorsToCSV(Dv, Coord, Vel, Div, Vort, S, Omega, Q, Qcsv)
    ExportTensorsToVTK(Dv, Coord, Vel, Div, Vort, S, Omega, Q, Qvtu)


# %% [Main]
args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
ic(args, len(args))

if len(args) != 5:
    print('Usage: %s <settings file> <nx> <nr> <start index> <end index>' % Path(__file__).name)
    sys.exit(0)
else:
    settings_filename = args[0]
    nx = int(args[1])
    nr = int(args[2])
    d0 = int(args[3])
    d1 = int(args[4])

global settings
settings = RunSettings(settings_filename)
display(settings)

SourceFolder = Path(settings['RootPath'], settings['OutputPath']).parent
ic(SourceFolder)
volfolder = Path(SourceFolder, 'Volume')
volfolder.mkdir(exist_ok=True)

dirs = [item for item in SourceFolder.iterdir() if item.is_dir() and '-' in item.name]
dirs = sorted(dirs, key=lambda x: int((x.name).split('-')[1]))
ic(dirs)

DataPath = Path(settings['OutFolder'], '%s_Stats5.fth' % settings['Case'])
if not DataPath.exists():
    print('%s does not exist' % DataPath)
    sys.exit(0)

Data = pd.read_feather(DataPath)
# display(Data)
seen = set()
Planes = pd.DataFrame(columns=['id', 'x'])
for x in Data['Plane']:
    if x in settings['SkipPlanes']:
        continue
    if x not in seen:
        xp = Data.loc[Data['Plane'] == x, 'X (mm)'].mean()
        Planes.loc[len(Planes)] = {'id': x, 'x': xp}
        seen.add(x)
Planes.sort_values(by=['x'], ascending=False, inplace=True)
Planes.reset_index(drop=True, inplace=True)
ic(Planes)
# Planes = Planes[20:41]

ic("%d planes:" % len(Planes), Planes)

sw = 'S%04dW%04d' % (settings['Step']*100, settings['Wslot']*100)
ic(sw)

with tqdm(total=len(dirs), dynamic_ncols=True, desc=dirs[0].name) as pbar:
    for dir in dirs[d0:d1:]:
        pbar.desc = dir.name
        pbar.update(1)

        datafolder = Path(dir, settings['Case'], 'PolarStats', sw, 'Csv')
        ic(datafolder)
        outfolder = Path(datafolder, 'Slice')
        outfolder.mkdir(exist_ok=True)

        angle = int((dir.name).split('-')[1])
        Vcsv = Path(volfolder, 'Vol_V%03d' % angle).with_suffix('.csv')
        Vvtu = Path(volfolder, 'Vol_V%03d' % angle).with_suffix('.vtu')
        Qcsv = Path(volfolder, 'Vol_Q%03d' % angle).with_suffix('.csv')
        Qvtu = Path(volfolder, 'Vol_Q%03d' % angle).with_suffix('.vtu')
        ic(Vcsv, Vcsv.exists(), Vvtu, Vvtu.exists())
        if not Vcsv.exists():
            print('Creating %s' % Vcsv)

            Vol = Slice(Planes, dir, sw, datafolder, outfolder, nx, nr, settings['Smoothing'], settings['Verbose'])
            Vol.to_csv(Vcsv, index=False)
            ExportDatasetToVTK(Vol, Vvtu)

            Derivatives(Vol, Qcsv, Qvtu)

        else:
            Vol = pd.read_csv(Vcsv)
            if not Vvtu.exists():
                print('Creating %s' % Vvtu)

                ExportDatasetToVTK(Vol, Vvtu)

                Derivatives(Vol, Qcsv, Qvtu)
            else:
                if not Qvtu.exists():
                    print('Creating %s' % Qvtu)

                    Derivatives(Vol, Qcsv, Qvtu)
