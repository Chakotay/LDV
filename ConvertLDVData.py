
# %% [imports]
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# Import numpy
import numpy as np
import numpy.core._multiarray_umath
import scipy.io as io
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay

import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from vtk import VTK_TRIANGLE, vtkDoubleArray, vtkPoints
from vtk import vtkUnstructuredGrid, vtkXMLUnstructuredGridWriter
import vtkmodules.util.data_model
import vtkmodules.util.execution_model

# Import pandas
import pandas as pd
import pyarrow
import openpyxl

# from tqdm.notebook import tqdm
from tqdm import tqdm

from IPython.display import display
import timeit
from icecream import ic
from textwrap import dedent


# %% [Graphical setup]
# %matplotlib qt
# %matplotlib --list
# %matplotlib inline
plt.isinteractive()
# mpl.use('gtk4agg')
# mpl.use('kitcat')
# plt.rc('font', family='sans-serif')
# plt.rc('text', usetex=True)
# plt.rcParams['text.latex.preamble'] = [
#       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
#       r'\usepackage{helvet}',    # set the normal font here
#       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
# ]

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
    nStd;4.0;Number of std to remove spurious data
    Period;360.0;Modulo
    Step;2.0;Step between slots
    Wleft;1.0;Slot width to the left
    Wright;1.0;Slot width to the right
    Overwrite;False;Overwrite existing VField files, otherwise skip

    ### Plot generation setup
    GeneratePolarPlots;True;True to generate polar plots
    RotationSign;-1;Rotation sign (-1,+1)
    RefractiveIndexCorrection;0.98;Refractive index correction (1.0 for no correction)
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

    Values = ['ExternalChannels', 'PlaneRange']
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


# %% [VTU export]
def MakeVtkDataset(tri, z):
    """
    Creates a VTK unstructured grid dataset from a Delaunay triangulation and a specified z-coordinate.
    Parameters:
    tri (scipy.spatial.Delaunay): A Delaunay triangulation object containing points and simplices.
    z (float): The z-coordinate to be assigned to all points in the dataset.
    Returns:
    vtkUnstructuredGrid: A VTK unstructured grid dataset with the specified points and triangles.
    """

    vtk_dataset = vtkUnstructuredGrid()
    pts = vtkPoints()
    for id, pt in enumerate(tri.points):
        x, y = pt
        pts.InsertPoint(id, [x, y, z])
    vtk_dataset.SetPoints(pts)

    vtk_dataset.Allocate(tri.nsimplex)
    for point_ids in tri.simplices:
        vtk_dataset.InsertNextCell(VTK_TRIANGLE, 3, point_ids)

    return vtk_dataset


def AddArray(vtk_dataset, name, data, cnames):
    """
    Adds a named array to a VTK dataset.
    Parameters:
    vtk_dataset (vtk.vtkDataSet): The VTK dataset to which the array will be added.
    name (str): The name of the array to be added.
    data (list or numpy.ndarray): The data to be added to the array. Should be a list or array of values.
    cnames (list of str): The names of the components in the array.
    Returns:
    vtk.vtkDataSet: The VTK dataset with the added array.
    """

    npoints = vtk_dataset.GetNumberOfPoints()
    ndata = len(data)

    # ic(vtk_dataset.GetNumberOfPoints())
    array = vtkDoubleArray()
    array.SetName(name)
    array.SetNumberOfComponents(ndata)
    for i, cname in enumerate(cnames):
        array.SetComponentName(i, cname)

    array.SetNumberOfTuples(npoints)
    dat = np.dstack((data)).reshape(npoints, ndata)
    # print(dat)
    for i, val in enumerate(dat):
        # print(val)
        array.SetTuple(i, val)
    vtk_dataset.GetPointData().AddArray(array)

    return vtk_dataset


def ExportToVTKVtu(block, plane, orientation, X):
    """
    Exports data to a VTK (.vtu) file format.
    Parameters:
    block (tuple): A tuple containing theta, rad, Vn, Vm, Vs arrays.
    row (dict): A dictionary containing metadata for the current row, including 'Orientation' and 'Plane'.
    X (float): A scalar value used for normalization.
    Returns:
    None
    The function performs the following steps:
    1. Constructs the output folder path based on settings.
    2. Extracts and normalizes the radial and axial coordinates.
    3. Creates a Delaunay triangulation of the points.
    4. Determines the orientation and labels for the velocity components.
    5. Interpolates the velocity data if the orientation is 'Left' or 'Right'.
    6. Adds the velocity data arrays to the VTK dataset.
    7. Writes the VTK dataset to a .vtu file.
    """

    sw = 'S%04dW%04d' % (settings['Step']*100, settings['Wslot']*100)
    outfolder = Path(settings['OutFolder'], 'PolarStats', sw, 'Vtk')
    outfolder.mkdir(exist_ok=True)

    theta, rad, Vn, Vm, Vs = block

    Z = X / settings['Rref']

    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    # z = np.full((x.shape[0], x.shape[1]), Z)
    points = np.vstack([x.flatten(), y.flatten()]).T
    points = np.vstack((points, (0, 0)))

    tri = Delaunay(points)
    vtk_dataset = MakeVtkDataset(tri, Z)

    orientation = ('Up' if orientation == 'Vu' else 'Down' if orientation == 'Vd' else
                   'Left' if orientation == 'Hl' else 'Right')
    Lbl = ['Radial velocity (%s)' % orientation, 'Axial velocity (%s)' % orientation]
    if orientation in ['Left', 'Right']:
        Lbl = ['Tangential velocity (%s)' % orientation, 'Axial velocity (%s)' % orientation]

    for k, lbl in enumerate(Lbl):

        V = [Vn[k, :, :], Vm[k, :, :], Vs[k, :, :]]

        # print('Interpolation:', settings['Interpolation'])
        if settings['Interpolation'] != 'none' and orientation in ['Left', 'Right']:
            fact = settings['RefractiveIndexCorrection']
            kernel = settings['Interpolation']
            # ic(kernel)

            pts = np.vstack([rad.flatten()*fact, theta.flatten()]).T
            Rmin = pts[:, 0].min()
            Rmax = pts[:, 0].max()
            for i in range(3):

                v = V[i].flatten()
                cond = ~np.isnan(v)
                Pts = pts[cond]
                v = v[cond]
                interp = RBFInterpolator(Pts, v,
                                         smoothing=0.1,
                                         kernel=kernel)
                Pts = np.vstack([rad.flatten(), theta.flatten()]).T

                V[i] = interp(Pts)

                cond = (Pts[:, 0] < Rmin) | (Pts[:, 0] > Rmax)
                V[i][cond] = np.nan

        for i in range(3):
            V[i] = np.append(V[i], [0])

        vtk_dataset = AddArray(vtk_dataset, lbl,
                               [V[0], V[1], V[2]],
                               ['Count', 'Mean', 'RMS'])

    writer = vtkXMLUnstructuredGridWriter()
    outfile = '%s_Stats_%s_%s_P%02d' % (settings['Case'], sw, orientation, plane)
    writer.SetFileName(Path(outfolder, outfile).with_suffix('.vtu'))
    writer.SetInputData(vtk_dataset)
    writer.Write()


def BuildBlocks(radii, ctrs, orientation, vstats, verbose=False):
    """
    BuildBlocks constructs and returns a set of data blocks for given radii, centers, and statistics.
    Parameters:
    radii (numpy.ndarray): Array of radii values.
    ctrs (numpy.ndarray): Array of center values.
    vstats (dict): Dictionary containing statistical data with keys 'Ch. 1 samples', 'Ch. 1 mean', 'Ch. 1 sdev',
                   'Ch. 2 samples', 'Ch. 2 mean', 'Ch. 2 sdev'.
    verbose (bool, optional): If True, displays the vstats dictionary. Default is False.
    Returns:
    list: A list containing:
        - theta (numpy.ndarray): 2D array of angular positions in radians.
        - rad (numpy.ndarray): 2D array of radial positions.
        - Vn (numpy.ndarray): 3D array of sample counts.
        - Vm (numpy.ndarray): 3D array of mean values.
        - Vs (numpy.ndarray): 3D array of standard deviation values.
    """

    if verbose:
        print('BuildBlocks:', orientation)
        ic(vstats)
    src1 = ['Ch. 1 samples', 'Ch. 1 mean', 'Ch. 1 sdev']
    src2 = ['Ch. 2 samples', 'Ch. 2 mean', 'Ch. 2 sdev']
    Src = [src1, src2]

    Vn = []
    Vm = []
    Vs = []
    for src in Src:
        # vn,vm,vs=Collect(vstats,src)
        vn = np.asarray(vstats[src[0]], dtype=float)
        vm = np.asarray(vstats[src[1]], dtype=float)
        vs = np.asarray(vstats[src[2]], dtype=float)
        Vn = np.append(Vn, vn)
        Vm = np.append(Vm, vm)
        Vs = np.append(Vs, vs)
    Vn = np.reshape(Vn, (len(Src), radii.size, ctrs.size))
    Vm = np.reshape(Vm, (len(Src), radii.size, ctrs.size))
    Vs = np.reshape(Vs, (len(Src), radii.size, ctrs.size))

    reps = int(360/settings['Period'])
    angle = []
    for i in range(reps):
        angle = np.append(angle, ctrs+settings['Period']*i)
    Vn = np.tile(Vn, (1, 1, reps))
    Vm = np.tile(Vm, (1, 1, reps))
    Vs = np.tile(Vs, (1, 1, reps))

    Vn = np.dstack((Vn, Vn[:, :, 0]))
    Vm = np.dstack((Vm, Vm[:, :, 0]))
    Vs = np.dstack((Vs, Vs[:, :, 0]))
    angle = np.append(angle, [360])

    offset = 0.0
    if orientation == 'Vu':
        offset = settings['VerticalUpPhaseOffset']
    if orientation == 'Vd':
        offset = settings['VerticalDownPhaseOffset']
    if orientation == 'Hl':
        offset = settings['HorizontalLeftPhaseOffset']
    if orientation == 'Hr':
        offset = settings['HorizontalRightPhaseOffset']
    # print(orientation,offset)
    angle = np.mod(angle+offset, 360)
    angle = np.deg2rad(angle)

    theta, rad = np.meshgrid(angle, radii)

    return [theta, rad, Vn, Vm, Vs]


def PolarPlots(verbose=False, show=False):
    """
    Generate polar plots for LDV measurements.
    Parameters:
    verbose (bool): If True, print detailed information and display data.
    show (bool): If True, display the plots.
    This function reads measurement data from a feather file, processes it, and generates polar plots.
    It checks for the existence of required data files and prints missing files if any.
    The function iterates over specified planes and orientations, processes the data, and generates plots.
    It also exports the processed data to VTK format.
    The function uses the following settings from a global `settings` dictionary:
    - 'OutFolder': Output folder path.
    - 'Case': Case identifier.
    - 'Step': Step size.
    - 'Wslot': Slot width.
    - 'PlaneRange': Range of planes to process.
    - 'Rref': Reference radius.
    - 'Period': Period for interval setting.
    - 'Wleft': Left width for interval setting.
    - 'Wright': Right width for interval setting.
    The function uses the following external functions:
    - SetIntervals: To set intervals for data processing.
    - BuildBlocks: To build data blocks for plotting.
    - PlotVStatsPolar: To plot the polar statistics.
    - ExportToVTKVtu: To export data to VTK format.
    """

    DataPath = Path(settings['OutFolder'], '%s_Stats5.fth' % settings['Case'])
    if not DataPath.exists():
        print('%s does not exist' % DataPath)
        sys.exit(0)

    Data = pd.read_feather(DataPath)
    # display(Data)

    sw = 'S%04dW%04d' % (settings['Step']*100, settings['Wslot']*100)
    datafolder = Path(settings['OutFolder'], 'PolarStats', sw, 'Csv')
    outfolder = Path(datafolder, 'Slice')
    outfolder.mkdir(exist_ok=True)
    # Statfile = [item for item in datafolder.iterdir()]
    # print(Statfile)

    Planes = settings['PlaneRange']
    if Planes == [-1]:
        seen = set()
        Planes = []
        for x in Data['Plane']:
            if x not in seen:
                Planes.append(x)
                seen.add(x)

        Planes.sort()

    print("%d planes:" % len(Planes), Planes)

    Intervals, Ctrs = SetIntervals(settings['Period'], settings['Step'],
                                   settings['Wleft'], settings['Wright'])
    vstat0 = pd.DataFrame([], columns=['Slot', 'Angular position (deg)', 'R', 'X',
                                       'Ch. 1 samples', 'Ch. 1 mean', 'Ch. 1 sdev',
                                       'Ch. 2 samples', 'Ch. 2 mean', 'Ch. 2 sdev'])
    vstat0['Slot'] = Intervals
    vstat0['Angular position (deg)'] = Intervals.left+settings['Wleft']
    vstat0['R'] = np.nan
    vstat0['X'] = np.nan
    vstat0['Ch. 1 samples'] = vstat0['Ch. 2 samples'] = np.nan
    vstat0['Ch. 1 mean'] = vstat0['Ch. 2 mean'] = np.nan
    vstat0['Ch. 1 sdev'] = vstat0['Ch. 2 sdev'] = np.nan

    with tqdm(total=len(Planes), dynamic_ncols=True) as pbar:
        for orientation in ['Vu', 'Vd', 'Hl', 'Hr']:
            cond0 = (Data['Orientation'] == orientation)
            cond1 = (Data['R (mm)'] > 0.0)
            data = Data[cond0 & cond1].copy()
            if len(data) == 0:
                continue

            vstats = pd.DataFrame([], columns=['Slot', 'Angular position (deg)', 'R', 'X',
                                               'Ch. 1 samples', 'Ch. 1 mean', 'Ch. 1 sdev',
                                               'Ch. 2 samples', 'Ch. 2 mean', 'Ch. 2 sdev'])
            count = 0
            data.reset_index(drop=True, inplace=True)
            for irow, row in data.iterrows():
                R = row['R (mm)'] / settings['Rref']
                if R < 1e-6:
                    continue
                X = row['X (mm)'] / settings['Rref']

                statfile = Path(datafolder, '%s_Stats_%s_P%06d' % (settings['Case'], sw, row['Point']))
                statfile = statfile.with_suffix('.csv')

                if statfile.exists():
                    vstat = pd.read_csv(statfile)
                    vstat['R'] = R
                    vstat['X'] = X
                    # ic(irow, len(vstat), len(vstats))
                    if irow == 0:
                        vstats = vstat.copy()
                    else:
                        vstats = pd.concat([vstats, vstat], ignore_index=True)
                    count += 1
                else:
                    if verbose:
                        print('Missing %s (file index %d)' % (statfile, irow))
                    if irow == 0:
                        vstats = vstat0.copy()
                    else:
                        vstats = pd.concat([vstats, vstat0], ignore_index=True)

                if verbose:
                    ic(count)
                    ic(vstats)

            if count < len(data):
                print('%d of %d points missing for %s' % (len(data)-count, len(data), orientation))
            ic(orientation, len(data), count)
            vstats.sort_values(by=['Angular position (deg)', 'R', 'X'], inplace=True)
            vstats.reset_index(drop=True, inplace=True)
            ic(vstats)
            # Block = BuildBlocks(R, Ctrs, orientation, vstats, verbose=verbose)
            # ExportToVTKVtu(Block, plane, orientation, X)

            ic(Ctrs)
            # PlotVStatsPolar(Block, orientation, X)
            for interval in Intervals[::]:
                slice = vstats.loc[vstats['Slot'] == str(interval)].copy()
                angle = np.median(slice['Angular position (deg)'].to_numpy())
                angle_s = np.std(slice['Angular position (deg)'].to_numpy())
                ic(angle, angle_s, len(slice))

                cos = np.cos(np.deg2rad(angle))
                sin = np.sin(np.deg2rad(angle))
                x = slice['X'].to_numpy()
                r = slice['R'].to_numpy()
                v = slice['Ch. 1 mean'].to_numpy()
                pts = np.vstack([x, r]).T

                slice['X'] = x
                slice['Y'] = r*cos
                slice['Z'] = r*sin
                if interval == Intervals[0]:
                    slice.to_csv(Path(outfolder, 'slice_%s.csv' % orientation), index=False)

                np.nan_to_num(v, copy=False)
                cond = ~np.isnan(v)
                ic(np.count_nonzero(cond))
                pts = pts[cond]
                v = v[cond]
                ic(len(v))

                interp = RBFInterpolator(pts, v,
                                         smoothing=0.0,
                                         kernel='linear')
                ic(interp.kernel)

                X = np.linspace(x.min(), x.max(), 50)
                Y = np.linspace(r.min(), r.max(), 20)
                Xi, Yi = np.meshgrid(X, Y)
                xi = Xi.flatten()
                yi = Yi.flatten()
                # xi = x
                # yi = r
                Pts = np.vstack([xi, yi]).T
                V = interp(Pts)

                zi = yi*sin
                yi = yi*cos
                ic(xi.shape, yi.shape, zi.shape, V.shape)

                Slice = pd.DataFrame([xi, yi, zi, V]).T
                Slice.columns = ['X', 'Y', 'Z', 'V']
                ic(Slice)

                slicefile = Path(outfolder, 'Slice_%s_A%03d.csv' % (orientation, int(angle)))
                Slice.to_csv(slicefile, index=False)

                # Block = BuildBlocks(vstats['R'].to_numpy(), ctr, orientation, vstats, verbose=verbose)
                # ExportToVTKVtu(Block, row['Plane'], orientation, X)

# %% [Main]
args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
# print("Args:", args)
if len(args) != 1:
    if Path('SettingsLDV.csv').exists():
        settings_filename = 'SettingsLDV.csv'
        print('Using default settings file')
    else:
        print('Found no default settings file (SettingsLDV.csv)\n')
        sys.exit(0)
else:
    settings_filename = args[0]

global settings
settings = RunSettings(settings_filename)
# display(settings)

PolarPlots(settings['Verbose'], settings['ShowPolarPlots'])
