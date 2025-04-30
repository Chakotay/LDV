
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
        ic(id, pt)
        x, y = pt
        pts.InsertPoint(id, [x, y, z[id]])
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


def Slice(nx=10, ny=10, verbose=False):

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
    vstat0 = pd.DataFrame([], columns=['Slot', 'Angular position (deg)',
                                       'R', 'X',
                                       'Ch. 1 samples', 'Ch. 1 mean', 'Ch. 1 sdev',
                                       'Ch. 2 samples', 'Ch. 2 mean', 'Ch. 2 sdev'])
    vstat0['Slot'] = Intervals
    vstat0['Angular position (deg)'] = Intervals.left+settings['Wleft']
    vstat0['R'] = np.nan
    vstat0['X'] = np.nan
    vstat0['Ch. 1 samples'] = vstat0['Ch. 2 samples'] = np.nan
    vstat0['Ch. 1 mean'] = vstat0['Ch. 2 mean'] = np.nan
    vstat0['Ch. 1 sdev'] = vstat0['Ch. 2 sdev'] = np.nan

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

            cond0 = (Data['Orientation'] == orientation)
            cond1 = (Data['R (mm)'] > 0.0)
            for plane in Planes[:5:]:

                cond2 = (Data['Plane'] == plane)
                data = Data[cond0 & cond1 & cond2].copy()
                if len(data) == 0:
                    continue
                ic(orient, len(Data[cond0 & cond1]), len(Data[cond0 & cond1 & cond2]))

                count = 0
                data.reset_index(drop=True, inplace=True)
                for irow, row in data.iterrows():
                    R = row['R (mm)'] / settings['Rref']
                    if R < 1e-6:
                        continue
                    Xp = row['X (mm)'] / settings['Rref']

                    statfile = Path(datafolder, '%s_Stats_%s_P%06d' % (settings['Case'], sw, row['Point']))
                    statfile = statfile.with_suffix('.csv')

                    if statfile.exists():
                        vstat = pd.read_csv(statfile)
                        vstat['R'] = R
                        vstat['X'] = Xp
                        # ic(irow, len(vstat), len(vstats))
                        if count == 0:
                            vstatp = vstat.copy()
                        else:
                            vstatp = pd.concat([vstatp, vstat], ignore_index=True)
                        count += 1
                    else:
                        print('File not found:', statfile)
                        continue
                        # if verbose:
                        #     print('Missing %s (file index %d)' % (statfile, irow))
                        # if irow == 0:
                        #     vstatp = vstat0.copy()
                        # else:
                        #     vstatp = pd.concat([vstatp, vstat0], ignore_index=True)

                    # ic(count)
                    # ic(vstatp, len(vstatp))

                if count < len(data):
                    print('%d of %d points missing for %s (%s) in plane %d' % (len(data)-count, len(data), orient, orientation, plane))
                ic(orient, plane, len(data), count)

                vstatp['Y'] = vstatp['R'] # * np.cos(np.deg2rad(vstatp['Angular position (deg)']))
                vstatp['Z'] = vstatp['R'] # * np.sin(np.deg2rad(vstatp['Angular position (deg)']))
                vstatp.sort_values(by=['Angular position (deg)', 'R', 'X'], inplace=True)
                vstatp.reset_index(drop=True, inplace=True)

                slicefile = Path(outfolder, 'Slice_%s_P%03d' % (orient, plane))
                vstatp.to_csv(slicefile.with_suffix('.csv'), index=False)

                rad = vstatp['R'].to_numpy()
                theta = vstatp['Angular position (deg)'].to_numpy()

                pts = np.vstack([theta, rad]).T

                # Create interpolation grid
                Rad = np.unique(rad)
                Theta = np.unique(theta)
                ic(Rad, Theta)
                ic(len(Rad), len(Theta))

                X = np.linspace(Theta.min(), Theta.max(), len(Theta)*2)
                Y = np.linspace(Rad.min(), Rad.max(), len(Rad)*2)

                # X = theta
                # Y = rad
                Xi, Yi = np.meshgrid(X, Y)
                xi = Xi.flatten()
                yi = Yi.flatten()
                Pts = np.vstack([xi, yi]).T

                vstatpi = pd.DataFrame()
                vstatpi['R'] = yi
                vstatpi['Y'] = yi # * np.cos(np.deg2rad(xi))
                vstatpi['Z'] = yi # * np.sin(np.deg2rad(xi))
                vstatpi['X'] = Xp

                # Use original grid
                Pts = pts
                vstatpi = vstatp.copy()

                for i in [1, 2]:
                    if orient in ['Up', 'Down']:
                        comp = 'radial' if i == 1 else 'axial'
                    else:
                        comp = 'tangential' if i == 1 else 'axial'

                    vn = 'Count %s velocity (%s)' % (comp, orient)
                    vm = 'Mean %s velocity (%s)' % (comp, orient)
                    vs = 'RMS %s velocity (%s)' % (comp, orient)
                    Var = [[f'Ch. {i} samples', vn], [f'Ch. {i} mean', vm], [f'Ch. {i} sdev', vs]]
                    for var in Var:
                        ch, name = var

                        v = vstatp[ch].to_numpy()
                        np.nan_to_num(v, copy=False)
                        cond = ~np.isnan(v)
                        # ic(np.count_nonzero(cond))
                        pts = pts[cond]
                        v = v[cond]
                        # ic(ch, len(v))

                        interp = RBFInterpolator(pts, v,
                                                 smoothing=1,
                                                 kernel=settings['Interpolation'])
                        # ic(interp.kernel)

                        V = interp(Pts)
                        # ic(V.shape)

                        if i == 1 and var[0] == 'Ch. 1 samples':
                            vstatpi[name] = V.T
                        else:
                            vstatpi = pd.concat([vstatpi, pd.DataFrame({name: V.T})], axis=1)
                        # ic(vstatpi[name])

                if plane == 0:
                    vstats[orient] = vstatpi.copy()
                else:
                    vstats[orient] = pd.concat([vstats[orient], vstatpi], ignore_index=True)

                slicefile = Path(outfolder, 'Slice_%s_Interp_P%03d' % (orient, plane))
                vstatpi.to_csv(slicefile.with_suffix('.csv'), index=False)

            if len(vstats[orient]) == 0:
                vstats.pop(orient)
                continue
            ic(plane, vstats[orient], vstats.keys())

        xmin = ymin = 1e6
        xmax = ymax = -1e6
        for orient in vstats.keys():
            ic(orient, len(vstats[orient]))
            if xmin > vstats[orient]['X'].min():
                xmin = vstats[orient]['X'].min()
            if xmax < vstats[orient]['X'].max():
                xmax = vstats[orient]['X'].max()
            if ymin > vstats[orient]['R'].min():
                ymin = vstats[orient]['R'].min()
            if ymax < vstats[orient]['R'].max():
                ymax = vstats[orient]['R'].max()
        X = np.linspace(xmin, xmax, nx)
        Y = np.linspace(ymin, ymax, ny)
        ic(X, Y)
        dX = (xmax - xmin)/(len(X)-1)
        dY = (ymax - ymin)/(len(Y)-1)
        ic(dX, dY)

        Xi, Yi = np.meshgrid(X, Y)
        xi = Xi.flatten()
        yi = Yi.flatten()

        for interval in Intervals[::]:
            new_slice = True
            for orient in vstats.keys():
                # ic('key:', orient)

                slice = vstats[orient].loc[vstats[orient]['Slot'] == str(interval)].copy()

                angle = np.median(slice['Angular position (deg)'].to_numpy())
                angle_s = np.std(slice['Angular position (deg)'].to_numpy())
                # ic(angle, angle_s, len(slice))

                cos = np.cos(np.deg2rad(angle))
                sin = np.sin(np.deg2rad(angle))
                x = slice['X'].to_numpy()
                r = slice['R'].to_numpy()
                pts = np.vstack([x, r]).T

                #slice['X'] = x
                #slice['Y'] = r*cos
                #slice['Z'] = r*sin
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

                for i in [1, 2]:
                    if orient in ['Up', 'Down']:
                        comp = 'radial' if i == 1 else 'axial'
                    else:
                        comp = 'tangential' if i == 1 else 'axial'

                    vn = 'Count %s velocity (%s)' % (comp, orient)
                    vm = 'Mean %s velocity (%s)' % (comp, orient)
                    vs = 'RMS %s velocity (%s)' % (comp, orient)
                    Var = [[f'Ch. {i} samples', vn], [f'Ch. {i} mean', vm], [f'Ch. {i} sdev', vs]]
                    for var in Var:
                        ch, name = var

                        v = slice[ch].to_numpy()
                        np.nan_to_num(v, copy=False)
                        cond = ~np.isnan(v)
                        # ic(np.count_nonzero(cond))
                        pts = pts[cond]
                        v = v[cond]
                        # ic(ch, len(v))

                        interp = RBFInterpolator(pts, v,
                                                 smoothing=0.0,
                                                 kernel=settings['Interpolation'])
                        # ic(interp.kernel)

                        V = interp(Pts)
                        # ic(V.shape)

                        Slice[name] = V.T
                        # ic(Slice)

            Slice['Velocity magnitude (Up)'] = np.sqrt(
                Slice['Mean axial velocity (Up)']**2 +
                Slice['Mean radial velocity (Up)']**2 +
                Slice['Mean tangential velocity (Left)']**2)
            Slice['Velocity magnitude (Left)'] = np.sqrt(
                Slice['Mean axial velocity (Left)']**2 +
                Slice['Mean radial velocity (Up)']**2 +
                Slice['Mean tangential velocity (Left)']**2)

            Vort = {'Mean axial vorticity': ['Mean tangential velocity (Left)', 'Mean radial velocity (Up)'],
                    'Mean radial vorticity (Up)': ['Mean axial velocity (Up)', 'Mean tangential velocity (Left)'],
                    'Mean radial vorticity (Left)': ['Mean axial velocity (Left)', 'Mean tangential velocity (Left)'],
                    'Mean tangential vorticity (Up)': ['Mean radial velocity (Up)', 'Mean axial velocity (Up)'],
                    'Mean tangential vorticity (Left)': ['Mean radial velocity (Up)', 'Mean axial velocity (Left)']}
            for vort in Vort.keys():
                # ic(vort, Vort[vort], Vort[vort][0], Vort[vort][1], dX, dY)
                u = Slice[Vort[vort][0]].to_numpy()
                v = Slice[Vort[vort][1]].to_numpy()
                # ic(u.shape, v.shape)
                u = np.reshape(u, (ny, nx))
                v = np.reshape(v, (ny, nx))
                # ic(u.shape, v.shape)
                du = np.gradient(u, dY, axis=0, edge_order=2)
                dv = np.gradient(v, dX, axis=1, edge_order=2)
                Slice[vort] = (dv - du).flatten()
            Slice['Vorticity magnitude (Up)'] = np.sqrt(
                Slice['Mean axial vorticity']**2 +
                Slice['Mean radial vorticity (Up)']**2 +
                Slice['Mean tangential vorticity (Up)']**2)
            Slice['Vorticity magnitude (Left)'] = np.sqrt(
                Slice['Mean axial vorticity']**2 +
                Slice['Mean radial vorticity (Left)']**2 +
                Slice['Mean tangential vorticity (Left)']**2)

            Slice['TKE (Up)'] = (
                Slice['RMS axial velocity (Up)']**2 +
                Slice['RMS radial velocity (Up)']**2 +
                Slice['RMS tangential velocity (Left)']**2)/2

            Slice['TKE (Left)'] = (
                Slice['RMS axial velocity (Left)']**2 +
                Slice['RMS radial velocity (Up)']**2 +
                Slice['RMS tangential velocity (Left)']**2)/2

            # Vol = Slice.copy()
            # ic(Vol['Angular position (deg)'])
            # for rep in np.arange(1, reps):
            #     rot = Vol['Angular position (deg)'] + settings['Period']*rep
            #     cos = np.cos(np.deg2rad(rot))
            #     sin = np.sin(np.deg2rad(rot))
            #     Vol['Y'] = Vol['R']*cos
            #     Vol['Z'] = Vol['R']*sin
            #     Slice = pd.concat([Slice, Vol], ignore_index=True)

            Slice.sort_values(by=['X', 'R', 'Angular position (deg)'], inplace=True)
            slicefile = Path(outfolder, 'Slice_A%03d' % int(angle))
            Slice.to_csv(slicefile.with_suffix('.csv'), index=False)

            # Pts = np.vstack([xi, yi]).T
            # tri = Delaunay(Pts)
            # Z = Slice['Z'].to_numpy()
            # vtk_dataset = MakeVtkDataset(tri, Slice['Z'].to_numpy().T)
            # ic(vtk_dataset.GetNumberOfPoints())
            # axial = [var for var in Slice.columns if 'axial velocity' in var.lower()]
            # radial = [var for var in Slice.columns if 'radial velocity' in var.lower()]
            # tangential = [var for var in Slice.columns if 'tangential velocity' in var.lower()]
            # magnitude = [var for var in Slice.columns if 'velocity magnitude' in var.lower()]
            # vorticity = [var for var in Slice.columns if 'vorticity' in var.lower()]
            # turbulence = [var for var in Slice.columns if 'tke' in var.lower()]
            # Var = [[axial, 'Axial velocity'],
            #        [radial, 'Radial velocity'],
            #        [tangential, 'Tangential velocity'],
            #        [magnitude, 'Velocity magnitude'],
            #        [vorticity, 'Vorticity'],
            #        [turbulence, 'TKE']]
            # for var in Var:
            #     cnames, lbl = var
            #     vtk_dataset = AddArray(vtk_dataset, lbl,
            #                            [slice for slice in Slice[cnames].to_numpy().T], cnames)
            # writer = vtkXMLUnstructuredGridWriter()
            # writer.SetFileName(slicefile.with_suffix('.vtu'))
            # writer.SetInputData(vtk_dataset)
            # writer.Write()

            # Block = BuildBlocks(vstats['R'].to_numpy(), ctr, orientation, vstats, verbose=verbose)
            # ExportToVTKVtu(Block, row['Plane'], orientation, X)


# %% [Main]
args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
ic(args, len(args))

if len(args) != 3:
    print('Usage: %s <settings file> <nx> <ny>' % Path(__file__).name)
    sys.exit(0)
else:
    settings_filename = args[0]
    nx = int(args[1])
    ny = int(args[2])
global settings
settings = RunSettings(settings_filename)
# display(settings)

Slice(nx, ny, settings['Verbose'])
