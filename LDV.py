"""
LDV Data Analysis Script
"""

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
from vtk import vtkMultiBlockDataSet, vtkXMLMultiBlockDataWriter
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


# %% [Various functions]
def myround(x, margin, base=5):
    """
    Rounds a number to the nearest multiple of a specified base, adjusted by a margin.

    Parameters:
    x (float): The number to be rounded.
    margin (float): The margin to adjust the rounding direction.
    base (int, optional): The base to which the number should be rounded. Default is 5.

    Returns:
    float: The rounded number.
    """

    x = x*10+np.sign(margin)*base
    return base*round(x/base)/10


def Check(outfolder, verbose=False):
    """
    Checks the existence and contents of a specified folder.
    If the folder does not exist, it creates the folder and returns True.
    If the folder exists, it checks for files ending with 'fth'. If there are fewer than 6 such files, it returns True.
    Otherwise, it returns False.
    Args:
        outfolder (Path): The path to the folder to check.
    Returns:
        bool: True if the folder was created or contains fewer than 6 'fth' files, False otherwise.
    """

    if not outfolder.exists():
        outfolder.mkdir(parents=True, exist_ok=True)
        return True
    else:
        items = [item for item in outfolder.iterdir() if item.is_file() and item.name.endswith('fth')]
        if verbose:
            display(items)
        return False


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
    RefractiveIndexCorrection;0.9Refractive index correction (1.0 for no correction)

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
    Colormap;viridis;Colormap for polar plots (viridis, plasma, inferno, magma, cividis, jet, rainbow)

    ### VTK output
    Interpolation;thin_plate_spline;linear/thin_plate_spline/cubic/quintic/gaussian/none
    Smoothing;0.0001;Smoothing factor for interpolation (non-zero)

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
            'RefractiveIndexCorrection', 'Smoothing']
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


# %% [I/O functions]
def SaveCSV(folder, file, data, verbose=False):
    """
    Save a DataFrame to a CSV file in the specified folder.

    Parameters:
    folder (Path): The directory where the CSV file will be saved. If it does not exist, it will be created.
    file (str): The name of the CSV file (without the .csv extension).
    data (DataFrame): The data to be saved to the CSV file.
    verbose (bool, optional): If True, the path of the saved file will be displayed. Default is True.

    Returns:
    None
    """

    folder.mkdir(parents=True, exist_ok=True)
    dataout = Path(folder, file+'.csv')
    if verbose:
        display(dataout)
    data.to_csv(dataout)


def SaveXLS(folder, file, data, verbose=False):
    """
    Save the given data to an Excel file in the specified folder.

    Parameters:
    folder (Path): The directory where the Excel file will be saved.
    file (str): The name of the Excel file (without extension).
    data (DataFrame): The data to be saved in the Excel file.
    verbose (bool, optional): If True, display the path of the saved file. Default is True.

    Returns:
    None
    """

    folder.mkdir(parents=True, exist_ok=True)
    dataout = Path(folder, file+'.xlsx')
    if verbose:
        display(dataout)
    writer = pd.ExcelWriter(dataout, engine='openpyxl')
    data.to_excel(writer, sheet_name=('Stats'))
    writer.close()


def SaveFTH(folder, file, data, verbose=False):
    """
    Save a DataFrame to a Feather file with LZ4 compression.

    Parameters:
    folder (Path): The directory where the Feather file will be saved.
    file (str): The name of the Feather file (without extension).
    data (DataFrame): The DataFrame to be saved.
    verbose (bool, optional): If True, display the path of the saved file. Default is True.

    Returns:
    None
    """

    folder.mkdir(parents=True, exist_ok=True)
    dataout = Path(folder, file+'.fth')
    if verbose:
        display(dataout)
    data.to_feather(dataout, compression='lz4')


def SaveMAT(folder, file, data, verbose=False):
    """
    Save data to a .mat file in the specified folder.

    Parameters:
    folder (Path): The directory where the .mat file will be saved.
    file (str): The name of the .mat file (without extension).
    data (DataFrame): The data to be saved, which will be converted to a dictionary of lists.
    verbose (bool, optional): If True, display the path of the saved file. Default is True.

    Returns:
    None
    """

    folder.mkdir(parents=True, exist_ok=True)
    dataout = Path(folder, file+'.mat')
    if verbose:
        display(dataout)
    mydict = data.to_dict('list')
    io.savemat(str(dataout), {'structs': mydict})


def LoadFTH(folder, file):

    v = pd.DataFrame()
    f = Path(folder, file+'.fth')
    if f.exists():
        v = pd.read_feather(f)

    return v


# %% [Read FlowSizer export files]
def ReadStatFiles(datafolder, case, verbose=False):
    """
    Reads statistical files from a specified data folder and case, and returns a concatenated DataFrame.
    Parameters:
    datafolder (Path): The path to the folder containing the data.
    case (str): The specific case to read data for.
    verbose (bool, optional): If True, prints and displays additional information for debugging. Default is False.
    Returns:
    DataFrame: A pandas DataFrame containing the concatenated data from all blocks.
    None: If the number of files in a block does not match the number of entries in the data.
    Notes:
    - The function expects the data files to be in CSV format.
    - The function assumes that each block folder contains a CSV file named after the block.
    - The function also assumes that each block folder contains a subfolder named 'CH1'
    with files whose names end in a 6-digit number.
    - The function will print and display additional information if verbose is set to True.
    """

    if verbose:
        print(datafolder, case)
    Block = [item for item in datafolder.iterdir() if item.is_dir()]
    if len(Block) == 0:
        print('No blocks found')
        sys.exit(0)
    Block = sorted(Block, key=lambda x: x.name)
    if verbose:
        ic(Block)
    Data = pd.DataFrame()
    for block in Block[:]:
        file = Path(block, block.name+'.csv')
        if verbose:
            display(file)
        if file.exists():
            # filenames=pd.read_csv(file,sep=',',skipinitialspace=True,skiprows=lambda x: x % 3,names=['File'])
            # header=pd.read_csv(file,sep=',',skipinitialspace=True,skiprows=1,nrows=1,header=None).values[0]
            # display(filenames)
            # display(header)

            data = pd.read_csv(file, sep=',', skipinitialspace=True)
            if verbose:
                ic(data)
            data.columns = [col.strip('\"') for col in data.columns]
            data.columns = [col.strip() for col in data.columns]
            # data=pd.concat([data,filenames],axis=1)
            data['Block'] = block.name
            Files1 = [item.stem for item in Path(block, 'CH1').iterdir() if item.is_file()]
            Files2 = [item.stem for item in Path(block, 'CH2').iterdir() if item.is_file()]
            if len(Files1) != len(Files2):
                print('Number of files in CH1 does not match number of files in CH2')
                return None

            Files = Files1.copy()
            Files.sort(key=lambda x: int(x[-6:]))
            index = [int(item[-6:]) for item in Files]
            if verbose:
                ic(len(Files))
                ic(len(data))
            if len(Files) > len(data):
                print('Number of files larger than number of entries in block file')
                if any(index[:len(data)]) > len(data):
                    print('Index out of range')
                    return None
                Files = Files[:len(data)]
            if len(Files) < len(data):
                print('More entries in block file than data files')
                if max(index) > len(data):
                    print('Index out of range')
                    return None
                data = data.loc[index]
            data['File'] = Files
            data['Point'] = data['Sequence Number'].apply(lambda x: int(x))
            Data = pd.concat([Data, data])
        else:
            print(f'Block file {file} not found')
            continue

    if verbose:
        ic(Data)
    if Data.empty:
        return None

    Data.sort_values(by=['Block', 'Point'], inplace=True, ascending=True, ignore_index=True)
    # Data.sort_values(by=['X Axis Position','Y Axis Position','Z Axis Position'],
    # inplace=True,ascending=True,ignore_index=True)

    if verbose:
        Display(Data)
        ic(Data.info())

    return Data


def LoadStatFiles(verbose=False):
    """
    Load statistical files, process the data, and save the results.
    This function reads statistical data files from a specified folder,
    processes the data by changing axis orientation and scaling,
    and then saves the processed data to output files. Optionally,
    it can display the first few rows of the processed data.
    Args:
        verbose (bool, optional): If True, displays the first few rows of the processed data.
        Defaults to False.
    Returns:
        None
    """

    # exist = Path(settings['OutFolder'],'%s_Stats0' % settings['Case']+'.xlsx').exists()
    # if exist and not settings['Overwrite']:
    #     return

    Data = ReadStatFiles(settings['DataFolder'], settings['Case'], verbose=verbose)
    if Data is None:
        print('No blocks found')
        sys.exit(0)
    if verbose:
        ic(Data)

    Block = Data['Block'].drop_duplicates().to_list()
    if verbose:
        ic(Block)
    Data = Data.reset_index()

    # Change axis orientation and scaling
    Data['X (mm)'] = Data['X Axis Position'].apply(lambda x: x*settings['AxisScaleFactor'][0])
    Data['Y (mm)'] = Data['Y Axis Position'].apply(lambda x: x*settings['AxisScaleFactor'][1])
    Data['Z (mm)'] = Data['Z Axis Position'].apply(lambda x: x*settings['AxisScaleFactor'][2])
    Data['R (mm)'] = Data['Y (mm)']**2+Data['Z (mm)']**2
    Data['R (mm)'] = Data['R (mm)'].apply(lambda x: np.sqrt(x))

    SaveXLS(settings['OutFolder'], '%s_Stats0' % settings['Case'], Data)
    SaveFTH(settings['OutFolder'], '%s_Stats0' % settings['Case'], Data)

    if verbose:
        ic(Data.head())


# %% [Categorize into planes]
def FindPlanes(verbose=False):
    """
    Find and categorize planes from measurement data.
    This function reads measurement data, performs hierarchical clustering to identify planes,
    and visualizes the results in a 3D scatter plot. The identified planes are saved to files.
    Parameters:
    verbose (bool): If True, displays intermediate data and plots. Default is False.
    Returns:
    None
    Notes:
    - The function reads data from a feather file specified by the 'settings' dictionary.
    - Hierarchical clustering is performed using the 'ward' method.
    - The clustering results are visualized using dendrograms and 3D scatter plots.
    - The identified planes are saved to Excel and feather files.
    """

    Data = pd.read_feather(Path(settings['OutFolder'], '%s_Stats0.fth' % settings['Case']))
    Block = Data['Block'].drop_duplicates().to_list()
    if verbose:
        print('Found %d blocks' % len(Block))
    locations = Data.loc[:, ['X (mm)']]

    # BlockList=['B00']
    # display(Data.loc[Data['Block'].isin(BlockList)])
    # locations=Data.loc[Data['Block'].isin(BlockList),['X (mm)']]

    plt.close(fig='all')

    # Display dendograms
    plt.figure(figsize=(10, 7))
    plt.title("Plane dendogram")
    shc.dendrogram(shc.linkage(locations, method='ward'), no_plot=False)

    # Run clustering
    thresh = 0.25
    cluster = AgglomerativeClustering(n_clusters=None, compute_full_tree=True,
                                      distance_threshold=thresh, compute_distances=True,
                                      metric='euclidean', linkage='complete')

    cluster.fit_predict(locations)
    # display(len(locations),cluster.n_clusters_,cluster.n_clusters,cluster.labels_,cluster.children_,cluster.distances_)
    Cluster = pd.DataFrame([cluster.children_[:, 0], cluster.children_[:, 1], cluster.distances_, cluster.labels_]).T
    Cluster.columns = ['Start', 'End', 'Distance', 'Label']

    # display(len(Cluster),cluster.n_clusters_,cluster.n_clusters,cluster.labels_)
    # Display(Cluster)

    # Categorize
    Cl = pd.DataFrame(columns=['cl', 'X (mm)'])
    for cl in range(cluster.n_clusters_):
        index = Cluster.index[Cluster['Label'] == cl]
        X = Data.loc[locations.index[index], 'X (mm)']
        Cl.loc[len(Cl)] = [cl, np.mean(X)]

    Cl = Cl.astype({"cl": int})
    Cl.sort_values(by=['X (mm)'], inplace=True, ascending=False, ignore_index=True)

    if verbose:
        Display(Cl)
    print('Found %d planes' % len(Cl))

    # Show plane clusters
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    plt.title("Plane clusters")

    clrs = iter(cm.rainbow(np.linspace(0, 1, num=len(Block))))
    for i, block in enumerate(Block):
        # if block.name not in BlockList:
        #     continue
        c = next(clrs)
        c = 'k'

        xs = Data.loc[Data['Block'] == block, 'X (mm)']
        ys = Data.loc[Data['Block'] == block, 'Y (mm)']
        zs = Data.loc[Data['Block'] == block, 'Z (mm)']

        ax.scatter(xs, ys, zs,
                   marker='o', color=c, s=5,
                   fc=c, ec=c,
                   # label=block.name,
                   depthshade=True, alpha=1, zorder=-i)

    clrs = iter(cm.rainbow(np.linspace(0, 1.0, num=cluster.n_clusters_+2)))

    Data['Plane'] = -1
    for p in range(len(Cl)):
        c = next(clrs)
        index = Cluster.index[Cluster['Label'] == Cl.loc[p, 'cl']]

        ncl = len(index)
        if ncl:
            Data.loc[locations.index[index], 'Plane'] = p

            X = Data.loc[locations.index[index], 'X (mm)']
            Y = Data.loc[locations.index[index], 'Y (mm)']
            Z = Data.loc[locations.index[index], 'Z (mm)']
            ax.scatter(X, Y, Z,
                       s=50, ec=c, fc='none',
                       label=('Plane %02d' % p),
                       depthshade=None)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend(fontsize=10)

    if verbose:
        plt.show()

    # print(Data.loc[Data['Plane'].isnull()])
    Data['Plane'] = Data['Plane'].astype(int)
    Data.sort_values(by=['Plane', 'Z (mm)', 'Y (mm)'], inplace=True, ascending=(True, False, False), ignore_index=True)
    if verbose:
        ic(Data)

    SaveXLS(settings['OutFolder'], '%s_Stats1' % settings['Case'], Data)
    SaveFTH(settings['OutFolder'], '%s_Stats1' % settings['Case'], Data)


# %% [Look for repeated points]
def FindRepeats(verbose=False):
    """
    Identifies and visualizes repeated measurement points in a dataset using hierarchical clustering.
    Parameters:
    verbose (bool): If True, displays the dendrogram and 3D scatter plot of clustered points.
    Returns:
    None
    The function performs the following steps:
    1. Reads measurement data from a feather file.
    2. Extracts unique blocks and their corresponding locations.
    3. Displays a dendrogram of the measurement points.
    4. Performs agglomerative hierarchical clustering on the measurement points.
    5. Creates a DataFrame of the clustering results.
    6. Visualizes the clustered points in a 3D scatter plot.
    7. Identifies and marks repeated points in the dataset.
    8. Saves the updated dataset to an Excel and feather file.
    Notes:
    - The function assumes the existence of certain global variables and functions such as `OutFolder`,
    `settings`, `SaveXLS`, and `SaveFTH`.
    - The clustering threshold is set to 0.25.
    - The function uses the 'ward' method for linkage in the dendrogram and 'complete' linkage for clustering.
    """

    Data = pd.read_feather(Path(settings['OutFolder'], '%s_Stats1.fth' % settings['Case']))
    Block = Data['Block'].drop_duplicates().to_list()
    if verbose:
        print('Found %d blocks' % len(Block))
    locations = Data.loc[:, ['X (mm)', 'Y (mm)', 'Z (mm)']]

    plt.close(fig='all')

    # Display dendograms
    plt.figure(figsize=(10, 7))
    plt.title("Repeated points dendogram")
    shc.dendrogram(shc.linkage(locations, method='ward'), no_plot=False)

    # Run clustering
    thresh = 0.25
    cluster = AgglomerativeClustering(n_clusters=None, compute_full_tree=True,
                                      distance_threshold=thresh, compute_distances=True,
                                      metric='euclidean', linkage='complete')

    cluster.fit_predict(locations)
    # display(len(locations),cluster.n_clusters_,cluster.n_clusters,cluster.labels_,cluster.children_,cluster.distances_)
    Cluster = pd.DataFrame([cluster.children_[:, 0], cluster.children_[:, 1], cluster.distances_, cluster.labels_]).T
    Cluster.columns = ['Start', 'End', 'Distance', 'Label']

    # display(len(Cluster),cluster.n_clusters_,cluster.n_clusters,cluster.labels_)
    # Display(Cluster)

    # Show repeated points
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    plt.title("Repeated points")

    clrs = iter(cm.rainbow(np.linspace(0, 1, num=len(Block))))
    for i, block in enumerate(Block):
        c = next(clrs)

        xs = Data.loc[Data['Block'] == block, 'X (mm)']
        ys = Data.loc[Data['Block'] == block, 'Y (mm)']
        zs = Data.loc[Data['Block'] == block, 'Z (mm)']

        ax.scatter(xs, ys, zs, marker='o', color=c, s=20,
                   fc=c, ec='k', label=block,
                   depthshade=True, alpha=1, zorder=-i)

    for cl in range(cluster.n_clusters_):
        index = Cluster.index[Cluster['Label'] == cl]
        ovl = ''
        for ind in index:
            ovl += Data.loc[ind, 'File']+'/'
        ovl = ovl[:-1]
        ncl = len(index)
        if ncl:
            X = Data.loc[locations.index[index], 'X (mm)']
            Y = Data.loc[locations.index[index], 'Y (mm)']
            Z = Data.loc[locations.index[index], 'Z (mm)']

            c = Cluster.loc[index, 'Label'].values.astype(int)
            ax.scatter(X, Y, Z, s=30*ncl, ec='red', fc='none', depthshade=None)
        for ind in index:
            Data.loc[index, 'Repeat'] = ovl

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend(fontsize=10)

    if verbose:
        plt.show()

    SaveXLS(settings['OutFolder'], '%s_Stats2' % settings['Case'], Data)
    SaveFTH(settings['OutFolder'], '%s_Stats2' % settings['Case'], Data)


# %% [Look for corresponding vertical/horizontal points]
def FindMatches(verbose=False):
    """
    Find and visualize matching points in a dataset based on specific conditions.
    Parameters:
    verbose (bool): If True, displays the plots and data. Default is False.
    Returns:
    None
    This function performs the following steps:
    1. Reads data from a feather file.
    2. Identifies points based on specific conditions and assigns orientations.
    3. Adjusts the coordinates of certain points.
    4. Visualizes the points in a 3D scatter plot.
    5. Displays dendrograms for hierarchical clustering.
    6. Performs agglomerative clustering on the points.
    7. Visualizes the clustered points in a 3D scatter plot.
    8. Updates the data with radial matching information.
    9. Sorts the data and saves it to Excel and feather files.
    Note:
    - The function assumes the existence of certain global variables and functions such as `pd`, `Path`, `OutFolder`,
    `settings`, `plt`, `shc`, `AgglomerativeClustering`, `cm`, `SaveXLS`, and `SaveFTH`.
    - The function modifies the input data in place and saves the results to specified output files.
    """

    Data = pd.read_feather(Path(settings['OutFolder'], '%s_Stats2.fth' % settings['Case']))
    locations = Data.loc[:, ['X (mm)', 'Y (mm)', 'Z (mm)']]

    plt.close(fig='all')

    condVu = (np.abs(Data['Y (mm)']) < 0.05) & (Data['Z (mm)'] >= 0)
    condVd = (np.abs(Data['Y (mm)']) < 0.05) & (Data['Z (mm)'] < 0)
    condHl = (np.abs(Data['Z (mm)']) < 0.05) & (Data['Y (mm)'] <= 0)
    condHr = (np.abs(Data['Z (mm)']) < 0.05) & (Data['Y (mm)'] > 0)

    Data.loc[condVu, 'Orientation'] = 'Vu'
    Data.loc[condVd, 'Orientation'] = 'Vd'
    Data.loc[condHl, 'Orientation'] = 'Hl'
    Data.loc[condHr, 'Orientation'] = 'Hr'

    # Display(locations[:40])
    for cond in [condHr, condHl]:
        locations.loc[cond, 'temp'] = locations.loc[cond, 'Y (mm)']
        locations.loc[cond, 'Y (mm)'] = locations.loc[cond, 'Z (mm)']
        locations.loc[cond, 'Z (mm)'] = locations.loc[cond, 'temp']
        locations = locations.drop('temp', axis=1)

    locations.loc[condHl, 'Z (mm)'] = locations.loc[condHl, 'Z (mm)'].apply(lambda x: -x)
    locations.loc[condVd, 'Z (mm)'] = locations.loc[condVd, 'Z (mm)'].apply(lambda x: -x)
    # Display(locations[:40])

    plt.close(fig='all')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Matching points by orientation")

    x = locations.loc[condVu, 'X (mm)']
    y = locations.loc[condVu, 'Y (mm)']
    z = locations.loc[condVu, 'Z (mm)']
    ax.scatter(x, y, z, s=10, fc='none', ec='red', depthshade=None)
    x = locations.loc[condVd, 'X (mm)']
    y = locations.loc[condVd, 'Y (mm)']
    z = locations.loc[condVd, 'Z (mm)']
    ax.scatter(x, y, z, s=60, fc='none', ec='green', depthshade=None)
    x = locations.loc[condHl, 'X (mm)']
    y = locations.loc[condHl, 'Y (mm)']
    z = locations.loc[condHl, 'Z (mm)']
    ax.scatter(x, y, z, s=120, fc='none', ec='blue', depthshade=None)
    x = locations.loc[condHr, 'X (mm)']
    y = locations.loc[condHr, 'Y (mm)']
    z = locations.loc[condHr, 'Z (mm)']
    # ax.scatter(x,y,z,s=180,fc='none',ec='black',depthshade=None)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    # ax.set_zlim(0,10)

    # Display dendograms
    plt.figure(figsize=(10, 7))
    plt.title("Matching point dendogram")
    shc.dendrogram(shc.linkage(locations, method='ward'))

    # Run clustering
    thresh = 0.25
    cluster = AgglomerativeClustering(n_clusters=None, compute_full_tree=True,
                                      distance_threshold=thresh, compute_distances=True,
                                      metric='euclidean', linkage='complete')

    cluster.fit_predict(locations)
    # display(len(locations),cluster.n_clusters_,cluster.n_clusters,cluster.labels_,cluster.children_,cluster.distances_)
    Cluster = pd.DataFrame([cluster.children_[:, 0], cluster.children_[:, 1], cluster.distances_, cluster.labels_]).T
    Cluster.columns = ['Start', 'End', 'Distance', 'Label']

    # display(len(Cluster), cluster.n_clusters_, cluster.n_clusters, cluster.labels_)
    # Display(Cluster)

    # Show matching points
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    plt.title("Matching points")

    xs = locations['X (mm)']
    ys = locations['Y (mm)']
    zs = locations['Z (mm)']

    ax.scatter(xs, ys, zs, marker='o', color='k', s=5,
               fc='k', ec='k',
               depthshade=True, alpha=1, zorder=0)

    # colors = iter(cm.rainbow(np.linspace(0, 1, num=cluster.n_clusters_)))
    for cl in range(cluster.n_clusters_):
        index = Cluster.index[Cluster['Label'] == cl]
        ovl = ''
        for ind in index:
            ovl += Data.loc[ind, 'File']+'/'
        ovl = ovl[:-1]
        ncl = len(index)
        if ncl:
            # color = next(colors)
            X = locations.loc[locations.index[index], 'X (mm)']
            Y = locations.loc[locations.index[index], 'Y (mm)']
            Z = locations.loc[locations.index[index], 'Z (mm)']
            # c = Cluster.loc[index, 'Label'].values.astype(int)
            ax.scatter(X, Y, Z, s=30, ec='red', fc='none',
                       depthshade=None)
        for ind in index:
            Data.loc[index, 'Radial matching'] = ovl

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

    if verbose:
        plt.show()

    Data.sort_values(by=['Plane', 'Point'], inplace=True, ascending=True, ignore_index=True)
    if verbose:
        ic(Data)

    SaveXLS(settings['OutFolder'], '%s_Stats3' % settings['Case'], Data)
    SaveFTH(settings['OutFolder'], '%s_Stats3' % settings['Case'], Data)


# %% [Collect velocity components]
def ReadCsv(block, src, Ch, file):
    """
    Reads a CSV file and processes its data based on the provided parameters.
    Parameters:
    block (str): The block name or identifier used to construct the file path.
    src (int): The source channel number used to filter and process the data.
    Ch (list of int or None): A list of channel numbers to be processed. If None, no additional channels are processed.
    file (str): The name of the CSV file to be read.
    Returns:
    pandas.DataFrame: A DataFrame containing the processed data from the CSV file.
    Notes:
    - The function constructs the file path using the provided block and file parameters.
    - It reads the CSV file into a DataFrame, skipping the first row and using the second row as the header.
    - If Ch is not None, it processes each channel in Ch by aligning its time data with the source channel's time data.
    - The function drops rows where all elements are NaN and returns the cleaned DataFrame.
    """

    file = Path(settings['DataFolder'], block, 'CH%d' % src, file)
    # ic(file, src)

    data = pd.DataFrame()
    try:
        data = pd.read_csv(file, sep=',', skipinitialspace=True, skiprows=[0],
                           header=0, nrows=None, low_memory=False)
    except FileNotFoundError:
        print('FileNotFoundError: %s' % file)
        return data
    except pd.errors.ParserError:
        print('ParserError: %s' % file)
        return data
    except pd.errors.EmptyDataError:
        print('EmptyDataError: %s' % file)
        return data

    # print(data.columns)
    data.rename(columns={"Velocity Ch.%d (m/sec)" % src: "Velocity Ch. %d (m/sec)" % src}, inplace=True)
    data.rename(columns={"Velocity Ch. %d (msec)" % src: "Velocity Ch. %d (m/sec)" % src}, inplace=True)
    data.rename(columns={"Analog Extern Ch 2": "Analog Extern Ch. 2"}, inplace=True)
    data.rename(columns={"AnalogExtern Time 2 (sec)": "Analog Extern Time 2 (sec)"}, inplace=True)
    # print(data.columns)

    columns = ['Time Ch. %d (sec)' % src,
               'Velocity Ch. %d (m/sec)' % src,
               'RMR%d Time (sec)' % src,
               'RMR%d (degree)' % src,
               ]
    for col in columns:
        if col not in data.columns:
            raise ValueError('Column \'%s\' not found in %s' % (col, file.name))

    for ch in Ch:
        if ch == -1:
            continue
        ext = 'Analog Extern Ch. %d' % ch
        if ext in data.columns:
            # Display(data[:50])
            # display(data[5340:])
            # print(file)
            # print((~np.isnan(data['Time Ch. 1 (sec)'])).sum())
            ach = (data[['Analog Extern Ch. %d' % ch, 'Analog Extern Time %d (sec)' % ch]])
            dif = ach[ach['Analog Extern Time %d (sec)' % ch].isin(data['Time Ch. %d (sec)' % src])]
            dif = dif.drop_duplicates(subset='Analog Extern Time %d (sec)' % ch, keep='first')
            dif.sort_values(by=['Analog Extern Time %d (sec)' % ch], inplace=True)

            # dif.dropna(inplace=True)
            dif = dif.reset_index(drop=False)
            # display(dif)
            data['Analog Extern Time %d (sec)' % ch] = dif['Analog Extern Time %d (sec)' % ch]
            data['Analog Extern Ch. %d' % ch] = dif['Analog Extern Ch. %d' % ch]
        else:
            raise ValueError('Column "%s" not found in %s' % (ext, file))

    data.dropna(how='all', inplace=True)
    # display(data.info())

    return data


def LoadPointData(point, verbose=False):
    """
    Loads and processes point data from a CSV file.
    Args:
        point (dict): A dictionary containing 'File' and 'Block' keys with corresponding values.
        verbose (bool, optional): If True, displays the initial data for both channels. Defaults to False.
    Returns:
        tuple: A tuple containing two pandas DataFrames (ch1, ch2) with processed data.
    The function performs the following steps:
    1. Constructs the file name from the 'File' value in the point dictionary.
    2. Reads data from the CSV file for two channels using the ReadCsv function.
    3. Displays the initial data for both channels if verbose is True.
    4. Converts columns with object data types to numeric, coercing errors to NaN.
    5. Drops rows where all elements are NaN.
    6. Checks for non-numeric values in the original data and prints a message if any are found.
    """

    file = point['File'].values[0]+'.csv'
    block = point['Block'].values[0]

    ch1 = ReadCsv(block, 1, settings['ExternalChannels'], file)
    ch2 = ReadCsv(block, 2, settings['ExternalChannels'], file)

    if verbose:
        ic(ch1)
        ic(ch2)

    if not ch1.empty:
        count01 = len(ch1)
        cols = ch1.columns[ch1.dtypes.eq('object')]
        ch1[cols] = ch1[cols].apply(pd.to_numeric, errors='coerce')
        ch1.dropna(how='all', inplace=True)
        ch1 = ch1.loc[ch1['Velocity Ch. 1 (m/sec)'].notnull()]
        count11 = len(ch1)
        if count01 != count11:
            print(">>> Non-numeric value in %s CH1" % file)

    if not ch2.empty:
        count02 = len(ch2)
        cols = ch2.columns[ch2.dtypes.eq('object')]
        ch2[cols] = ch2[cols].apply(pd.to_numeric, errors='coerce')
        ch2.dropna(how='all', inplace=True)
        ch2 = ch2.loc[ch2['Velocity Ch. 2 (m/sec)'].notnull()]
        count12 = len(ch2)
        if count02 != count12:
            print(">>> Non-numeric value in %s CH2" % file)

    return [ch1, ch2]


def ChannelToComponent(currow, ch, v, verbose):
    """
    Adjusts the orientation and velocity of measurement channels based on the given row's orientation.
    Parameters:
    currow (DataFrame): A DataFrame row containing metadata about the file, plane, and orientation.
    ch (list of DataFrame): A list containing two DataFrames, each representing a measurement channel.
    v (list of DataFrame): A list containing two DataFrames to store the adjusted measurement data.
    Returns:
    tuple: A tuple containing two DataFrames with the adjusted measurement data.
    The function performs the following operations:
    1. Prints the file, plane, and orientation information from the current row.
    2. Adjusts the 'RMR1 (degree)' and 'RMR2 (degree)' columns in the measurement channels based on the orientation.
    3. Inverts the 'Velocity Ch. 2 (m/sec)' values in the second measurement channel.
    4. Concatenates the adjusted measurement data into the provided DataFrames `v`.
    Note:
    - The 'Orientation' column in `currow` can have values 'Hl', 'Vd', or 'Hr',
    which correspond to offsets of 90, 180, and 270 degrees respectively.
    - If the DataFrames in `v` are empty, they are initialized with the adjusted measurement data.
    """

    if verbose:
        print("File:", currow['File'].values[0],
              "Plane", currow['Plane'].values[0],
              "Orientation", currow['Orientation'].values[0])

    offset = 0.0
    if currow['Orientation'].values[0] == 'Hl':
        offset = 90.0
    if currow['Orientation'].values[0] == 'Vd':
        offset = 180.0
    if currow['Orientation'].values[0] == 'Hr':
        offset = 270.0

    if len(ch[0]):
        ch[0]['RMR1 (degree)'] = ch[0]['RMR1 (degree)'].apply(lambda x: np.mod(x+offset, 360))
        if len(v[0]) == 0:
            v[0] = ch[0][v[0].columns].copy()
        else:
            v[0] = pd.concat([v[0], ch[0]], axis=0, join='inner').dropna(how='all')
    if len(ch[1]):
        ch[1]['RMR2 (degree)'] = ch[1]['RMR2 (degree)'].apply(lambda x: np.mod(x+offset, 360))
        ch[1]['Velocity Ch. 2 (m/sec)'] = ch[1]['Velocity Ch. 2 (m/sec)'].apply(lambda x: -x)
        if len(v[1]) == 0:
            v[1] = ch[1][v[1].columns].copy()
        else:
            v[1] = pd.concat([v[1], ch[1]], axis=0, join='inner').dropna(how='all')

    return v


def GetPointVelocity(currow, v, verbose):
    """
    Calculate the point velocity from the given data.
    Args:
        currow (DataFrame): The current row of data containing 'File' and 'Orientation' information.
        v (list): A list of velocity components.
        verbose (bool): If True, display additional information for debugging purposes.
    Returns:
        list: A list containing the processed velocity components for channels 1 and 2.
    Notes:
        - The function loads point data using the `LoadPointData` function.
        - If `verbose` is True, it displays the head of the data for both channels.
        - The function converts the channel data to velocity components using `ChannelToComponent`.
        - It processes the velocity data by removing outliers based on the mean and standard deviation.
    """

    # print("Currow", currow['File'], currow['Orientation'])
    Ch = LoadPointData(currow, verbose)
    # print("ch1:", len(Ch[0]))
    # print("ch2:", len(Ch[1]))
    if verbose:
        Display(Ch[0].head())
        Display(Ch[1].head())
    # SaveXLS(settings['OutFolder'], 'ch1', Ch[0])
    # SaveXLS(settings['OutFolder'], 'ch2', Ch[1])

    V = ChannelToComponent(currow, Ch, v, verbose)

    Lbl = ['Velocity Ch. 1 (m/sec)', 'Velocity Ch. 2 (m/sec)']
    for i, lbl in enumerate(Lbl):

        v = V[i].dropna(how='all')
        if len(v) < 1:
            continue
        mean = np.mean(v[lbl])
        std_deviation = np.std(v[lbl], axis=0)
        V[i] = v.drop(v[np.abs(v[lbl]-mean) > settings['nStd'] * std_deviation]. index)

    return V


# %% [Compute stats]
def DataStats(data, file, labels, v):
    """
    Computes and updates statistical metrics for a given dataset.
    Parameters:
    data (pandas.DataFrame): The DataFrame containing the data to be updated.
    file (str): The file identifier to locate the specific row in the DataFrame.
    labels (list of str): A list of column names where the computed statistics will be stored.
                          The list should contain at least 5 elements corresponding to:
                          [count, mean, standard deviation, skewness, kurtosis].
    v (array-like): The data values for which the statistics are to be computed.
    Returns:
    pandas.DataFrame: The updated DataFrame with the computed statistics.
    """

    mean = np.mean(v)
    # variance = np.var(v)
    std_deviation = np.std(v, axis=0)
    skewness = np.mean(((v - mean) / std_deviation) ** 3)
    flatness = np.mean(((v - mean) / std_deviation) ** 4)

    data.loc[data['File'] == file, labels[0]] = len(v)
    data.loc[data['File'] == file, labels[1]] = mean
    data.loc[data['File'] == file, labels[2]] = std_deviation
    data.loc[data['File'] == file, labels[3]] = skewness
    data.loc[data['File'] == file, labels[4]] = flatness

    return data


def Diff(data):
    """
    Calculate the difference between sample data and valid velocity count,
    and the difference between various velocity statistics.
    Parameters:
    data (DataFrame): A pandas DataFrame containing the following columns:
        - 'Samples Ch. 1', 'Samples Ch. 2': Sample data for channels 1 and 2.
        - 'Valid Vel.Count Ch. 1', 'Valid Vel.Count Ch. 2': Valid velocity count for channels 1 and 2.
        - 'V_Mean Ch. 1 (m/sec)', 'V_Mean Ch. 2 (m/sec)': Mean velocity for channels 1 and 2.
        - 'V_RMS Ch. 1 (m/sec)', 'V_RMS Ch. 2 (m/sec)': RMS velocity for channels 1 and 2.
        - 'V_Skewness Ch. 1 ()', 'V_Skewness Ch. 2 ()': Skewness of velocity for channels 1 and 2.
        - 'V_Flatness Ch. 1 ()', 'V_Flatness Ch. 2 ()': Flatness of velocity for channels 1 and 2.
        - 'Velocity Mean Ch. 1 (m/sec)', 'Velocity Mean Ch. 2 (m/sec)': Mean velocity for channels 1 and 2.
        - 'Velocity RMS Ch. 1 (m/sec)', 'Velocity RMS Ch. 2 (m/sec)': RMS velocity for channels 1 and 2.
        - 'Velocity Skewness Ch. 1 ()', 'Velocity Skewness Ch. 2 ()': Skewness of velocity for channels 1 and 2.
        - 'Velocity Flatness Ch. 1 ()', 'Velocity Flatness Ch. 2 ()': Flatness of velocity for channels 1 and 2.
    Returns:
    DataFrame: The input DataFrame with additional columns:
        - 'diff Samples Ch. 1', 'diff Samples Ch. 2':
        Difference between sample data and valid velocity count for channels 1 and 2.
        - 'Diff Velocity Mean Ch. 1 (m/sec)', 'Diff Velocity Mean Ch. 2 (m/sec)':
        Difference in mean velocity for channels 1 and 2.
        - 'Diff Velocity RMS Ch. 1 (m/sec)', 'Diff Velocity RMS Ch. 2 (m/sec)':
        Difference in RMS velocity for channels 1 and 2.
        - 'Diff Velocity Skewness Ch. 1 ()', 'Diff Velocity Skewness Ch. 2 ()':
        Difference in skewness of velocity for channels 1 and 2.
        - 'Diff Velocity Flatness Ch. 1 ()', 'Diff Velocity Flatness Ch. 2 ()':
        Difference in flatness of velocity for channels 1 and 2.
    """

    stats = [['Mean', 'm/sec'], ['RMS', 'm/sec'], ['Skewness', ''], ['Flatness', '']]

    for i in range(2):
        data[f'diff Samples Ch. {i+1}'] = data[f'Samples Ch. {i+1}'] - data[f'Valid Vel.Count Ch. {i+1}']
        for stat, unit in stats:
            data[f'Diff Velocity {stat} Ch. {i+1} ({unit})'] = (data[f'V_{stat} Ch. {i+1} ({unit})'] -
                                                                data[f'Velocity {stat} Ch. {i+1} ({unit})'])

    return data


def ExtractVelocityField(Data, verbose=False):
    """
    Extracts the velocity field from the given data and updates the Data DataFrame.

    Parameters:
    Data (pd.DataFrame): The main DataFrame containing all the data.
    data (pd.DataFrame): The DataFrame containing the specific data to be processed.
    concat (bool, optional): If True, concatenates the velocity data. Defaults to True.
    verbose (bool, optional): If True, prints detailed information during processing. Defaults to False.

    Returns:
    pd.DataFrame: The updated Data DataFrame with the processed velocity field data.
    """

    # Display(data)
    # display(data.info())
    Lbls_Stats = []
    Lbls_Time = []
    for i in range(2):
        lbls = [f'Samples Ch. {i+1}',
                f'V_Mean Ch. {i+1} (m/sec)', f'V_RMS Ch. {i+1} (m/sec)',
                f'V_Skewness Ch. {i+1} ()', f'V_Flatness Ch. {i+1} ()']
        Lbls_Stats.append(lbls)

        lbls = [f'Time Ch. {i+1} (sec)', f'RMR{i+1} (degree)', f'Velocity Ch. {i+1} (m/sec)']
        Lbls_Time.append(lbls)

    with tqdm(total=len(Data), dynamic_ncols=True, desc=Data.loc[Data.index[0], 'File']) as pbar:

        for _, Row in Data.iterrows():
            outFolder = Path(settings['OutFolder'], 'VField')
            outFolder.mkdir(parents=True, exist_ok=True)

            Repeat = Row['Repeat'].split('/')
            Matching = Row['Radial matching'].split('/')
            if verbose:
                print("Repeat:", Repeat)
                print("Matching:", Matching)

            v1 = pd.DataFrame(columns=Lbls_Time[0])
            v2 = pd.DataFrame(columns=Lbls_Time[1])
            V = [v1, v2]

            for repeat in Repeat:
                row = Data.loc[Data['File'] == repeat]
                file = row['File'].values[0]
                block = row['Block'].values[0]
                file1 = Path(settings['DataFolder'], block, 'CH1', file).with_suffix('.csv')
                file2 = Path(settings['DataFolder'], block, 'CH2', file).with_suffix('.csv')
                if not file1.exists() or not file2.exists():
                    if verbose:
                        print("Data not found:", file)
                    Data.loc[Data['File'] == repeat, 'Processed'] = -1
                    continue

                if row['Processed'].values[0]:
                    if verbose:
                        display('Skipped repeat %s' % repeat)
                    continue

                if verbose:
                    display('Repeat: '+repeat)

                V = GetPointVelocity(row, V, verbose)
                Data.loc[Data['File'] == repeat, 'Processed'] = Repeat.index(repeat)+1

                for i in range(2):
                    if len(V[i]):
                        Data = DataStats(Data, repeat, Lbls_Stats[i],
                                         V[i]['Velocity Ch. %d (m/sec)' % (i+1)])

            if verbose:
                print(Row['File'])
                print('V1')
                display(V[0].info())
                print('V2')
                display(V[1].info())
            for i in range(2):
                V[i] = V[i].reset_index(drop=True)

            V = pd.concat([V[0], V[1]], axis=1)
            V = V.dropna(how='all')

            SaveFTH(outFolder, Row['File'], V, verbose=False)
            if settings['ExportMat']:
                SaveMAT(outFolder, Row['File'], V, verbose=False)
            if settings['ExportCsv']:
                SaveCSV(outFolder, Row['File'], V, verbose=False)

            pbar.desc = Row['File']
            pbar.update()

    return Data


def ComputeStats(verbose=False):
    """
    Compute and save statistical analysis of velocity field data.
    This function reads velocity field data from a Feather file, processes it to extract
    the velocity field, computes differences, and saves the results in both Feather and
    Excel formats. Optionally, it prints the maximum absolute values of axial, radial,
    and tangential velocities for different orientations.
    Args:
        verbose (bool): If True, prints the maximum absolute values of velocities for
                        different orientations. Default is False.
    Returns:
        None
    """

    Data = pd.read_feather(Path(settings['OutFolder'], '%s_Stats3.fth' % settings['Case']))
    Data['Processed'] = 0

    Data = ExtractVelocityField(Data, verbose=False)
    SaveFTH(Path(settings['OutFolder'], '.'), '%s_Stats4' % settings['Case'], Data, verbose=False)
    SaveXLS(Path(settings['OutFolder'], '.'), '%s_Stats4' % settings['Case'], Data, verbose=False)

    Data = Data.loc[Data['Processed'] == 1]
    Data = Data.reset_index(drop=True)

    Data = Diff(Data)

    # display(Data)

    SaveFTH(Path(settings['OutFolder'], '.'), '%s_Stats5' % settings['Case'], Data, verbose=False)
    SaveXLS(Path(settings['OutFolder'], '.'), '%s_Stats5' % settings['Case'], Data, verbose=False)

    if verbose:
        print('Axial Vu', np.abs(Data.loc[Data['Orientation'] == 'Vu', 'V_Mean Ch. 2 (m/sec)']).max())
        print('Axial Vd', np.abs(Data.loc[Data['Orientation'] == 'Vd', 'V_Mean Ch. 2 (m/sec)']).max())
        print('Axial Hl', np.abs(Data.loc[Data['Orientation'] == 'Hl', 'V_Mean Ch. 2 (m/sec)']).max())
        print('Axial Hr', np.abs(Data.loc[Data['Orientation'] == 'Hr', 'V_Mean Ch. 2 (m/sec)']).max())
        print('Radial Vu', np.abs(Data.loc[Data['Orientation'] == 'Vu', 'V_Mean Ch. 1 (m/sec)']).max())
        print('Radial Vd', np.abs(Data.loc[Data['Orientation'] == 'Vd', 'V_Mean Ch. 1 (m/sec)']).max())
        print('Tangential Hl', np.abs(Data.loc[Data['Orientation'] == 'Hl', 'V_Mean Ch. 1 (m/sec)']).max())
        print('Tangential Hr', np.abs(Data.loc[Data['Orientation'] == 'Hr', 'V_Mean Ch. 1 (m/sec)']).max())


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


def SetPhase(v):
    """
    Adjusts the phase of the given DataFrame `v` based on the orientation specified in `row`.
    Parameters:
    v (pd.DataFrame): DataFrame containing phase data to be adjusted.
    row (pd.Series): Series containing the 'Orientation' key to determine the phase offset.
    Returns:
    pd.DataFrame: The DataFrame `v` with adjusted phase values.
    The function uses the 'Orientation' value from `row` to determine the appropriate phase offset
    from the `settings` dictionary.
    It then applies this offset to the 'RMR1 (degree)' and 'RMR2 (degree)' columns of the DataFrame `v`,
    using modulo operation with the given `period`.
    """

    period = settings['Period']

    for i in range(2):
        val = 'RMR%d (degree)' % (i+1)
        v.loc[pd.notna(v[val]), val].apply(lambda x: np.mod(x, period))

    return (v)


# %% [Plot point distributions]
def PlotPointData(v, row):
    """
    Plots point data from a DataFrame based on the given row's orientation and file information.
    Parameters:
    v (pd.DataFrame): The DataFrame containing the point data to be plotted. The first three columns are considered as
    one set of data (v1) and the remaining columns as another set (v2).
    row (pd.Series): A Series containing metadata for the plot. It should have the keys 'Orientation' and 'File'.
    The function creates a figure with three subplots:
    1. Scatter plot of the first two columns of v1 and v2.
    2. Scatter plot of the first and third columns of v1.
    3. Scatter plot of the first and third columns of v2.
    The title of the plot is derived from the 'File' and 'Orientation' values in the row.
    """

    v1 = v[v.columns[:3]]
    v2 = v[v.columns[3:]]
    # display(v1)
    # display(v2)
    orientation = row['Orientation']
    title = '%s' % row['File'] + ' ' + ('(Up)' if orientation == 'Vu' else '(Down)' if orientation == 'Vd'
                                        else '(Left)' if orientation == 'Hl' else '(Right)')

    fig = plt.figure(figsize=(18, 12), facecolor='white')
    ax = fig.add_subplot(311)
    ax.scatter(v1[v1.columns[1]], v1[v1.columns[0]], c='r', s=0.1)
    ax.scatter(v2[v2.columns[1]], v2[v2.columns[0]], c='g', s=0.1)
    ax.set_xlabel('')
    ax.set_ylabel(v1.columns[0])
    # ax.set_xlim(0, 540)
    plt.title(title, size=24)

    ax = fig.add_subplot(312)
    ax.scatter(v1[v1.columns[1]], v1[v1.columns[2]], c='r', s=0.1)
    ax.set_xlabel(v1.columns[1])
    ax.set_ylabel(v1.columns[2])
    # ax.set_xlim(0, 540)

    ax = fig.add_subplot(313)
    ax.scatter(v2[v2.columns[1]], v2[v2.columns[2]], c='g', s=0.1)
    ax.set_xlabel(v2.columns[1])
    ax.set_ylabel(v2.columns[2])
    # ax.set_xlim(0, 540)

    plt.tight_layout()
    plt.show()


# %% [Plot phase distribution]
def PlotPhaseDistribution(v, row):
    """
    Plots the phase distribution of given data.
    Parameters:
    V (DataFrame): A pandas DataFrame containing the phase data with columns 'RMR1 (degree)' and 'RMR2 (degree)'.
    Row (Series): A pandas Series containing metadata for the plot, including 'Orientation' and 'File'.
    The function generates a scatter plot of 'RMR1 (degree)' vs 'RMR2 (degree)' and titles the plot based on the
    'Orientation' and 'File' values in the row.
    """

    orientation = row['Orientation']
    title = ('Up' if orientation == 'Vu' else 'Down' if orientation == 'Vd'
             else 'Left' if orientation == 'Hl' else 'Right')
    title = '%s (%s)' % (row['File'], title)

    if len(v) > 0:
        fig = plt.figure(figsize=(12, 10), facecolor='white')
        ax = fig.add_subplot(111)

        theta = v['RMR1 (degree)']
        phi = v['RMR2 (degree)']

        print(len(theta), len(phi))
        ax.scatter(theta, phi, s=0.05)
        ax.set_xlabel('RMR1', size=12)
        ax.set_ylabel('RMR2', size=12)
        ax.set_title(title, size=12)
        plt.show()


# %% [Slot distribution]
def SlotDistribution(v, intervals):
    """
    Calculate the slot distribution for given periods and intervals.

    Parameters:
    period (int): The period for the slot distribution.
    intervals (list of pd.Interval): List of intervals to be used for distribution.
    v (pd.DataFrame): DataFrame containing the data to be analyzed.
        The DataFrame is expected to have at least 6 columns,
        where the first three columns correspond to the first set of data and the next three columns correspond
        to the second set of data.

    Returns:
    list of pd.DataFrame: A list containing two DataFrames,
        each corresponding to the slot distribution for the two sets of data.
        Each DataFrame includes the original data along with the calculated slot distributions.
    """

    period = settings['Period']

    a1 = v[v.columns[:3]]
    a2 = v[v.columns[3:]]
    A = [a1, a2]

    V = []
    for i, col in enumerate(['RMR1 (degree)', 'RMR2 (degree)']):
        a = A[i]
        # print('Input:', a, len(a))
        if not len(a):
            continue
        Inter = pd.DataFrame(columns=intervals)

        for inter in intervals[:]:
            # display(inter)

            # In=pd.DataFrame()
            # In[inter] = a[col].apply(lambda x: x in inter)

            In = pd.DataFrame()
            In[inter] = np.where(np.logical_and(a[col] >= inter.left, a[col] < inter.right), True, False)
            # In[inter] = a[col][cond], axis=1)
            if inter.left < 0:
                linter = pd.Interval(inter.left+period, period, closed=settings['IntervalClosed'])
                # display(linter)
                In[linter] = np.where(np.logical_and(a[col] >= linter.left, a[col] < linter.right), True, False)
            if inter.right > period:
                rinter = pd.Interval(0, inter.right-period, closed=settings['IntervalClosed'])
                # display(rinter)
                In[rinter] = np.where(np.logical_and(a[col] >= rinter.left, a[col] < rinter.right), True, False)

            # display(In)
            # Inter[inter] = In.fillna(False).select_dtypes(include=['bool']).sum(axis=1)
            Inter[inter] = In.sum(axis=1)
        # display(a, Inter)
        a = pd.concat([a, Inter], axis=1)
        # df_.sort_values(by=[col], inplace=True, ascending=True, ignore_index=True)

        nsamples = pd.DataFrame(columns=a.columns)
        for inter in intervals[:]:
            nsamples.at[0, inter] = a[inter].sum()
        nsamples.at[0, col] = nsamples[intervals].sum(axis=1).values[0].astype(int)
        # display(nsamples)
        # a=pd.concat([a, nsamples], ignore_index=True)
        # display(a)
        if i == 0:
            V = [a.copy()]
        else:
            V.append(a.copy())

    return V


# %% [Velocity stats]
def GetVStats(v, intervals):
    """
    Calculate velocity statistics for given intervals.
    Parameters:
    v (list of pd.DataFrame): List of DataFrames containing velocity data for channels.
    intervals (pd.IntervalIndex): IntervalIndex object defining the intervals for analysis.
    Returns:
    pd.DataFrame: DataFrame containing the calculated statistics for each interval.
        Columns include:
        - 'Slot': Interval slots.
        - 'Angular position (deg)': Angular position in degrees.
        - 'Ch. 1 samples': Number of samples in Channel 1.
        - 'Ch. 1 mean': Mean velocity in Channel 1.
        - 'Ch. 1 sdev': Standard deviation of velocity in Channel 1.
        - 'Ch. 2 samples': Number of samples in Channel 2.
        - 'Ch. 2 mean': Mean velocity in Channel 2.
        - 'Ch. 2 sdev': Standard deviation of velocity in Channel 2.
    """

    vstats = pd.DataFrame([], columns=['Slot',
                                       'Ch. 1 samples', 'Ch. 1 mean', 'Ch. 1 sdev',
                                       'Ch. 2 samples', 'Ch. 2 mean', 'Ch. 2 sdev'])
    vstats['Slot'] = intervals
    vstats['Angular position (deg)'] = intervals.left+settings['Wleft']
    for i, col in enumerate(['Velocity Ch. 1 (m/sec)', 'Velocity Ch. 2 (m/sec)']):
        var = v[i]
        icol = i*3+1
        # print('Input:', vstats.columns[icol], var, len(var))

        for j, inter in enumerate(intervals[:]):

            var_ = var.loc[var[inter] > 0][col]
            vstats.iloc[j, icol] = var[inter].sum()
            vstats.iloc[j, icol+1] = np.mean(var_)
            vstats.iloc[j, icol+2] = np.std(var_)

    column = vstats.pop('Angular position (deg)')
    vstats.insert(1, 'Angular position (deg)', column)

    return vstats


# %% [Plot velocity stats]
def PlotVStats(ctrs, vstats, row):

    orientation = row['Orientation']
    orientation = ('Up' if orientation == 'Vu' else 'Down' if orientation == 'Vd' else
                   'Left' if orientation == 'Hl' else 'Right')
    title1 = '%s (%s, Ch. 1)' % (row['File'], orientation)
    title2 = '%s (%s, Ch. 2)' % (row['File'], orientation)
    Src = [['Ch. 1 samples', 'Ch. 1 mean', 'Ch. 1 sdev'],
           ['Ch. 2 samples', 'Ch. 2 mean', 'Ch. 2 sdev']]
    Lbls = [[r'$\theta$', r'$\overline{V}_1\ (m/s)$', r'$\widetilde{V}_1\ (m/s)$', title1],
            [r'$\theta$', r'$\overline{V}_2\ (m/s)$', r'$\widetilde{V}_2\ (m/s)$', title2]]

    alpha = 0.8
    for src, lbl in list(zip(Src, Lbls)):

        if vstats[src[0]].isnull().all():
            continue

        Vn = np.asarray(vstats[src[0]], dtype=float)
        vm = np.asarray(vstats[src[1]], dtype=float)
        vs = np.asarray(vstats[src[2]], dtype=float)
        Vm = vm[~np.isnan(vm)]
        Vs = vs[~np.isnan(vs)]

        V = [Vn, Vm, Vs]

        fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True,
                                figsize=(12, 8), facecolor='white')

        axs[-1].set_xlabel(lbl[0], size=14)
        axs[0].set_ylabel('Counts', size=14)
        axs[1].set_ylabel(lbl[1], size=14)
        axs[2].set_ylabel(lbl[2], size=14)

        for i, v in enumerate(V):
            bars = []
            for x, h in zip(ctrs, v):
                rect = Rectangle((x, 0), settings['Wslot'], h)
                bars.append(rect)
            norm = mcolors.Normalize(vmin=np.min(v), vmax=np.max(v))
            colors = cm.jet(norm(v))
            pc = PatchCollection(bars, facecolor=colors, alpha=alpha, edgecolor='black')
            axs[i].add_collection(pc)

            axs[i].autoscale(enable=True, axis='both', tight=False)

        plt.suptitle(lbl[3], size=20)
        plt.show()


# %% [Single point analysis]
def SinglePointAnalysis(row, show=True):
    """
    Perform single point analysis on the given row of data.
    This function loads velocity field data, processes it through various stages,
    and optionally displays plots at each stage. The final statistics are saved
    to a CSV file.
    Parameters:
    row (pd.Series): A pandas Series containing the data for a single point.
    show (bool): A flag to indicate whether to display plots. Default is True.
    Returns:
    None
    """

    plt.close('all')

    sw = 'S%04dW%04d' % (settings['Step']*100, settings['Wslot']*100)
    outfolder = Path(settings['OutFolder'], 'PolarStats', sw, 'Csv')
    outfolder.mkdir(parents=True, exist_ok=True)
    outfile = '%s_Stats_%s_P%06d' % (settings['Case'], sw, row['Point'])

    exist = Path(outfolder, outfile+'.csv').exists()
    if exist and not settings['Overwrite']:
        return

    v0 = LoadFTH(Path(settings['OutFolder'], 'VField'), row['File'])
    # print(v0,len(v0))
    if v0.empty or len(v0) == 0:
        return

    if show:
        PlotPointData(v0, row)
    v1 = SetPhase(v0)
    if show:
        PlotPhaseDistribution(v1, row)

    Intervals, Ctrs = SetIntervals(settings['Period'], settings['Step'], settings['Wleft'], settings['Wright'])
    v2 = SlotDistribution(v1, Intervals)

    # display(((v2[0]).iloc[:,3:]).sum().sum())
    # display(((v2[1]).iloc[:,3:]).sum().sum())
    VStats = GetVStats(v2, Intervals)
    # display(VStats,VStats['Ch. 1 samples'].sum(), VStats['Ch. 2 samples'].sum())
    if show:
        PlotVStats(Ctrs, VStats, row)

    VStats.to_csv(Path(outfolder, outfile).with_suffix('.csv'), index=False)


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


def ExportToVTKVtm(vtk_block, iorient, block, plane, orientation, sw, X):
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

    outfolder = Path(settings['OutFolder'], 'PolarStats', sw, 'Vtk')
    outfolder.mkdir(exist_ok=True)

    theta, rad, Vn, Vm, Vs = block

    Z = X / settings['Rref']

    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    points = np.vstack([x.flatten(), y.flatten()]).T
    R = np.sqrt(points[:, 0]**2 + points[:, 1]**2)

    orientation = ('Up' if orientation == 'Vu' else 'Down' if orientation == 'Vd' else
                   'Left' if orientation == 'Hl' else 'Right')
    Lbl = ['Radial velocity (%s)' % orientation, 'Axial velocity (%s)' % orientation]
    if orientation in ['Left', 'Right']:
        Lbl = ['Tangential velocity (%s)' % orientation, 'Axial velocity (%s)' % orientation]

    for k, lbl in enumerate(Lbl):

        V = [Vn[k, :, :], Vm[k, :, :], Vs[k, :, :]]

        Pts = points.copy()
        if orientation in ['Left', 'Right']:
            Pts = Pts*settings['RefractiveIndexCorrection']
        Points = np.vstack((Pts, (0, 0)))
        tri = Delaunay(Points)
        vtk_dataset = MakeVtkDataset(tri, Z)

        r = np.sqrt(Pts[:, 0]**2 + Pts[:, 1]**2)
        Rmin = r.min()
        Rmax = r.max()
        ic(Rmin, Rmax)

        kernel = settings['Interpolation']
        ic(kernel)
        ic(settings['Smoothing'])
        for i in range(3):

            ic(lbl, i)
            v = V[i].flatten()
            if kernel == 'none':
                V[i] = v.copy()
            else:
                cond = ~np.isnan(v)
                pts = Pts[cond]
                v = v[cond]

                ic(v, len(v), pts, len(pts))
                interp = RBFInterpolator(pts, v,
                                         smoothing=settings['Smoothing'],
                                         kernel=kernel)

                V[i] = interp(points)

            R = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            cond = (R < Rmin) | (R > Rmax)
            V[i][cond] = np.nan

        for i in range(3):
            V[i] = np.append(V[i], [0])

        vtk_dataset = AddArray(vtk_dataset, lbl,
                               [V[0], V[1], V[2]],
                               ['Count', 'Mean', 'RMS'])

    vtk_block.SetBlock(iorient, vtk_dataset)
    vtk_block.GetMetaData(iorient).Set(vtkMultiBlockDataSet.NAME(), orientation)
    
    return vtk_block


def ExportToVTKVtu(block, plane, orientation, sw, X):
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

    outfolder = Path(settings['OutFolder'], 'PolarStats', sw, 'Vtk')
    outfolder.mkdir(exist_ok=True)

    theta, rad, Vn, Vm, Vs = block

    Z = X / settings['Rref']
    orientation = ('Up' if orientation == 'Vu' else 'Down' if orientation == 'Vd' else
                   'Left' if orientation == 'Hl' else 'Right')

    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    points = np.vstack([x.flatten(), y.flatten()]).T

    if orientation in ['Left', 'Right']:
        points = points*settings['RefractiveIndexCorrection']

    Points = np.vstack((points, (0, 0)))
    tri = Delaunay(Points)
    vtk_dataset = MakeVtkDataset(tri, Z)

    Lbl = ['Radial velocity (%s)' % orientation, 'Axial velocity (%s)' % orientation]
    if orientation in ['Left', 'Right']:
        Lbl = ['Tangential velocity (%s)' % orientation, 'Axial velocity (%s)' % orientation]

    kernel = settings['Interpolation']
    # ic(kernel)
    # ic(settings['Smoothing'])
    for k, lbl in enumerate(Lbl):

        V = [Vn[k, :, :], Vm[k, :, :], Vs[k, :, :]]

        for i in range(3):

            # ic(lbl, i)
            v = V[i].flatten()
            if kernel == 'none':
                V[i] = v.copy()
            else:
                cond = ~np.isnan(v)
                pts = points[cond]
                v = v[cond]

                # ic(v, len(v), pts, len(pts))
                interp = RBFInterpolator(pts, v,
                                         smoothing=settings['Smoothing'],
                                         kernel=kernel)

                V[i] = interp(points)

        for i in range(3):
            V[i] = np.append(V[i], [np.nan])

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
    angle = np.deg2rad(angle)*settings['RotationSign']

    theta, rad = np.meshgrid(angle, radii)

    return [theta, rad, Vn, Vm, Vs]


# %% [Polar plot)]
def PlotVStatsPolar(block, plane, orientation, sw, X, show=False):
    """
    Plots velocity statistics on a polar plot.
    Parameters:
    block (tuple): A tuple containing theta, rad, Vn, Vm, Vs arrays.
    row (dict): A dictionary containing metadata for the plot, including 'Orientation' and 'Plane'.
    X (float): The X coordinate value for the plot.
    show (bool, optional): If True, displays the plot. Defaults to False.
    Returns:
    None
    This function generates polar plots for velocity statistics (radial, tangential, and axial velocities)
    based on the provided data and settings. The plots are saved as PNG files in the specified output folder.
    """

    plt.close('all')

    RadiusLimits = settings['PolarPlotRadiusLimits']

    outfolder = Path(settings['OutFolder'], 'PolarStats', sw, 'Plots')
    outfolder.mkdir(exist_ok=True)

    theta, rad, Vn, Vm, Vs = block

    alpha = 1.0
    cmap = mpl.colormaps[settings['Colormap']]

    if 'V' in orientation:
        Range_V1 = [settings['Vr_samp_range'], settings['Vr_mean_range'], settings['Vr_sdev_range']]
        lbl1 = [r'Count', r'$\overline{V}_r\ (m/s)$', r'$\widetilde{V}_r\ (m/s)$', 'Radial velocity']
    else:
        Range_V1 = [settings['Vt_samp_range'], settings['Vt_mean_range'], settings['Vt_sdev_range']]
        lbl1 = [r'Count', r'$\overline{V}_t\ (m/s)$', r'$\widetilde{V}_t\ (m/s)$', 'Tangential velocity']
    Range_V2 = [settings['Va_samp_range'], settings['Va_mean_range'], settings['Va_sdev_range']]
    lbl2 = [r'Count', r'$\overline{V}_a\ (m/s)$', r'$\widetilde{V}_a\ (m/s)$', 'Axial velocity']
    Lbl = [lbl1, lbl2]

    src1 = ['Ch. 1 samples', 'Ch. 1 mean', 'Ch. 1 sdev', Range_V1]
    src2 = ['Ch. 2 samples', 'Ch. 2 mean', 'Ch. 2 sdev', Range_V2]
    Src = [src1, src2]

    orientation = ('Up' if orientation == 'Vu' else 'Down' if orientation == 'Vd' else
                   'Left' if orientation == 'Hl' else 'Right')
    overlap = (1 - settings['Step'] / settings['Wslot'])*100  # overlap between adjacent slots

    for k, src, lbl in list(zip(range(len(Src)), Src, Lbl)):

        V = [Vn[k, :, :], Vm[k, :, :], Vs[k, :, :]]

        fig = plt.figure(figsize=(18, 8), facecolor='white')
        figtitle = '%s (%s)' % (lbl[-1], orientation)
        figtitle = figtitle + '\n' + r'Plane %d | X=%.0f mm' % (plane, X)
        figtitle = figtitle + r' | Slot spacing=%0.2f$^\circ$' % settings['Step']
        figtitle = figtitle + r' | Slot width=%0.2f$^\circ$' % settings['Wslot']
        figtitle = figtitle + r' | Overlap=%.0f%%' % (overlap)
        fig.suptitle(figtitle, fontsize=20)
        gs = fig.add_gridspec(nrows=2, ncols=3, figure=fig, left=0.05, right=0.95, top=0.9,
                              width_ratios=[0.3, 0.3, 0.3], height_ratios=[0.95, 0.05])

        for i, v in enumerate(V):

            vma = np.ma.masked_invalid(v)
            cond = (rad < RadiusLimits[0]) | (rad > RadiusLimits[1])
            vma = np.ma.masked_array(vma, mask=cond)

            vmin = src[-1][i][0]
            vmax = src[-1][i][1]
            if settings['Autoscale']:
                if i == 1:
                    vmin = np.mean(vma) - 2*np.std(vma)
                else:
                    vmin = np.min(vma)
                vmax = np.mean(vma) + 2*np.std(vma)
            # print(vmin, vmax)

            fact = 1.0
            if orientation in ['Left', 'Right']:
                fact = settings['RefractiveIndexCorrection']
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            ax = fig.add_subplot(gs[0, i], projection='polar')
            ax.set_facecolor('white')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(settings['RotationSign'])
            ax.pcolormesh(theta, rad*fact, vma, cmap=cmap, norm=norm,
                          edgecolors='none', linewidth=0.1,
                          shading='gouraud', alpha=alpha)
            ax.set_xlabel(lbl[i], size=20)
            # ax.set_xlim(0,2*np.pi)
            ax.set_ylim(0, 1.25)
            ax.grid(which='both')
            cax = plt.subplot(gs[1, i])
            ColorbarBase(ax=cax, cmap=cmap, norm=norm, orientation='horizontal')
            # break

        outfile = '%s_Stats_%s_%s%s_P%02d' % (settings['Case'], sw,
                                              lbl[-1].split(' ')[0], orientation, plane)
        plt.savefig(Path(outfolder, outfile).with_suffix('.png'),
                    facecolor=fig.get_facecolor(), dpi=300)

        if show:
            plt.pause(0.1)
        # break


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
                                       'Ch. 1 samples', 'Ch. 1 mean', 'Ch. 1 sdev',
                                       'Ch. 2 samples', 'Ch. 2 mean', 'Ch. 2 sdev'])
    vstat0['Slot'] = Intervals
    vstat0['Angular position (deg)'] = Intervals.left+settings['Wleft']
    vstat0['Ch. 1 samples'] = vstat0['Ch. 2 samples'] = np.nan
    vstat0['Ch. 1 mean'] = vstat0['Ch. 2 mean'] = np.nan
    vstat0['Ch. 1 sdev'] = vstat0['Ch. 2 sdev'] = np.nan

    with tqdm(total=len(Planes), dynamic_ncols=True, desc="Plane P%02d" % Planes[0]) as pbar:
        for plane in Planes:
            cond0 = (Data['Plane'] == plane)

            # vtk_block = vtkMultiBlockDataSet()
            # vtk_block.SetNumberOfBlocks(2)

            start_time = timeit.default_timer()
            for iorient, orientation in enumerate(['Vu', 'Vd', 'Hl', 'Hr']):
                cond1 = (Data['Orientation'] == orientation)
                data = Data[cond0 & cond1].copy()

                if len(data) == 0:
                    continue

                data.sort_values(by=['R (mm)'], inplace=True, ignore_index=True)
                R = data['R (mm)'] / settings['Rref']
                # R = np.sort(R)
                X = np.mean(data.loc[data['Plane'] == plane, 'X (mm)'])

                if np.mean(R) < 1e-6:
                    print('Skipping plane %d (%s): R = %f' % (plane, orientation, np.mean(R)))
                    continue
                # print('R:', len(R), data['Orientation'], data['Point'], len(data))
                vstats = pd.DataFrame([], columns=['Slot', 'Angular position (deg)',
                                                   'Ch. 1 samples', 'Ch. 1 mean', 'Ch. 1 sdev',
                                                   'Ch. 2 samples', 'Ch. 2 mean', 'Ch. 2 sdev'])
                count = 0
                data.reset_index(drop=True, inplace=True)
                for irow, row in data.iterrows():
                    plane = row['Plane']

                    statfile = Path(datafolder, '%s_Stats_%s_P%06d' % (settings['Case'], sw, row['Point']))
                    statfile = statfile.with_suffix('.csv')
                    # ic(irow, statfile)
                    if statfile.exists():
                        vstat = pd.read_csv(statfile)
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
                    print('%d of %d points missing in plane %d for %s' % (len(data)-count, len(data),
                                                                          plane, orientation))
                Block = BuildBlocks(R, Ctrs, orientation, vstats, verbose=verbose)
                PlotVStatsPolar(Block, plane, orientation, sw, X, show)
                ExportToVTKVtu(Block, plane, orientation, sw, X)

                # vtk_block = ExportToVTKVtm(vtk_block, iorient, Block, plane, orientation, sw, X)

            # writer = vtkXMLMultiBlockDataWriter()
            # outfolder = Path(settings['OutFolder'], 'PolarStats', sw, 'Vtk')
            # outfile = '%s_Stats_%s_P%02d' % (settings['Case'], sw, plane)
            # writer.SetFileName(Path(outfolder, outfile).with_suffix('.vtm'))
            # writer.SetInputData(vtk_block)
            # writer.Write()

            if verbose:
                print('Plane %02d: %f seconds' % (plane, timeit.default_timer() - start_time))

            pbar.desc = "P%02d" % plane
            pbar.update()


# %% [Phase analysis]
def PhaseAnalysis(verbose=False, show=False):
    """
    Perform phase analysis on measurement data.
    Parameters:
    verbose (bool): If True, display the original and filtered data. Default is False.
    show (bool): If True, show the results of the single point analysis. Default is False.
    Returns:
    None
    This function reads measurement data from a feather file, filters it based on specified
    conditions, and performs single point analysis on the filtered data. The progress of the
    analysis is displayed using a progress bar. If the verbose parameter is set to True, the
    original and filtered data are displayed. If the show parameter is set to True, the results
    of the single point analysis are shown.
    """

    DataPath = Path(settings['OutFolder'], '%s_Stats5.fth' % settings['Case'])
    if not DataPath.exists():
        print('%s does not exist' % DataPath)
        sys.exit(0)
    Data = pd.read_feather(DataPath)

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

    cond0 = Data['Plane'].apply(lambda x: x in Planes)
    cond1 = Data['R (mm)'].apply(lambda x: x >= settings['RadiusRange'][0])
    cond2 = Data['R (mm)'].apply(lambda x: x <= settings['RadiusRange'][-1])
    data = Data[cond0 & cond1 & cond2]
    if len(data) == 0:
        print('No data for phase analysis')
        return

    if verbose:
        display(Data)
        display(data)

    with tqdm(total=len(data), dynamic_ncols=True,
              desc="%s P%02d" % (data.loc[data.index[0], 'File'],
                                 data.loc[data.index[0], 'Plane'])) as pbar:
        for _, row in data.iterrows():
            file = row['File']
            plane = row['Plane']

            start_time = timeit.default_timer()

            SinglePointAnalysis(row, show)

            elapsed = timeit.default_timer() - start_time

            pbar.desc = "%s P%02d [%.1fs]" % (file, plane, elapsed)
            pbar.update()


# %% [Main]
def authorship():
    return print(dedent("""LDV Data Analysis Script

    Author:
    - Francisco Alves Pereira (francisco.alvespereira@cnr.it)

    Version: 0.9.3
    Date: 2025-03-04

    Description:
    This script performs data analysis on Laser Doppler Velocimetry (LDV) measurements.
    It includes functions for loading, processing, and visualizing LDV data,
    as well as generating statistical summaries and exporting results to various formats.

    Dependencies:
    - Python 3.8+
    - numpy
    - pandas
    - matplotlib
    - scipy
    - scikit-learn
    - vtk
    - tqdm
    - openpyxl

    License:
    This script is released under the MIT License.

    Usage:
    python LDV.py <settings_filename>
    """))


def GenerateDatabase(verbose=False):
    LoadStatFiles(verbose)
    FindPlanes(verbose)
    FindRepeats(verbose)
    FindMatches(verbose)
    ComputeStats(verbose)


args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
# print("Args:", args)
if len(args) != 1:
    if Path('SettingsLDV.csv').exists():
        settings_filename = 'SettingsLDV.csv'
        print('Using default settings file')
    else:
        print('Found no default settings file (SettingsLDV.csv)\n')
        authorship()
        sys.exit(0)
else:
    settings_filename = args[0]

global settings
settings = RunSettings(settings_filename)
# display(settings)

if settings['GenerateDatabase']:
    print('Generating database')
    GenerateDatabase(settings['Verbose'])

if settings['PhaseAnalysis']:
    print('Running phase analysis')
    PhaseAnalysis(settings['Verbose'], settings['ShowPhasePlots'])

if settings['GeneratePolarPlots']:
    print('Generating polar plots')
    PolarPlots(settings['Verbose'], settings['ShowPolarPlots'])

plt.close('all')
print('Finished')

# %% [End]
