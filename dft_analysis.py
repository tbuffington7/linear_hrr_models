import numpy as np
import pdb
import pandas as pd
import yaml
from scipy.spatial import distance_matrix

def rename_DFT_columns(df):
    """
    Takes a raw dataframe from the graphtec and renames the columns to more convenient names. 
    Note the J indicates the DFT is from Jan's set up i.e. DFTJ01

    inputs
    ------
    df: pandas.DataFrame
        A dataframe of raw DFT measurements from the graphtec

    returns
    -------
    updated_df: pandas.DataFrame
        The same dataframe with updated column names

    """
    updated_df = df.copy()
    for column in df.columns:
        if 'Sensor_' in column:
            updated_df = updated_df.rename(columns={column: 'DFTJ'+column.replace('Sensor_','')[0].zfill(2)})
        elif 'Time' in column:
            updated_df = updated_df.rename(columns={column: 'Time'})
    return updated_df

def normalize_vector(vector):
    """
    Normalizes a vector so that the magnitude is one, but has the same direction

    inputs
    ------
    vector: array_like
        A vector to be normalized

    returns
    -------
    norm_vector: array_like
        The normalized vector
    """
    vector = np.array(vector)
    magnitude = np.sqrt(np.sum(vector**2))
    norm_vector = vector/magnitude
    return norm_vector

def flame_height_calc(Q_dot, burner_width):
    """
    Uses a correlation to estimate flame height

    inputs
    ------
    Q_dot: float
        The total (not radiative) heat release rate in kW
    burner_width: float
        The width of the burner in m

    returns
    -------
    zf: float
        The estimated flame height in m
    """

    # Calculates fire origin of moving fire
    zf = (0.23*Q_dot**(2/5) - 1.02*burner_width) # Quinterre Eq 10.50b
    return zf

def distance_calc(vector1, vector2):
    """
    Computes the l2 distance between two vectors. Note that the two vectors must have the same length

    inputs
    ------
    vector1: array_like
        The first vector
    vector2: array_like
        the second vector

    returns:
    -------
    distance: float
        The l2 norm of the two vectors
    """
    distance = np.sqrt(np.sum((vector1-vector2)**2))
    return distance

def virtual_origin_calc(Q_dot, burner_loc, burner_width):
    """
    Computes the virual origin based on the flame height for a burner

    inputs
    ------
    Q_dot: float
        The total heat release rate
    burner_loc: array_like
        The location, i.e. (x,y,z) coordinates of the burner in m. The z is the top of the burner.
    burner_width: float
        The width of the burner in m

    returns:
    -------
    virtual_source: array_like
        The location of the virtual source in (x,y,z) coordinates
    """

    flame_height = flame_height_calc(Q_dot, burner_width)
    virtual_source = burner_loc + np.array((0,0,(1/3.)*flame_height))
    return virtual_source

def normal_vec_from_integer(value):
    """
    Determines the normal vector from the FDS convention. 
    i.e. -3 is the negative z-direction, so would return (0,0,-1)
    2 is the positive y-direction so would return (0,1,0)

    inputs
    ------
    value: int
        The integer to convert to a normal vector

    returns
    -------
    normal_vec: array_like
        The normal vector indicated by the integer

    """

    normal_vec = np.zeros(3)
    sign = np.sign(value)
    normal_vec[int(abs(value))-1] = sign
    return normal_vec

def point_source_slope(virtual_origin, target_loc, target_orient):
    """
    Computes the beta such that 
    \dot{Q}_R = \beta*q" from the point source method

    inputs
    ------
    virtual_origin: array_like
        The (x,y,z) coordinates of the virtual origin
    target_loc: array_like
        The (x,y,z) coordinates of the target
    target_orient: array_like
        The normal vector of the target 

    returns
    -------
    beta: float
        The point source slope

    >>> round(point_source_slope(np.array([0,0,0]), np.array([1/2**.5, 1/2**.5, 0]), np.array([0,-1,0])), 4)
    17.7715
    """
    distance = distance_calc(virtual_origin, target_loc)
    line_of_sight = virtual_origin - target_loc
    sight_direction = normalize_vector(line_of_sight)
    cos_theta = sight_direction@target_orient
    beta = 4*np.pi*distance**2/cos_theta
    return beta

def point_source_calc(burner_loc, target_loc, target_orient, burner_width=0.32, Q_dot=100):
    """
    Runs all the methods to compute beta from a heat release rate

    inputs
    ------
    burner_loc: array_like
        The (x,y,z) coordinates describing the location of the burner
    target_loc: array_like
        The (x,y,z) coordinates of the target
    target_orient: array_like
        The normal vector of the target 
    burner_width: float
        The width (in m) of the burner
    Q_dot: float
        The heat release rate used to determine the virtual source (flame height correlation)

    returns
    -------
    beta: float
        The point source slope
    """

    virtual_origin = virtual_origin_calc(Q_dot, burner_loc, burner_width)
    beta = point_source_slope(virtual_origin, target_loc, target_orient)
    return beta

def read_yaml(path):
    """
    Convenience method from reading yaml files
    """
    with open(path) as infile:
        yaml_file = yaml.load(infile, Loader=yaml.FullLoader)
    return yaml_file

def load_DFT_data(exp_name, data_loc='../jan_setup/'):
    """
    Loads and cleans data for a given experiment

    inputs
    ------
    exp_name: str
        The name of the experiment without any extensions
    data_loc: str
        The path of the folder containing the experiment

    returns
    -------
    df: pandas.DataFrame
        A dataframe with the experimental data

    """


    df = pd.read_csv(data_loc+exp_name+'.csv')
    df = rename_DFT_columns(df)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df

def extract_dfts(df, substring='DFTJ'):
    """
    Extracts only DFTs from a file with information on all sensors

    inputs
    ------
    df: pandas.DataFrame
        Dataframe with sensor information
    substring: str
        The string to identify the sensors of interest. Default is DFTJ, which extracts only Jan's DFTs.

    returns:
    -------
    result: pandas.DataFrame
        The dataframe with only information on sensors of interest (dfts)

    """

    result = df[df['id'].str.contains(substring)].reset_index(drop=True)
    return result

def load_ramp(ramp_name, data_loc='../ramps/'):
    """
    Loads a hrr ramp from the directory containing ramps

    inputs
    ------
    ramp_name: string
        The name of the ramp to load
    data_locs: string
        The path to the folder that contains the HRR ramps

    returns
    -------
    ramp: pandas.DataFrame
        The HRR ramp. One column is time; the other is the HRR
    """
    ramp = pd.read_csv(data_loc+ramp_name)
    return ramp

class sensor:
    def __init__(self, sensor_info):
        for key in sensor_info.keys():
            setattr(self, key, sensor_info[key])
        self.location = np.array([self.x, self.y, self.z])

class burner:
    def __init__(self, burner_info, chi_R_dict):
        for key in burner_info:
            setattr(self, key, burner_info[key])

        self.chi_R = chi_R_dict[self.fuel]

        #Loading the HRR ramp
        self.HRR_ramp = load_ramp(self.HRR_ramp)
        
        #Converting to radiative HRR
        self.HRR_ramp['rad_HRR'] = self.chi_R*self.HRR_ramp['HRR']
        
        #Shifting the ignition time
        self.HRR_ramp['Time'] = self.HRR_ramp['Time'] + self.ignition_time
        
        #Making the location a numpy array
        self.extract_location()
    
    def extract_location(self):
        self.location = np.array([self.location['x'], 
                                  self.location['y'], 
                                  self.location['z']])
        
        
    def ramp_interp(self, time_vec, rad_hrr=False):
        if rad_hrr:
            interp_ramp = np.interp(time_vec, self.HRR_ramp['Time'], self.HRR_ramp['rad_HRR'])
        else:
            interp_ramp = np.interp(time_vec, self.HRR_ramp['Time'], self.HRR_ramp['HRR'])
        return interp_ramp
        
class experiment:
    def __init__(self, name, exp_info, sensors, chi_R):
        self.name = name

        self.dft_df = load_DFT_data(self.name)
        self.time_vec = np.array(self.dft_df['Time'])
        self.sensors = sensors
        self.chi_R_dict = chi_R

        self.make_burners(exp_info)
        self.compute_point_source_slopes()
        self.hrr_calc()
        self.compute_distance_matrix()
        self.predict_fluxes()
        self.predict_rad_hrr()

    def make_burners(self, exp_info):
        self.burners = {}
        for key in exp_info:
            self.burners[key] = burner(exp_info[key], self.chi_R_dict)


    def compute_point_source_slopes(self):
        for i in self.burners.keys():
            self.burners[i].point_source_coeff = {}
            for j in self.sensors.keys():
                burner_loc = self.burners[i].location
                target_loc = self.sensors[j].location
                target_orient = normal_vec_from_integer(self.sensors[j].orientation)
                beta = point_source_calc(burner_loc, target_loc, target_orient)
                self.burners[i].point_source_coeff[j] = beta

    def hrr_calc(self):
        self.rad_hrr_ramps = pd.DataFrame()
        for key in self.burners.keys():
            self.rad_hrr_ramps[key] = self.burners[key].ramp_interp(self.time_vec, rad_hrr=True)

        burner_cols = self.rad_hrr_ramps.columns
        self.rad_hrr_ramps['total'] = np.sum(self.rad_hrr_ramps[burner_cols].values, axis=1)
        self.rad_hrr_ramps['num_burning'] = np.sum(self.rad_hrr_ramps[burner_cols].values > 0, axis=1)

    def compute_distance_matrix(self):
        burner_locs = [self.burners[i].location for i in self.burners.keys()]
        sensor_locs = [self.sensors[i].location for i in self.sensors.keys()]
        self.dist_mat = distance_matrix(burner_locs, sensor_locs)

    def predict_fluxes(self):
        #Saves the flux contribution from each burner to each burner object
        #Total precicted flux is the sum of all individual burner contributions. 
        self.predicted_fluxes = np.zeros([len(self.rad_hrr_ramps), len(list(self.sensors.keys()))])
        for key in self.burners.keys():
            #Getting the burner object
            b = self.burners[key]
            slopes = np.array(list(b.point_source_coeff.values()))
            hrr_vector = np.array(self.rad_hrr_ramps[key])
            result = hrr_vector[:, np.newaxis]/slopes[np.newaxis,:]
            self.predicted_fluxes = self.predicted_fluxes + result
            result_df = pd.DataFrame(result, columns=b.point_source_coeff.keys())
            b.predicted_fluxes = result_df
        self.predicted_fluxes = pd.DataFrame(self.predicted_fluxes, columns=list(self.sensors.keys()))
    def predict_rad_hrr(self):
        """
            Currently only works when there is one burner
        """
        slopes = np.array(list(self.burners['burner_1'].point_source_coeff.values()))
        measurements = self.dft_df[self.sensors.keys()]
        self.pred_rad_hrr = measurements*slopes[np.newaxis,:]

class analysis:
    def __init__(self, config_loc='configs.yaml', 
                 exp_info_loc='experiment_info.yaml',
                 sensor_info_loc='../sensor_locs.csv'):

        configs = read_yaml(config_loc)
        for key in configs.keys():
            setattr(self, key, configs[key])
        self.exp_info_loc = exp_info_loc
        self.sensor_info_loc = sensor_info_loc

        self.load_sensors()
        self.make_experiments()

    def load_sensors(self):
        sensor_locs = pd.read_csv(self.sensor_info_loc)
        sensor_locs = extract_dfts(sensor_locs)
        self.sensors= {}

        for i,row in sensor_locs.iterrows():
            self.sensors[row['id']] = sensor(row)

    def make_experiments(self):
        all_exp_info = read_yaml(self.exp_info_loc)
        self.experiments = {}
        for name in self.exp_names:
            exp_info = all_exp_info[name]
            self.experiments[name] = experiment(name, exp_info, self.sensors, self.chi_R)

    def aggregate_dft_data(self, test_names=None):
        if test_names is None:
            test_names = list(a.experiments.keys())
        df_list = [self.experiments[i].dft_df for i in test_names]
        df = pd.concat(df_list).reset_index(drop=True)
        df = df.drop('Time', axis=1)
        return df

    def aggregate_hrr_data(self, test_names=None):
        if test_names is None:
            test_names = list(a.experiments.keys())
        df_list = [self.experiments[i].rad_hrr_ramps for i in test_names]
        df = pd.concat(df_list, sort=True).reset_index(drop=True)
        return df

    def aggregate_hrr_predictions(self, test_names=None):
        if test_names is None:
            test_names = list(a.experiments.keys())
        df_list = [self.experiments[i].pred_rad_hrr for i in test_names]
        df = pd.concat(df_list, sort=True).reset_index(drop=True)
        return df
