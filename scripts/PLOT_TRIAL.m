%% PLOT_TRIAL: Demonstrates accessing to particular information within the
% dataset by selecting certain trial files and plotting the data.
%
% Written by: Jonathan Camargo
% Checked by: Will Flanagan and Aditya Ramanathan
% ========================================================================
% REQUIREMENTS
% The demonstration scripts rely on two Matlab toolboxes. Copies of the 
% toolboxes are provided in the folders.
%
% EpicToolbox 
% EpicToolbox is a Matlab toolbox for data processing of time series data. 
% Find the original repository at https://github.com/JonathanCamargo/EpicToolbox.
% 
% MoCapTools 
% MocapTools is a Matlab toolbox for motion capture analysis, including 
% programmatically running OpenSim, and automatic gap-filling of data. 
% Find the original repository at https://github.com/JonathanCamargo/MocapTools.
%
% ========================================================================
%
%% Initialize some paths and variables
% The init function creates a FileMananager object (defined in EpicToolbox)
% to facilitate access to the nested folder structure of the dataset. Once
% you download the dataset and move it to its final location, make sure that
% datasetpath is updated in the init.m script.
init(); def=defaults;

% f is a FileManager object. It simplifies the accessing to data within 
% nested folders and improves the readability of the code. 

% For example: use f to list all the mat files for an ambulation mode:
% allfiles=f.fileList('Subject','AB06','Mode','treadmill','Trial','*');


%% Retrieve the data from a particular trial and plot the signals
%% Select a subject/ambulation/trial and load data to matlab
SUBJECT='AB09';
AMBULATION='treadmill';
TRIAL='treadmill_02_01.mat';

% Get a list of all the files that belong to a trial (e.g. treadmill_01_01.mat')
allfiles=f.fileList('Subject',SUBJECT,'Mode',AMBULATION,'Trial',TRIAL);
% Load all the data as a struct where each field is a sensor name. 
% Instead of manually loading each file, use f.EpicToolbox to load and return
% a cell array with all the trials that were loaded (you can use it to load
% data from multiple trials at the same time.
trials=f.EpicToolbox(allfiles); % numel(trials)==1 Since allfiles belong from only one trial

% % You can use the same method to load multiple trials. For example, this will load
% % two trials:
% TRIAL={'treadmill_01_01.mat','treadmill_02_01.mat'};
% allfiles=f.fileList('Subject',SUBJECT,'Mode',AMBULATION,'Trial',TRIAL);
% trials=f.EpicToolbox(allfiles); % numel(trials)==2 
% % Or TRIAL='treadmill*.mat'; will load all treadmill trials.
% % You can also load partial data by using the parameter 'Sensor' to only
% % load the information you need, instead of all the data from a trial. For
% % example, only load emg and conditions:
% SENSORS={'emg','conditions'};
% allfiles=f.fileList('Subject',SUBJECT,'Mode',AMBULATION,'Sensor',SENSORS,'Trial',TRIAL);
% trials=f.EpicToolbox(allfiles);
 

%% Use the different data from trials to plot the time series of raw data
trial=trials{1};

% Plot Walking speed from the trial conditions
subplot(4,1,1);
plot(trial.conditions.speed.Header,trial.conditions.speed.Speed);
xlabel('Time(s)'); ylabel('Speed (m/s)');
% Plot Gastrocnemius medialis from the EMG data
subplot(4,1,2);
plot(trial.emg.Header,trial.emg.gastrocmed);
xlabel('Time(s)'); ylabel('EMG');
% Plot the hip flexion from the inverse kinematics
subplot(4,1,3);
plot(trial.ik.Header,trial.ik.hip_flexion_r);
xlabel('Time(s)'); ylabel('Hip angle (deg)');
% Plot the hip flexion moment from the inverse dynamics
subplot(4,1,4);
plot(trial.id.Header,trial.id.hip_flexion_r_moment);
xlabel('Time(s)'); ylabel('Hip moment (N/m)');

%% Plot anything else for this trial
sensor = 'fp';
channel = 'Treadmill_R_vy';

if(isfield(trial, sensor))
    if(any(strcmp(trial.(sensor).Properties.VariableNames, channel)))
        figure();
        plot(trial.(sensor).Header,trial.(sensor).(channel));
        xlabel('Time(s)');
        title(['Sensor: ', sensor, '; Channel: ', channel], 'Interpreter', 'none');
    else
        warning('Available Channels:');
        trial.(sensor).Properties.VariableNames'
        error('Channel not found... please make sure the channel name is correct');    
    end
else
    warning('Available Sensors:');
    fieldnames(trial)'
    error('Sensor not found... please make sure the sensor name is correct');
end







