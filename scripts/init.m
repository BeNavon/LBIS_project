% Check for prerequisites/toolboxes
if ~exist('FileManager','class')
	error(['Please make sure you install EpicToolbox to add the required dependencies to your path.\n'...
	'Run the install.m script from the EpicToolbox directory '])	
end
if isempty(which('Osim.IK'))
	warning('MoCapTools not detected. This is only needed if you want to use RUN_OSIM.m to regenerate id and ik');
	warning('Run the install.m script from the MoCapTools directory');
end

f=FileManager('..','PathStructure',{'Subject','Date','Mode','Sensor','Trial'});
fstrides=FileManager('STRIDES','PathStructure',{'Subject','File'});
fosim=FileManager('..','PathStructure',{'Subject','Sensor','File'});

addpath('lib');

%% Defaults
defaults=struct();
defaults.OVERWRITE=true;
defaults.SUBJECT='AB09';
defaults.TRIAL='*';
defaults.RAWMODES={'levelground','ramp','stair','treadmill'};
defaults.IMUTOPICS= {'foot.Accel', 'foot.Gyro',...
        'shank.Accel', 'shank.Gyro',...
        'thigh.Accel', 'thigh.Gyro',...
        'trunk.Accel', 'trunk.Gyro'};      
defaults.SUBJECT_INFO_FILE='../SubjectInfo.mat';
    
if ~exist('SUBJECT','var')
    SUBJECT=defaults.SUBJECT;
end
if ~exist('RAWMODES','var')
    RAWMODES=defaults.RAWMODES;
end
if ~exist('TRIAL','var')
    TRIAL=defaults.TRIAL;
end


