init();def=defaults;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Select an ambulation mode and a sensor/channel of interest
AMBULATION='treadmill';
%sensor='ik'; channel='hip_flexion_r';
%sensor='emg'; channel='gastrocmed';
%sensor='imu'; channel='thigh_Gyro_Z'; 
sensor='id'; channel='ankle_angle_r_moment';

allfiles=fstrides.fileList('File',[AMBULATION,'.mat']);
subjects=fstrides.getFields(allfiles,'Subject');
data=multiload(allfiles);
data=cellfun(@(x)(x.strides),data,'Uni',0);
% Upon loading the data is in the form {{subject1strides},{subject2strides},...}}

if(checkFields(data, sensor, channel)) %check to make sure the sensor/channels exist
    x=cell(size(data)); y=cell(size(data));
    for subjectIdx=1:numel(subjects)
        y{subjectIdx}=GetChannelFromStrides(data{subjectIdx},sensor,channel);
        % First dimension is the header, second dimension is the stride                 
        condition=GetConditionFromStrides(data{subjectIdx});
        x{subjectIdx}=condition(~isnan(condition));
        y{subjectIdx}=y{subjectIdx}(:,~isnan(condition));
    end
end

%% Plot every stride showing each subject in a subplot
h=figure(1); h.Name='StridesBySubject'; clf;
M=floor(sqrt(numel(subjects))); N=ceil(numel(subjects)/M);
unique_x=unique(x{1});
cmap=parula(numel(unique_x));
for subjectIdx=1:numel(subjects)
    subplot(M,N,subjectIdx);
    yy=y{subjectIdx};
    xx=x{subjectIdx};    
    for i=1:numel(unique_x)
        plot(yy(:,xx==unique_x(i)),'Color',cmap(i,:)); hold on;
    end
    title(sprintf('%s',subjects{subjectIdx}));
    xlabel('Gait cycle (%)');
    ylabel(yLabelSelector(sensor, channel));
end

%% Average per condition and show every subject in a subplot
h=figure(2); h.Name='MeanBySubject'; clf;
M=floor(sqrt(numel(subjects))); N=ceil(numel(subjects)/M);
unique_x=unique(x{1});
cmap=parula(numel(unique_x));
for subjectIdx=1:numel(subjects)
    subplot(M,N,subjectIdx);
    yy=y{subjectIdx};
    xx=x{subjectIdx};    
    for i=1:numel(unique_x)
        plot(nanmean(yy(:,xx==unique_x(i)),2),'Color',cmap(i,:)); hold on;
    end
    title(sprintf('%s',subjects{subjectIdx}));
    xlabel('Gait cycle (%)');
    ylabel(yLabelSelector(sensor, channel));
end

%% Average cross subjects
h=figure(3); h.Name='MeanCrossSubjects'; clf;
unique_x=unique(x{1});
cmap=parula(numel(unique_x));
ymean=cell(size(y)); xmean=cell(size(y));
for subjectIdx=1:numel(subjects)    
    yy=y{subjectIdx};
    xx=x{subjectIdx};
    yymean=zeros(size(yy,1),numel(unique_x));
    for i=1:numel(unique_x)
       yymean(:,i)=nanmean(yy(:,xx==unique_x(i)),2);
    end
    ymean{subjectIdx}=yymean;
    xmean{subjectIdx}=unique_x;   
end
yavg=nanmean(cat(3,ymean{:}),3);
xavg=round(nanmean(cat(3,xmean{:}),3),4); % Not real need for this as all xmean terms should be the same
for i=1:numel(unique_x)
    plot(nanmean(yavg(:,xavg==unique_x(i)),2),'Color',cmap(i,:)); hold on;
end
title('Across Subjects');
xlabel('Gait cycle (%)');
ylabel(yLabelSelector(sensor, channel));


function out = checkFields(data, sensor, channel)
%% Make sure the data contains the selected sensor and channel
    out = false;
    
    trial = data{1}{1};
    if(isfield(trial, sensor))
        if(any(strcmp(trial.(sensor).Properties.VariableNames, channel)))
            out = true;
            return;
        else
            warning('Available Channels:');
            trial.(sensor).Properties.VariableNames'
            error('Channel not found... please make sure the channel name is correct');    
        end
    else
        warning('Available Sensors:');
        fieldnames(trial)
        error('Sensor not found... please make sure the sensor name is correct');
    end
end