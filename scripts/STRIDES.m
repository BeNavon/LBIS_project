init();def=defaults;

%% For each ambulation mode split every stride with the right leg heel strike.
%% Define 'SUBJECT' prior to running this script, or let it use the default subject

%ambulationModes={'levelground','ramp','stair','treadmill'};
ambulationModes={'treadmill'};

sensors={'markers','gcLeft','gcRight','conditions','ik','id','emg','imu','gon','jp'};

%% Load subject information to use when normalizing data by weight/mass
subjectInfo=load(def.SUBJECT_INFO_FILE); subjectInfo=subjectInfo.data;
subjectMass=subjectInfo.Weight(strcmp(subjectInfo.Subject,SUBJECT));

%% Load this subjects Treadmill data and compute the normalization ranges for
% EMG based on a speed of 1.35m/s.
fprintf('Processing strides for subject %s\n',SUBJECT);
treadmillFiles=f.fileList('Subject',SUBJECT,'Mode','treadmill','Sensor',{'emg','gcRight','conditions'},'Trial','*.mat');
emgTreadmillTrials=f.EpicToolbox(treadmillFiles);
emgTreadmillTrials=Topics.processTopics(@rectify,emgTreadmillTrials,{'emg'});
[minvalues,maxvalues]=getEMGNormalization(emgTreadmillTrials,1.35);

%% For each ambulation mode retrieve the trials and segment the data based on
% right foot strides. Saves the individual strides information on a cell
% array per subject/mode.


for i=1:numel(ambulationModes)    
    ambulationMode=ambulationModes{i};
    fprintf('\t %s strides\n',ambulationMode);
    allfiles=f.fileList('Subject',SUBJECT,'Mode',ambulationMode,'Sensor',sensors);
    trials=f.EpicToolbox(allfiles);
    trials=Topics.processTopics(@rectify,trials,{'emg'});
    trials=Topics.transform(@(x)((x-minvalues)./(maxvalues-minvalues)),trials,{'emg'});

    allstrides=[];   
    % Get every stride
    for j=1:numel(trials)
        trial=trials{j};                     
        % plotTrial(trial);        
        %% Segment data by gait cycle of the right leg        
        strides=segment_gc(trial,'GCtopic','gcRight','GCchannel','HeelStrike');        
        allstrides=[allstrides;strides];                
    end
    
	% Example of how to compute temporal-spatial metrics of each stride     
    ENABLE_COMPUTE_STRIDE_METRICS=false;
    if ENABLE_COMPUTE_STRIDE_METRICS
        allmetrics=[];
        for j=1:numel(allstrides)
            stride=allstrides{j};
            metrics=ComputeStrideMetrics(stride);
            allmetrics=[allmetrics; metrics];        
        end
    end
        
    %% Normalize strides by gait cycle (0-100%) and assign to the list of profiles for each
    % sensor by label.
    allstrides=Topics.normalize(allstrides,Topics.topics(allstrides),'Header');
    allstrides=Topics.interpolate(allstrides,0:0.01:1);
    
    %% Normalize ID, JP by subject mass  
    % allstrides=Topics.processTopics(@(x)FPNormalizeByWeight(x,subjectMass*9.81),allstrides,{'FPLeft','FPRight'});    % forceplate data by subject's weight  
    allstrides=Topics.transform(@(x)(x/subjectMass),allstrides,{'id','jp'});        
    
    %% Classify stride based on the label    
    alllabels=cell(size(allstrides));
    for j=1:numel(allstrides)
        stride=allstrides{j};
        c=stride.conditions;
        
        if strcmp(ambulationMode,'treadmill')           
            % Only select steady speed walking and discard strides
            % belonging to acceleration periods.
            if numel(unique(c.speed.Speed))>2
                alllabels{j}='discard';
            else
                alllabels{j}='treadmill';
            end
            continue;
        end
        
        labels=c.labels;
        alllabels{j}='discard';
        if strcmp(ambulationMode,'levelground')
            % Discard idle "steps"
            if all(strcmp(labels.Label,'idle'))
                continue;
            elseif strcmp(labels.Label{1},'idle')
                continue;
            elseif strcmp(labels.Label{1},'stand-walk') && c.leadingLegStart{1}=='l'
                alllabels{j}='stand-walk';
            elseif strcmp(labels.Label{1},'stand') && strcmp(labels.Label{end},'stand-walk') && c.leadingLegStart{1}=='r'
                continue;
            elseif strcmp(labels.Label{1},'stand') && strcmp(labels.Label{end},'stand-walk') && c.leadingLegStart{1}=='l'
                alllabels{j}='stand-walk';
            elseif any(contains(labels.Label,'turn')) && strcmp(c.turn,'ccw')
                alllabels{j}='turnccw';
            elseif any(contains(labels.Label,'turn')) && strcmp(c.turn,'cw')
                alllabels{j}='turncw';
            elseif all(strcmp(labels.Label,'walk'))
                alllabels{j}='walk';
            elseif strcmp(labels.Label{end},'walk-stand') && c.leadingLegStop{1}=='r'
                alllabels{j}='walk-stand';
            elseif strcmp(labels.Label{1},'walk-stand') && strcmp(labels.Label{end},'stand') && c.leadingLegStop{1}=='r'
                alllabels{j}='stand';
            elseif strcmp(labels.Label{1},'stand-walk') && strcmp(labels.Label{end},'walk') && c.leadingLegStart{1}=='r'
                alllabels{j}='walk';
            elseif strcmp(labels.Label{1},'walk') && strcmp(labels.Label{end},'walk-stand') && c.leadingLegStop{1}=='l'
                alllabels{j}='walk-stand';
            elseif strcmp(labels.Label{1},'walk-stand') &&  c.leadingLegStop{1}=='l'
                continue;
            elseif strcmp(labels.Label{1},'walk-stand') && strcmp(labels.Label{end},'idle') && c.leadingLegStop{1}=='r' ...
                    && any(contains(labels.Label,'stand'))
                continue;            
            elseif strcmp(labels.Label{1},'walk-stand') && strcmp(labels.Label{end},'idle') && c.leadingLegStop{1}=='l' ...
                    && any(contains(labels.Label,'stand'))
                continue;
            elseif strcmp(labels.Label{1},'walk-stand') && strcmp(labels.Label{end},'stand') && c.leadingLegStop{1}=='l'
                alllabels{j}='stand';  
             elseif strcmp(labels.Label{1},'walk') && strcmp(labels.Label{end},'idle') && c.leadingLegStop{1}=='r' ...
                && any(contains(labels.Label,'walk-stand')) && any(contains(labels.Label,'stand'))
                continue;
            elseif strcmp(labels.Label{1},'stand') && c.leadingLegStart{1}=='l'
                alllabels{j}='stand-walk';
            else
               j
                warning('Case not found');
                vis(stride);
                figure(1);
                plotTrial(stride);
                fprintf('Problem');
                error
            end      
        elseif strcmp(ambulationMode,'ramp')
            % Discard idle "steps"
            if all(strcmp(labels.Label,'idle'))
                continue;
            elseif strcmp(labels.Label{1},'idle')
                continue;
            elseif (strcmp(labels.Label{1},'rampascent-walk') ...
                    && strcmp(labels.Label{end},'walk-rampdescent') ...
                    && any(strcmp(labels.Label,'idle')))
                continue;
            elseif strcmp(labels.Label{1},'walk-rampascent') && c.transLegAscent(1)=='l'
                alllabels{j}='rampascent';
            elseif strcmp(labels.Label{1},'walk-rampascent') && c.transLegAscent(1)=='r'
                alllabels{j}='walk-rampascent';
            elseif all(strcmp(labels.Label,'rampascent'))
                alllabels{j}='rampascent';                
            elseif strcmp(labels.Label{end},'rampascent-walk')
                alllabels{j}='rampascent-walk';                
            elseif strcmp(labels.Label{end},'idle') && strcmp(labels.Label{1},'rampascent-walk')
                continue;
            elseif strcmp(labels.Label{1},'walk-rampdescent') && c.transLegDescent(1)=='l'
                alllabels{j}='rampdescent';
            elseif strcmp(labels.Label{1},'walk-rampdescent') && c.transLegDescent(1)=='r'
                alllabels{j}='walk-rampdescent';                
            elseif all(strcmp(labels.Label,'rampdescent'))
                alllabels{j}='rampdescent';      
            elseif strcmp(labels.Label{end},'rampdescent-walk')
                alllabels{j}='rampdescent-walk';               
            elseif strcmp(labels.Label{end},'idle') && strcmp(labels.Label{1},'rampdescent-walk')
                continue;
            elseif strcmp(labels.Label{1},'rampdescent-walk') && strcmp(labels.Label{end},'walk-rampascent') &&  c.transLegAscent(1)=='r'
                continue;
                warning('bad labeling?');
            elseif strcmp(labels.Label{1},'rampascent-walk') && strcmp(labels.Label{end},'rampdescent') && ...
                    any(contains(labels.Label,'walk-rampdescent'))
                continue;
           
            else
                j
                warning('Case not found');
                vis(stride);
                figure(1);
                plotTrial(stride);
                fprintf('Problem');
                error
            end                       
        elseif strcmp(ambulationMode,'stair')            
            if all(strcmp(labels.Label,'idle'))
                continue;
            elseif strcmp(labels.Label{1},'idle') && strcmp(labels.Label{end},'walk-stairascent') ...
                && (c.transLegAscent(1)=='l')
                alllabels{j}='walk-stairascent';  
            elseif strcmp(labels.Label{1},'walk-stairascent') && strcmp(labels.Label{end},'stairascent') ...
                && (c.transLegAscent(1)=='l')
                alllabels{j}='stairascent';  
            elseif strcmp(labels.Label{1},'stairascent') && strcmp(labels.Label{end},'stairascent-walk') ...
                && (c.transLegAscent(2)=='r')
                alllabels{j}='stairascent-walk';  
            elseif strcmp(labels.Label{1},'stairascent-walk') && strcmp(labels.Label{end},'idle') ...
                && (c.transLegAscent(2)=='r')
                continue; 
            elseif strcmp(labels.Label{1},'idle') && strcmp(labels.Label{end},'walk-stairdescent') ...
                && (c.transLegDescent(1)=='l')
                alllabels{j}='stairdescent-walk';  
            elseif strcmp(labels.Label{1},'walk-stairdescent') && strcmp(labels.Label{end},'stairdescent') ...
                && (c.transLegDescent(1)=='l')
                alllabels{j}='stairdescent';      
            elseif strcmp(labels.Label{1},'stairdescent') && strcmp(labels.Label{end},'stairdescent-walk') ...
                && (c.transLegDescent(2)=='r')
                alllabels{j}='stairdescent-walk';      
            elseif strcmp(labels.Label{1},'stairdescent-walk') && strcmp(labels.Label{end},'idle') ...
                && (c.transLegDescent(2)=='r')
                continue;        
            elseif all(strcmp(labels.Label{1},'stairascent-walk')) ...
                && (c.transLegDescent(2)=='l')
                continue;        
            elseif strcmp(labels.Label{1},'walk-stairascent') && strcmp(labels.Label{end},'stairascent') ...
                && (c.transLegAscent(1)=='r')
                alllabels{j}='stairascent';                  
            elseif all(strcmp(labels.Label,'stairascent'))
                alllabels{j}='stairascent';   
            elseif strcmp(labels.Label{1},'stairascent') && strcmp(labels.Label{end},'stairascent-walk') ...
                && (c.transLegAscent(1)=='r')
                alllabels{j}='stairascent-walk'; 
            elseif strcmp(labels.Label{1},'stairascent-walk') && strcmp(labels.Label{end},'idle') ...
                && (c.transLegAscent(2)=='l')
                continue; 
            elseif strcmp(labels.Label{1},'idle') && strcmp(labels.Label{end},'walk-stairdescent') ...
                && (c.transLegDescent(1)=='r')
                continue; 
            elseif strcmp(labels.Label{1},'walk-stairdescent') && strcmp(labels.Label{end},'stairdescent') ...
                && (c.transLegDescent(1)=='r')
                alllabels{j}='stairdescent'; 
             elseif all(strcmp(labels.Label,'stairdescent'))
                alllabels{j}='stairdescent'; 
             elseif strcmp(labels.Label{1},'stairdescent') && strcmp(labels.Label{end},'stairdescent-walk') ...
                && (c.transLegDescent(2)=='l')
                alllabels{j}='stairdescent-walk';
             elseif strcmp(labels.Label{1},'stairdescent-walk') && strcmp(labels.Label{end},'idle') ...
                && (c.transLegDescent(2)=='l')
                continue;
             elseif strcmp(labels.Label{1},'idle') && strcmp(labels.Label{end},'walk-stairascent') ...
                && (c.transLegAscent(1)=='r')
                continue;
             elseif strcmp(labels.Label{1},'stairascent-walk') && strcmp(labels.Label{end},'walk-stairdescent') ...
                && any(contains(labels.Label,'idle')) && (c.transLegAscent(2)=='l')
                continue;
             elseif strcmp(labels.Label{1},'stairascent-walk') && strcmp(labels.Label{end},'walk-stairdescent') ...
                && any(contains(labels.Label,'idle')) && (c.transLegAscent(2)=='l')
                continue;
            elseif strcmp(labels.Label{1},'stairascent-walk') && strcmp(labels.Label{end},'walk-stairdescent') ...
                && (c.transLegAscent(2)=='r')
                warning('badlabel');
                continue;
      
           
            else
                j
                warning('Case not found');
                vis(stride);
                figure(1);
                plotTrial(stride);
                fprintf('Problem');
                error
            end
            
        
        end
        %plotTrial(stride);
    end
    
    [u_labels,~,u_labels_idx]=unique(alllabels);
    for j=1:numel(u_labels)
        label=u_labels{j};
        if strcmp(label,'discard')
            continue;
        end       
        strides=allstrides(u_labels_idx==j);            
        outfile=fstrides.genList('Subject',SUBJECT,'File',[label '.mat']); outfile=outfile{1};
        mkdirfile(outfile);
        save(outfile,'strides');
    end
        
    
    %%        
    
end