function JointTorquesTable = RampID(trialData, IDTemplateFile)
% Calculate inverse dynamics for ramp trials, taking into account
% different feet going up vs. down. 
% 
    assert(all(isfield(trialData, {'conditions', 'fp', 'markers', 'ik'})), 'Trial data must have conditions, fp, trc, and ik');
    %% Determine where to split trials
    labels = trialData.conditions.labels;
    idleRegions = splitLogical(strcmp(labels.Label, 'idle'));
    mids = cellfun(@(vec) vec(round(end/2)), idleRegions); % split in the middle of every idle region
    splitTimes = labels.Header([1; mids; end]);
    %% Create setup files
    allTopics = {'fp', 'markers', 'ik', 'conditions.labels'};
    devices = strrep(trialData.fp.Properties.VariableNames(2:9:end), '_vx', '');
    sides = Osim.correlateForcePlates(trialData.markers, trialData.fp);
    % create one extLoads xml assuming that right foot is on the center
    % force plate for the whole trial, then create another one for left
    sides(3) = 'r';
    extR = Osim.createExternalLoads(devices, sides);
    sides(3) = 'l';
    extL = Osim.createExternalLoads(devices, sides);
    fpMot = Osim.writeMOT(trialData.fp);
    ikMot = Osim.writeMOT(trialData.ik);
    idXml = Osim.editSetupXML(IDTemplateFile, 'external_loads_file', extR);
    % run ID twice, once for left foot, once for right foot
    jointTorquesR = Osim.ID(fpMot, idXml, ikMot);
    JointTorquesTable = jointTorquesR; % initialize output with right foot 
    idXml = Osim.editSetupXML(idXml, 'external_loads_file', extL);
    jointTorquesL = Osim.ID(fpMot, idXml, ikMot);
    %% Merge two tables
    for idx = 1:length(splitTimes)-1
        % for each segment, determine which foot is on the center force
        % plate, and copy that data to the output table
        cutTrialData = Topics.cut(trialData, splitTimes(idx), splitTimes(idx+1), allTopics);
        sides = Osim.correlateForcePlates(cutTrialData.trc, cutTrialData.fp);
        if sides(3) == 'l'
            thisSegMask = JointTorquesTable.Header >= splitTimes(idx) & JointTorquesTable.Header <= splitTimes(idx+1);
            JointTorquesTable(thisSegMask, :) = jointTorquesL(thisSegMask, :);
        end
    end
end
