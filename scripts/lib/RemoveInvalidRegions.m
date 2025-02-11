function trialData = RemoveInvalidRegions(trialData)
% for any parts of the trial where a foot is on the ground but not on a
% force plate, replace that side's ID data with nan
    %% Parameters
    fpThresh = 200;
    filterFreq = 6;
    shortGapThresh = 25;
    
    simulatedfsr=simulatedFSRFromTRC(trialData.markers);

    %% Determine swing phase 
    markers = trialData.markers;
    RinAir = ~simulatedfsr.Right;
    LinAir = ~simulatedfsr.Left;
    %% Determine foot on force plate
    id = trialData.id;
    fp = trialData.fp;
        
    % estimate when a force plate is pressed and which feet are on which
    % force plates
    pressed = Vicon.Filter(fp{:, 3:9:end}, mean(diff(fp.Header)) * 2 * filterFreq) > fpThresh;
    pressed = pressed(1:5:end, :);
    sides = Osim.correlateForcePlates(markers, fp);
    
    % determine which moments apply to which sides
    rightSideMoments = contains(id.Properties.VariableNames, '_r_');
    leftSideMoments = contains(id.Properties.VariableNames, '_l_');
    bothSideMoments = ~(rightSideMoments | leftSideMoments);
    bothSideMoments(1) = false;
    
    if (size(pressed,1)-size(RinAir,1))~=0
        pressed=[pressed; repmat(pressed(end,:),size(RinAir,1)-size(pressed,1),1)];
    end
    % determine where right foot ID data is invalid
    rightNanMask = ~any(RinAir | pressed(:, sides == 'r'), 2);
    nans = splitLogical(rightNanMask); % in sf_pre/lib
    % short gaps are probably in between swing phase estimation and force
    % plate press, so do not invalidate the data there
    shortGapIdxs = nans(cellfun('length', nans) < shortGapThresh);
    rightNanMask([shortGapIdxs{:}]) = false;
    id{rightNanMask, rightSideMoments} = nan;

    leftNanMask = ~any(LinAir | pressed(:, sides == 'l'), 2);
    nans = splitLogical(leftNanMask);
    shortGapIdxs = nans(cellfun('length', nans) < shortGapThresh);
    leftNanMask([shortGapIdxs{:}]) = false;
    id{leftNanMask, leftSideMoments} = nan;

    bothNanMask = leftNanMask | rightNanMask;
    id{bothNanMask, bothSideMoments} = nan;
    trialData.id = id;
end