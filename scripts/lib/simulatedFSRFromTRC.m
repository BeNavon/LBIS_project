function simulatedFSR = simulatedFSRFromTRC(trc,varargin)
% simulatedFSR = simulatedFSRFromTRC(trc, varargin)
% Returns a table of simulated FSR data with Header, Left, and Right
% columns given trc data. 
% Optional Inputs: 
% Marker: The two markers to be used to detect ground contact, as a cell
% array of character vectors. Default is {'Heel', 'Toe_Tip'}
% Threshold: The threshold velocity (as a proportion of the maximum) to
% determine whether the foot in in swing phase or not. Default is 0.1 

% might want to consider changing threshold to 0.05 but lots of further
% testing would be necessary

    data = Osim.interpret(trc, 'TRC');
    p = inputParser;   
    addParameter(p,'RMarkers',{'R_Heel','R_Toe_Tip'},@iscell);
    addParameter(p,'LMarkers',{'L_Heel','L_Toe_Tip'},@iscell);
    addOptional(p,'Threshold',0.1,@(x) (isscalar(x) && isnumeric(x)));
    p.parse(varargin{:});
    RmarkerName=p.Results.RMarkers;
    LmarkerName=p.Results.LMarkers;
    threshold = p.Results.Threshold;
    
    markerRight1 = RmarkerName{1};
    markerRight2 = RmarkerName{2};
    markerLeft1  = LmarkerName{1};
    markerLeft2  = LmarkerName{2};
    
    opts = {'v_thresh_stance', threshold, 'v_thresh_swing', threshold};
    
    % get heel and toe info so that we can determine more accurately when
    % the foot is in contact with the ground
    [heelStrikes_r, heelOffs_r] = getStrikes(data{:, compose([markerRight1 '_%c'], 'xyz')}, opts{:});
    [heelStrikes_l, heelOffs_l] = getStrikes(data{:, compose([markerLeft1 '_%c'],'xyz')}, opts{:});
    [toeStrikes_r, toeOffs_r] = getStrikes(data{:, compose([markerRight2 '_%c'], 'xyz')}, opts{:});
    [toeStrikes_l, toeOffs_l] = getStrikes(data{:, compose([markerLeft2 '_%c'], 'xyz')}, opts{:});
    
    % decide whether heel or toe should be used in each strike/off
    [strikes_r, offs_r] = decideHeelToe(heelStrikes_r, heelOffs_r, toeStrikes_r, toeOffs_r);
    [strikes_l, offs_l] = decideHeelToe(heelStrikes_l, heelOffs_l, toeStrikes_l, toeOffs_l);
    
    %create simulatedFSR
    simulatedFSR = table(data.Header, 'VariableNames', {'Header'});
    n = length(data.Header);
    simulatedFSR.Right = generateFSRvector(strikes_r, offs_r, n);
    simulatedFSR.Left = generateFSRvector(strikes_l, offs_l, n);
end

function [strikes, offs] = decideHeelToe(hs, ho, ts, to)
    %{
    |x|xxxxxx]x]      |x|xxxxxx]x]      |xxxxxx]x]      |x|xxxxxx]      |xxxxxx] 
    |x|xxxxxx]x]      |x|xxxxxx]x]      |xxxxxx]x]      |x|xxxxxx]      |xxxxxx] 
    |x|xxxxxx]x]      |x|xxxxxx]x]      |xxxxxx]x]      |x|xxxxxx]      |xxxxxx] 
    |x|xxxxxx]x]      |x|xxxxxx]x]      |xxxxxx]x]      |x|xxxxxx]      |xxxxxx] 
    strk     off      strk     off     strk    off     strk     off    strk   off
    
    x = duration that should be marked as ground contact
    | = strike (either toe strike or heel strike) 
    ] = off (either toe off or heel off)
    
    these are the different cases of what we might see in the data when we
    take into account heel and toe (instead of just one). 
    %}
    
    % We want the last of the offs before a strike to mark the end of
    % standing
    allStrikes = sort([hs, ts]);
    allOffs = sort([ho, to]);
    offs = [];
    for strikeIdx = 1:numel(allStrikes)
        strikeTime = allStrikes(strikeIdx); 
        prevOffs = allOffs(allOffs < strikeTime); % get the last off before each strike (there will be doubles)
        if ~isempty(prevOffs)
            offs = [offs, prevOffs(end)];
        end
    end
    offs = [offs, allOffs(end)]; % in case we end in swing, add the last off (might be a double)
    offs = sort(unique(offs)); % remove doubles
    
    % Next we want the first of the strikes after each off to mark the
    % beginning of standing
    strikes = [];
    for offIdx = 1:numel(offs)
        offTime = offs(offIdx);
        nextStrikes = allStrikes(allStrikes > offTime);
        if ~isempty(nextStrikes)
            strikes = [strikes, nextStrikes(1)];
        end
    end
    strikes = [allStrikes(1), strikes]; % in case we start in swing, add the first strike (might be a double)
    strikes = sort(unique(strikes)); % remove doubles
end

function fsr_temp = generateFSRvector(hs, ho, len)
    fsr_temp = false(len, 1);
    if(ho(1) < hs(1)) % starts standing
        fsr_temp(1:ho(1)) = true;
    end
    for i = 1:length(ho)
        prevHeelStrike = hs(hs < ho(i));
        if(~isempty(prevHeelStrike))
            fsr_temp(prevHeelStrike(end):ho(i)) = true;
        end
    end
    if (hs(end) > ho(end))
        fsr_temp(hs(end):end) = true;
    end
end