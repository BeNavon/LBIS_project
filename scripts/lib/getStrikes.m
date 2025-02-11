function [strikes, offs] = getStrikes(markerDataArr, varargin)
%% Generate Strike and Off indices
% Generalized to any marker!!
%
% [strikes, offs] = getHeelStrikes(dataArr)
%
% Inputs:
%   dataArr = array of data for one marker [x(:), y(:), z(:)]
% Outputs:
%   strikes = vector strike indices
%   offs = vector of off indices
%
% strike is defined as first time after a swing phase where the total
%   velocity of the marker is below a threshhold
% off is defined as last time before swing phase where the total velocity is
%   above a threshold

    p = inputParser;
    p.KeepUnmatched = true;
    Names = {'v_thresh_stance', 'v_thresh_swing'};
    defaults = {0.1, 0.1};
    for i = 1:length(Names)
        addOptional(p,Names{i},defaults{i});
    end
    parse(p,varargin{:});
    v_thresh_stance = p.Results.v_thresh_stance; %threshhold for "zero" velocity (% of max velocity)
    v_thresh_swing = p.Results.v_thresh_swing; %threshold for swing phase (% of max velocity)
    
    v_x = gradient(smoothdata(markerDataArr(:,1), 'movmedian', 50));
    v_y = gradient(smoothdata(markerDataArr(:,2), 'movmedian', 50));
    v_z = gradient(smoothdata(markerDataArr(:,3), 'movmedian', 50));
    
    v_tot = sqrt(v_x.^2 + v_y.^2 + v_z.^2);
    v_tot = smoothdata(v_tot, 'movmedian', 100);
    v_tot = v_tot/max(abs(v_tot)); %normalize
    
    minpeakdistance=min(25,numel(v_tot)-2);
    [midPk, midSwings] = findpeaks(v_tot, 'MinPeakProminence', v_thresh_swing, 'MinPeakDistance',minpeakdistance);
    [stancePk, stance] = findpeaks(-1*v_tot, 'MinPeakProminence', v_thresh_stance);
    stancePk = -1*stancePk;
    
    %only take first heel strike per swing and last heel off previous to swing
    strikes = [];
    offs = [];
    for swingInd = 1:length(midSwings) + 1
        getOff = true;
        getStrike = true;
        if(swingInd == 1)
            beg = 1; %get first off
            getStrike = false;
        else
            beg = midSwings(swingInd - 1);
        end
        if(swingInd == length(midSwings) + 1)
            fin = length(v_tot); %get last strike
            getOff = false;
        else
            fin = midSwings(swingInd);
        end
        inds = beg:fin;
        
        v_temp = v_tot(beg:fin);
        if(getOff)
            high = midPk(swingInd);
            temp = inds(v_temp < 0.1*(high - min(v_temp)) + min(v_temp));
            offs = [offs, temp(end)];
        end
        if(getStrike) 
           high = midPk(swingInd - 1);
           temp = inds(v_temp < 0.1*(high - min(v_temp)) + min(v_temp));
           strikes = [strikes, temp(1)];
        end
    end
    
    %figure
    %plot(v_tot)
    %hold on
    %line = zeros(size(v_tot));
    %line(strikes) = 1;
    %plot(line);
    %line = zeros(size(v_tot));
    %line(offs) = 1;
    %plot(line);
end