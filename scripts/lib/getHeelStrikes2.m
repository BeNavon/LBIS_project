function [strikes, offs] = getHeelStrikes2(dataArr)
%% I hated the previous version as it has a for loop dependent on previous results.
%% Making it hard to debug. This version fixes that (but it is not fully changed in the pipeline until 
%% fully tested with more cases. (i.e. biomech)
%
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
%   velocity of the heel marker is below a threshhold
% off is defined as last time before swing phase the total velocity is
%   above a threshold

    v_thresh_stance = 0.1; %threshold for "zero" velocity (% of max velocity)
    v_thresh_swing = 0.1; %threshold for swing phase (% of max velocity)
    
    v_x = gradient(dataArr(:,1));
    v_y = gradient(dataArr(:,2));
    v_z = gradient(dataArr(:,3));
    
    v_tot = sqrt(v_x.^2 + v_y.^2 + v_z.^2);
    v_tot = smoothdata(v_tot, 'movmedian', 50);
    v_tot = v_tot/max(abs(v_tot)); %normalize
    
    digitalTrigger=digitalSchmittTrigger(v_tot,'Vlow',0.1*max(v_tot),'Vhigh',0.4*max(v_tot));
    intervals=splitLogical(digitalTrigger);
    
    swingEvents=cellfun(@(x)(x(1)),intervals);
    strikeEvents=cellfun(@(x)(x(end)),intervals);
    
    %Get strikes: first time after swing phase where heelmarkervel<threshold    
    strikes=nan(1,numel(strikeEvents));
    for i=1:numel(strikeEvents)
        idx=strikeEvents(i);
        a=v_tot(idx:end);
        dh=max([0 find((a<v_thresh_stance),1)-1]);                
        strikes(i)=strikeEvents(i)+dh;
    end
    
    %Get offs: last time before swing phase where heelmarkervel>threshold    
    offs=nan(1,numel(swingEvents));
    for i=1:numel(swingEvents)
        idx=swingEvents(i);
        a=v_tot(1:idx);
        dh=min([idx find((a>v_thresh_stance)-1,1,'last')]);                
        offs(i)=dh;
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
    ends