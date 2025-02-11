function gaitData=GaitCycleFromTRC(trcData,varargin)
% Generate gait cycle from the TRC
%
% Given TRC data file, uses the the data from an individual marker data to generate gait
% cycle signal from 0-100%.
% returns gait as table
% 
% 'Markers'   {'R_Heel'}    List of markers to use for generating gait each marker
% will output one gait column.
% 'ToeOff' true/(false) Compute toe off instead of heel strike.

trcData = Osim.interpret(trcData, 'TRC');

p = inputParser;   
addParameter(p,'Markers',{'R_Heel'},@(x)(iscell(x) || ischar(x)));
addParameter(p,'ToeOff',false,@islogical);
parse(p,varargin{:});

ToeOff=p.Results.ToeOff;
markerNames=p.Results.Markers;
if ~iscell(markerNames) && ischar(markerNames)
    markerNames={markerNames};
end

gaitNames=cell(size(markerNames));
for i=1:numel(gaitNames)
    if ToeOff
        gaitNames{i}=[markerNames{i} '_Off'];
    else
        gaitNames{i}=[markerNames{i} '_Strike'];
    end
end


    gait=zeros(height(trcData),numel(markerNames));
    for markerIdx=1:numel(markerNames)        
        marker_x=smoothdata(trcData.([markerNames{markerIdx},'_x']), 'movmean', 10);
        marker_y=smoothdata(trcData.([markerNames{markerIdx},'_y']), 'movmean', 10);
        marker_z=smoothdata(trcData.([markerNames{markerIdx},'_z']), 'movmean', 10);                
        [strikes,offs] = getStrikes([marker_x, marker_y, marker_z]);
        if ToeOff                        
              f = [0, offs, size(marker_x,1)];
              gaitArrOffs = [];
              for i = 2:length(f)
                gaitArrOffs=[gaitArrOffs; linspace(0,100,(f(i)-f(i-1)))']; 
              end
              gait(:,markerIdx) = gaitArrOffs;
        else
               f = [0, strikes, size(marker_x,1)];
              gaitArrStrikes = [];
              for i = 2:length(f)
                gaitArrStrikes=[gaitArrStrikes; linspace(0,100,(f(i)-f(i-1)))']; 
              end
              gait(:,markerIdx) = gaitArrStrikes;
        end            

    end    

    gaitData=array2table([trcData.Header gait],'VariableNames',['Header',gaitNames]);
end