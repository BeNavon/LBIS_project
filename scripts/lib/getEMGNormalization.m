function [minvalues,maxvalues]=getEMGNormalization(emgTreadmillTrials,speed)
% Get the min and max values of rectified emg at a selected speed (m/s) given a set of
% treadmill files.

idx=cellfun(@(trial)any(trial.conditions.speed.Speed==speed),emgTreadmillTrials);
trials1p2=emgTreadmillTrials(idx);
intervals=Topics.findTimes(@(x)(x.Speed==speed),trials1p2,{'conditions.speed'});
for i=1:numel(intervals)
    a=vertcat(intervals{i}.conditions.speed{:});
    intervalIdx=diff(a,1,2)>10;
    splitIntervals=a(intervalIdx,:);
    if isempty(splitIntervals)
        trials1p2{i}={};
    else
        a=Topics.segment(trials1p2{i},{splitIntervals},{'gcRight','emg','conditions.speed'});
        trials1p2{i}=a{1};
    end
end
trials1p2=trials1p2(cellfun(@(x)~isempty(x),trials1p2));

allstrides=[];
for i=1:numel(trials1p2)
    allstrides=[allstrides;segment_gc(trials1p2{i},'GCtopic','gcRight','GCchannel','HeelStrike')];
end
allstrides=Topics.normalize(allstrides,{'gcRight','emg'},'Header');
allstrides=Topics.interpolate(allstrides,0:0.01:1,{'gcRight','emg'});

meanvals=Topics.average(allstrides,'emg');
emgdata=meanvals.emg;
maxvalues=max(emgdata{:,2:end});minvalues=min(emgdata{:,2:end});