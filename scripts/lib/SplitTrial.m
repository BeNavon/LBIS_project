function allsegments=SplitTrial(trial)
%% For a sensor fusion trial divide it based on start and end trials times

   conditions=trial.conditions;
   %Construct an intervals cell array   
   intervals_array=[conditions.trialStarts conditions.trialEnds];
   
   lastTime=trial.markers.Header(trial.markers.Header<= intervals_array(end,2)); lastTime=lastTime(end);
   intervals_array(end,2)=lastTime;
   
   intervals=mat2cell(intervals_array,ones(size(intervals_array,1),1),2);   
   segments=Topics.segment(trial,intervals,Topics.topics(trial,'Header',true));   
   
   segmentNames=compose([trial.info.Trial(1:end-4) '_%02d'],1:numel(segments));
   allsegments={};
   for j=1:numel(segments)
      fprintf('Generating segment %s\n',segmentNames{j});
      segment=segments{j};
      segment.info.Trial=segmentNames{j};
      %Create the conditions for each segment
      c=conditions;
      interval=intervals{j};
      c.trialStarts=interval(1); c.trialEnds=interval(end);
      c=Topics.cut(c,interval(1),interval(2),{'labels'});
      if isfield(c,'leadingLegStart')
          c.leadingLegStart=c.leadingLegStart(j);c.leadingLegStop=c.leadingLegStop(j);      
      end
      if isfield(c,'transLegAscent')
          c.transLegAscent=c.transLegAscent(:,j);c.transLegDescent=c.transLegDescent(:,j);
      end      
      
      
      segment.conditions=c;      
      allsegments=[allsegments;{segment}];      
   end       

end