function condition=GetConditionFromStrides(strides)
% Helper function to retrieve the condition from strides of data
if iscell(strides)
    condition=cellfun(@GetConditionFromStrides,strides);    
    return;
else
    stride=strides;
end

switch stride.info.Mode
    
    case 'treadmill'
        speedIntervals=0.475:0.05:1.875;
        speedIntervalMean=(speedIntervals(1:end-1)+speedIntervals(2:end))/2;
        x=mean(stride.conditions.speed.Speed);
        idx=discretize(x,speedIntervals);
        if isnan(idx)
            condition=nan;
        else
            condition=speedIntervalMean(idx);
        end
        
    case 'stair'
        condition=stride.conditions.stairHeight;
        
    case 'ramp'
        condition=stride.conditions.rampIncline;
        
    case 'levelground'
        condition=stride.conditions.speed;
        switch condition
            case 'slow'
                condition=1;
            case 'normal'
                condition=2;
            case 'fast'
                condition=3;
        end        
            
        
end        

end