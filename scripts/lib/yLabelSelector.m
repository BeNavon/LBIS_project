function out=yLabelSelector(sensor, channel)
% Return the correct y-label for a plot based on the channel

    channel = lower(channel);

    %% Segment
    segment=[];
    if contains(channel, 'ankle')
        segment = 'Ankle';
    elseif contains(channel, 'knee')
        segment = 'Knee';
    elseif contains(channel, 'hip')
        segment = 'Hip';
    elseif contains(channel, 'foot')
        segment = 'Foot';
    elseif contains(channel, 'shank')
        segment = 'Shank';
    elseif contains(channel, 'thigh')
        segment = 'Thigh';
    elseif contains(channel, 'trunk')
        segment = 'Trunk';
    end
    
    %% Types and units
    if contains(channel,'accel')
        units = 'Speed (m/s^2)';
    elseif contains(channel,'gyro')
        units = 'Angular Velocity (rad/sec)';
    elseif contains(channel,'right_v') || contains(channel,'left_v') || contains(channel, 'force')
        units = 'Force (N/kg)';
    elseif contains(channel,'right_p') || contains(channel,'left_p')
        units = 'Center of Pressure (mm)';
    elseif ~contains(sensor, 'id') && (contains(channel,'angle') || contains(channel, 'adduction') || contains(channel, 'flexion') || ...
           contains(channel, 'rotation') || contains(channel, 'bending') || contains(channel, 'extension') || ...
           contains(channel, 'list') || contains(channel, 'tilt'))
        units = 'Angle (deg)';
    elseif contains(channel, 'moment')
        units = 'Moment (Nm/kg)';
    elseif contains(channel, 'power')
        units = 'Power (W/kg)';
    elseif contains(sensor, 'emg')
        units = 'EMG';
    else
        units = 'unknown';
    end
    
    out = [segment, ' ', units];
end