function jointPower = calculateJointPowers(trial)
% calculate joint powers given joint angles and joint torques by taking
% numerical derivative of angles and dot multiplying with torques. 
% Output is a table with the same columns, but with '_power' added to the
% end. 
% jointPower = calculateJointPowers(trial)
    
    ikData = trial.ik;
    idData = trial.id;
    f0 = 6; %Hz
    assert(isequal(size(ikData), size(idData)), 'IK and ID data must be the same size.');
    momentNames = idData.Properties.VariableNames;
    angleNames = strrep(momentNames, '_moment', '');
    angleNames = strrep(angleNames, '_force', '');
    % reorder columns to be in same order
    ikData = ikData(:, angleNames);

    thetaDot = ikData;
    for idx = 2:width(thetaDot)
        angle = thetaDot{:, idx};
        angle = gradient(angle)./gradient(thetaDot.Header);
        angle = deg2rad(angle);
        thetaDot{:, idx} = angle;
    end
    jointPower = thetaDot;
    jointPower{:, 2:end} = idData{:, 2:end} .* thetaDot{:, 2:end};
    labels = jointPower.Properties.VariableNames(2:end)';
    labels = compose('%s_power', string(labels));
    jointPower.Properties.VariableNames(2:end) = labels;
    jointPower{:, 2:end} = Vicon.Filter(jointPower{:, 2:end}, mean(diff(jointPower.Header)) * f0 * 2);
end