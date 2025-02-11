function plotTrial(trial)
% Convenience function to plot a trial
%% Plot the full trial        
M=3; N=3;
i=0;
topics=Topics.topics(trial);


if ismember('ik',topics)
    i=i+1;
    subplot(M,N,i);
    Topics.plot(trial,'ik','channels','hip_flexion_l');  hold on;  
    if isfield(trial.conditions,'labels')
        Topics.plot(trial,'conditions.labels','Shaded',true); hold off;
    end
end

if ismember('gcRight',topics)
    i=i+1;
    subplot(M,N,i);
    Topics.plot(trial,'gcRight');    
end

if ismember('gcLeft',topics)
    i=i+1;
    subplot(M,N,i);
    Topics.plot(trial,'gcLeft');    
end

if ismember('id',topics)
    i=i+1;
    subplot(M,N,i);
    Topics.plot(trial,'id','channels','hip_flexion_l_moment'); 
end

if ismember('FPLeft',topics)
    i=i+1;
    subplot(M,N,i);
    Topics.plot(trial,'FPLeft','channels','Left_vy'); hold on;
    Topics.plot(trial,'FPLeft','channels','OriginalForceplate','Shaded',true); hold off;
end


if ismember('FPRight',topics)
    i=i+1;
    subplot(M,N,i);
    Topics.plot(trial,'FPRight','channels','Right_vy'); hold on;
    Topics.plot(trial,'FPRight','channels','OriginalForceplate','Shaded',true); hold off;
end

i=i+1;
subplot(M,N,i);
Topics.plot(trial,'emg','channels','vastuslateralis'); 
i=i+1;
subplot(M,N,i);
Topics.plot(trial,'imu','channels','foot_Accel_X'); 
end