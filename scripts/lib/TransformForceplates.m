function trial=TransformForceplates(trial)
% For a trial with markers and fp reformat the forceplate data to bipedal
% l/r and a column with forceplate name.
%%
lheelcols={'Header','L_Heel_x','L_Heel_y','L_Heel_z'};
rheelcols={'Header','R_Heel_x','R_Heel_y','R_Heel_z'};

if strcmpi(trial.info.Mode,'treadmill')    
    trial.LeftFPNames=Osim.correlateForcePlates2(trial.markers(:,lheelcols),trial.corners,trial.fp,'Limits',[10,10,200]);
    trial.RightFPNames=Osim.correlateForcePlates2(trial.markers(:,rheelcols),trial.corners,trial.fp,'Limits',[10,10,200]);
elseif strcmpi(trial.info.Mode,'stair') %Subject might land the heel marker in air (heel outside of forceplate)
    trial.LeftFPNames=Osim.correlateForcePlates2(trial.markers(:,lheelcols),trial.corners,trial.fp,'Limits',[100,100,100]);
    trial.RightFPNames=Osim.correlateForcePlates2(trial.markers(:,rheelcols),trial.corners,trial.fp,'Limits',[100,100,100]);
else
    trial.LeftFPNames=Osim.correlateForcePlates2(trial.markers(:,lheelcols),trial.corners,trial.fp,'Limits',[10,10,200]);
    trial.RightFPNames=Osim.correlateForcePlates2(trial.markers(:,rheelcols),trial.corners,trial.fp,'Limits',[10,10,200]);
end

%Topics.plot(trial,'LeftFPNames','Shaded',true);

%% Plot Vertical force and correlated forceplate
%{ 
subplot(2,1,1);
Topics.plot(trial,'fp','channels',{'FP.*_vy'}); hold on;
Topics.plot(trial,'LeftFPNames','Shaded',true);
subplot(2,1,2);
Topics.plot(trial,'fp','channels',{'FP.*_vy'}); hold on;
Topics.plot(trial,'RightFPNames','Shaded',true);
%}
%%

trial=Topics.interpolate(trial,trial.fp.Header,{'LeftFPNames','RightFPNames'});

x=zeros(size(trial.fp,1),9);
colstr='%s_vx %s_vy %s_vz %s_px %s_py %s_pz %s_moment_x %s_moment_y %s_moment_z';

[u_fpnames,~,idx]=unique(trial.LeftFPNames.Forceplate);
for i=1:numel(u_fpnames)
    if strcmp(u_fpnames{i},'NONE')
        continue;
    end
    colnames=strsplit(strrep(colstr,'%s',u_fpnames{i}),' ');
    x(idx==i,:)=trial.fp{idx==i,colnames};
end
colnames=strsplit(strrep(colstr,'%s','Left'),' ');
trial.FPLeft=array2table([trial.fp.Header x],'VariableNames',['Header',colnames]);
trial.FPLeft.OriginalForceplate=trial.LeftFPNames.Forceplate;

x=zeros(size(trial.fp,1),9);
colstr='%s_vx %s_vy %s_vz %s_px %s_py %s_pz %s_moment_x %s_moment_y %s_moment_z';
[u_fpnames,~,idx]=unique(trial.RightFPNames.Forceplate);
for i=1:numel(u_fpnames)
    if strcmp(u_fpnames{i},'NONE')
        continue;
    end
    colnames=strsplit(strrep(colstr,'%s',u_fpnames{i}),' ');
    x(idx==i,:)=trial.fp{idx==i,colnames};
end
colnames=strsplit(strrep(colstr,'%s','Right'),' ');
trial.FPRight=array2table([trial.fp.Header x],'VariableNames',['Header',colnames]);
trial.FPRight.OriginalForceplate=trial.RightFPNames.Forceplate;

end


