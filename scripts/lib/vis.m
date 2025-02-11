function vis(trial)
% Export the data for 3D visualization
f=FileManager('DATA/SPLIT','PathStructure',{'Subject','Sensor','File'});
mkdir('tmp');
Osim.writeMOT(trial.ik,'FilePath',['tmp' filesep 'ik.mot']);

Subject=trial.info.Subject;
osimfiles=f.fileList('Subject',Subject,'Sensor','osimxml','File',[Subject '.osim']);
osimfile=osimfiles{1};

copyfile(osimfile,['tmp' filesep Subject '.osim']);

trial.locrot=Osim.FK(trial.ik,osimfile,'OutputType','loc_rot','Transform','zup');
header=trial.ik.Header;
% Topics.interpolate(trial,'locrot');

trial.locrot.Header=(1:numel(header))';
writetable(trial.locrot,['tmp' filesep 'fk.anim'],'FileType','text');

%fpheader=trial.FPLeft.Header;
trial=Topics.interpolate(trial,header,{'FPLeft','FPRight'});
trial.fp=[trial.FPLeft(:,1:end-1) trial.FPRight(:,2:end-1)];
trial.fp{:,2:end}=Vicon.transform(trial.fp{:,2:end},'ViconXYZ');
trial.fp.Header=(1:numel(header))';
writetable(trial.fp,['tmp' filesep 'fp.anim'],'FileType','text');




end