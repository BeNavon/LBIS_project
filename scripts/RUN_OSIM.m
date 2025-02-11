init(); def=defaults;
%% Use the marker and forceplate data to compute inverse kinematics and inverse dynamics programatically. 
allfiles=f.fileList('Subject',SUBJECT,'Trial',[TRIAL '.mat']);
trials=f.EpicToolbox(allfiles);
osim=fosim.fileList('Subject',SUBJECT,'File',[SUBJECT,'.osim']);
ikfile=fosim.fileList('Subject',SUBJECT,'File','IKSetup.xml');
idfile=fosim.fileList('Subject',SUBJECT,'File','IDSetup.xml');
%%
for i=1:numel(trials)
   trial=trials{i};         
   fprintf('Processing trial %s\n',trial.info.Trial);
   
   % Skip if static ik is already present
   staticIKFile=f.fileList('Subject',SUBJECT,'Sensor','ik','Trial','static*.mat');
   if  numel(staticIKFile)==0
      staticMarkersFile=f.fileList('Subject',SUBJECT,'Sensor','markers','Trial','static*.mat');
      staticMarkers=load(staticMarkersFile{1});
      Osim.editSetupXML(ikfile{1},'model_file',osim{1},'FilePath',ikfile{1})
      staticIK=Osim.IK(staticMarkers.data,ikfile{1});
      staticIKFile=f.modFileList(staticMarkersFile,'Sensor','ik');
      data=staticIK;
      mkdirfile(staticIKFile{1});
      save(staticIKFile{1},'data');
   end
   
      
   % Skip if ik is already present DATA   
   if  ~isfield(trial,'ik') || def.OVERWRITE                       
      %% Compute Inverse kinematics using the configuration IKSetup.xml      
      Osim.editSetupXML(ikfile{1},'model_file',osim{1},'FilePath',ikfile{1})
      ik=Osim.IK(trial.markers,ikfile{1});      
      trial.ik=ik;
   end
     
   % Skip if id is already present
   if  (~isfield(trial,'id') || def.OVERWRITE) && ~strcmp(trial.info.Mode,'static')
       %% Compute Inverse dynamics using the configuration IDSetup.xml (global forceplate data 'fp')
       if strcmpi(trial.info.Mode,'ramp') % In ramp trials subjects might use any foot on FP 5
            id = RampID(trial, idfile{1});
       else
            deviceNames = strrep(trial.fp.Properties.VariableNames(2:9:end), '_vx', '');
            sides = Osim.correlateForcePlates(trial.markers, trial.fp);
            extLoads = Osim.createExternalLoads(deviceNames, sides);           
            idSetupFile = Osim.editSetupXML(idfile{1}, 'external_loads_file', extLoads);
            id = Osim.ID(trial.fp, idSetupFile, trial.ik);        
       end            
       id{:, 2:end} = Vicon.Filter(id{:, 2:end}, 6*2*mean(diff(id.Header))); 
       trial.id=id; 
   
       %% Remove regions with invalid ID data (unkown external forces)
       if ~strcmpi(trial.info.Mode,'treadmill')
           trial=RemoveInvalidRegions(trial);
       end   
   end

   %% Compute the joint power based on ik and id
   if  (~isfield(trial,'jp') || def.OVERWRITE) &&  ~strcmp(trial.info.Mode,'static')
       %% Compute the joint power based on ik and id
       jp = calculateJointPowers(trial);    
       trial.jp=jp;  
   end
   %%  Angles relative to tpose
   if  (~isfield(trial,'ik_offset') || def.OVERWRITE)
       ik=trial.ik;
       ik_offset=ik;
       staticIKFile=f.fileList('Subject',SUBJECT,'Sensor','ik','Trial','static*.mat');
       ik0=load(staticIKFile{1}); ik0=ik0.data;
       ik_offset{:,2:end}=ik{:,2:end}-mean(ik0{:,2:end});       
       trial.ik_offset=ik_offset;
   end   
              
   trial=Topics.select(trial,{'ik','id','jp','ik_offset','info'});   
   save_sfpost(trial,'FileManager',f,'Overwrite',def.OVERWRITE);
   
end