%% Batch processing on subject by subject basis
subjects=compose('AB%02d',(setdiff(6:30,[22,26,29])));

% A local mat file to preserve data amongst multiple re-runs of this
if exist('log.mat','file')
load('log.mat'); 
else
    tbl=array2table(cell(0,2), 'VariableNames',{'Subject','Error'});
    problems=tbl;
end

for subjectIdx=1:numel(subjects)    
    SUBJECT=subjects{subjectIdx};
    fprintf('Processing subject %s\n',SUBJECT);
				
          run RUN_OSIM;
          run STRIDES;   
end