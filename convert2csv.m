% Root directory (adjust as needed)
rootDir = 'dataset';

% Locate all .mat files in the treadmill/imu and treadmill/gcRight paths
imuFiles = dir(fullfile(rootDir, '*', 'treadmill', 'imu', '*.mat'));
gcRightFiles = dir(fullfile(rootDir, '*', 'treadmill', 'gcRight', '*.mat'));

% Combine the two lists
matFiles = [imuFiles; gcRightFiles];

for k = 1:numel(matFiles)
    % Construct the full path to the .mat file
    matFilePath = fullfile(matFiles(k).folder, matFiles(k).name);
    
    % Load all variables from this .mat file into a struct
    dataStruct = load(matFilePath);
    
    % Retrieve the names of the variables in the .mat file
    varNames = fieldnames(dataStruct);
    
    for v = 1:numel(varNames)
        varData = dataStruct.(varNames{v});
        
        if istable(varData)
            % Build output filename
            [folderPath, baseName, ~] = fileparts(matFilePath);
            csvName = sprintf('%s_%s.csv', baseName, varNames{v});
            csvPath = fullfile(folderPath, csvName);
            
            writetable(varData, csvPath);
            fprintf('Converted table: %s (variable "%s") -> %s\n', ...
                matFilePath, varNames{v}, csvPath);
        
        elseif isnumeric(varData) && ismatrix(varData)
            % Build output filename
            [folderPath, baseName, ~] = fileparts(matFilePath);
            csvName = sprintf('%s_%s.csv', baseName, varNames{v});
            csvPath = fullfile(folderPath, csvName);
        
            writematrix(varData, csvPath);
            fprintf('Converted numeric matrix: %s (variable "%s") -> %s\n', ...
                matFilePath, varNames{v}, csvPath);
        
        else
            fprintf('Skipping variable "%s" in %s (not numeric 2D or table)\n', ...
                varNames{v}, matFiles(k).name);
        end
    end
end
