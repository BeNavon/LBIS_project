% Set the root directory here
rootDir = 'dataset';

% Recursively find all .mat files from the root directory down
matFiles = dir(fullfile(rootDir, '**', '*.mat'));

for k = 1:numel(matFiles)
    % Construct the full file path
    matFilePath = fullfile(matFiles(k).folder, matFiles(k).name);
    
    % Load the .mat file
    data = load(matFilePath);
    
    % Check if the .mat file contains a variable named 'M'
    if isfield(data, 'M')
        % Construct .csv file path by replacing .mat with .csv
        [folderPath, baseFileName, ~] = fileparts(matFilePath);
        csvFilePath = fullfile(folderPath, [baseFileName, '.csv']);
        
        % Write data.M to a CSV file
        csvwrite(csvFilePath, data.M);
        
        fprintf('Converted: %s -> %s\n', matFilePath, csvFilePath);
    else
        warning('Variable "M" not found in %s. Skipping...', matFilePath);
    end
end
