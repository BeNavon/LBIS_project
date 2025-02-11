function gaitArrStrikes=StrikesToGC(strikes)
%% Generate Gait Phase in the range of 0-100% based on the strikes vector
% Function that takes in a Nx3 array of marker data and generates gait cycle
% using "strikes" found by getHeelStrikes(heelDataArr)
% see also getStrikes
  
    f = [0, strikes, size(strikes,1)];
    gaitArrStrikes = [];
    for i = 2:length(f)
        gaitArrStrikes=[gaitArrStrikes; linspace(0,100,(f(i)-f(i-1)))']; 
    end
    
end