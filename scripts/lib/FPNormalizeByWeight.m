function fptbl=FPNormalizeByWeight(fptbl,weight)
% For a table from FPLeft or FPRight divide the force and moment columns
% by the subjects weight (N).

% Last column is FP original name and center of pressure should be left
% unchanged.
fptbl{:,2:end-1}=fptbl{:,2:end-1}./[repmat(weight,1,3) 1 1 1  repmat(weight,1,3)];

end
