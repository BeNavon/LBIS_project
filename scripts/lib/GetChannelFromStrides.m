function out=GetChannelFromStrides(strides,sensor,channel)
% Helper function to retrieve one channel from one sensor as an array
% First dimension is the header, second dimension is the stride        

x=Topics.select(strides,sensor,'Channel',channel);

if iscell(x)        
    x=cellfun(@(x)(x.(sensor).Variables),x,'Uniform',0); 
    out=horzcat(x{:});
else
    out=x.(sensor).Variables;
end