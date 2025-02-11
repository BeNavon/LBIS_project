function metrics=ComputeStrideMetrics(stride)
% For an individual stride, compute spatio-temporal metrics and return as a
% struct. For this example we compute the stridetime as the time and the
% stride length given by the euclidean distance between the heel marker at
% the initial heelstrike and the final heelstrike of the gait cycle.

metrics=struct();
metrics.strideTime=stride.ik.Header(end)-stride.ik.Header(1);
posHeelStart=stride.markers{1,{'R_Heel_x','R_Heel_y','R_Heel_z'}};
posHeelEnd=stride.markers{end,{'R_Heel_x','R_Heel_y','R_Heel_z'}};
metrics.strideLength=norm(posHeelStart-posHeelEnd,2);

end