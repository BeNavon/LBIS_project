function tabledata=rectify(tabledata)
% Rectify the data from a table by applying low pass filter -> abs -> lowpass
    sensorData=tabledata{:,2:end};
    FS=mean(1./diff(tabledata.Header));
    df0 = designfilt('highpassiir','FilterOrder',4,'Halfpowerfrequency',10,'SampleRate',FS,'DesignMethod','butter');
    df1 = designfilt('bandpassfir','FilterOrder',20,'CutoffFrequency1',10,'CutoffFrequency2',450,'SampleRate',FS);
    df2 = designfilt('lowpassiir','FilterOrder',4,'Halfpowerfrequency',6,'SampleRate',FS,'DesignMethod','butter');
    sensorData=filtfilt(df0,sensorData);        
    sensorData=filtfilt(df1,sensorData);
    sensorData=abs(sensorData);
    sensorData=filtfilt(df2,sensorData);
    sensorData=abs(sensorData);    
    tabledata{:,2:end}=sensorData;    
end
