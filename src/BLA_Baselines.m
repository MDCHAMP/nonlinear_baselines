%% BLA estimation on benchmark datasets
clear all
close all
clc


%% Wiener-Hammerstein
clear all


nx = 6; % state dimension - provided in benchmark documentation 
nxInit = 50;

% load and split data
dataWH = load('WienerHammerstein\WienerHammerBenchmark.mat');
iTrain = 1:105200;
iVal = [];
iTest = 105201:184000;

uTot = dataWH.uBenchMark;
uTrain = dataWH.uBenchMark(iTrain);
uVal = dataWH.uBenchMark(iVal);
uTest = dataWH.uBenchMark(iTest);

yTot = dataWH.yBenchMark;
yTrain = dataWH.yBenchMark(iTrain);
yVal = dataWH.yBenchMark(iVal);
yTest = dataWH.yBenchMark(iTest);

% substract means
um = mean(uTrain); uTrainBLA = uTrain - um;
ym = mean(yTrain); yTrainBLA = yTrain - ym;

% estimate OE SS model
dataTrain = iddata(yTrainBLA(:),uTrainBLA(:),1);
tic
BLA_WH = ssest(dataTrain, nx,'Ts',1,'Feedthrough',1,'Focus','simulation');

% evaluate performance
ySimTot = lsim(BLA_WH,uTot-um)+ym;
eRmsTrain = rms(ySimTot(iTrain)-yTrain);
eRmsTestDown = rms(ySimTot(iTest(nxInit+1:end))-yTest(nxInit+1:end));

time = toc;

disp(' ')
disp('Wiener-Hammerstein Results')
disp(['Train: ' num2str(eRmsTrain*1000) ' mV'])
disp(['Test:  ' num2str(eRmsTestDown*1000) ' mV'])
disp(['time taken during training and testing: ' num2str(time/60) ' min'])

% plot
figure; hold on;
plot(yTot)
plot(iTrain,yTrain-ySimTot(iTrain))
plot(iTest,yTest-ySimTot(iTest))

% save results
A = BLA_WH.A;
B = BLA_WH.B;
C = BLA_WH.C;
D = BLA_WH.D;
save BLA_WH BLA_WH A B C D 


%% Silverbox
clear all

nx = 2; % state dimension - provided in benchmark documentation 
nxInit = 50;

% load and split data
dataSB = load('Silverbox\SNLS80mV.mat');
iTrain = 40651:105712;
iVal = [];
iTestMS = 105713:127400;
iTestArr = 101:40575;
iTestArrNoExtra = 101:32100;

uTot = dataSB.V1;
uTrain = dataSB.V1(iTrain);
uVal = dataSB.V1(iVal);
uTestMS = dataSB.V1(iTestMS);
uTestArr = dataSB.V1(iTestArr);
uTestArrNoExtra = dataSB.V1(iTestArrNoExtra);

yTot = dataSB.V2;
yTrain = dataSB.V2(iTrain);
yVal = dataSB.V2(iVal);
yTestMS = dataSB.V2(iTestMS);
yTestArr = dataSB.V2(iTestArr);
yTestArrNoExtra = dataSB.V2(iTestArrNoExtra);

% substract means
um = mean(uTrain); uTrainBLA = uTrain - um;
ym = mean(yTrain); yTrainBLA = yTrain - ym;

% estimate OE SS model
dataTrain = iddata(yTrainBLA(:),uTrainBLA(:),1);
tic
BLA_SB = ssest(dataTrain, nx,'Ts',1,'Feedthrough',1,'Focus','simulation');

% evaluate performance
ySimTot = lsim(BLA_SB,uTot-um)+ym;
eRmsTrain = rms(ySimTot(iTrain)-yTrain');
eRmsVal = rms(ySimTot(iVal(nxInit+1:end))-yVal(nxInit+1:end)');
eRmsTestMS = rms(ySimTot(iTestMS(nxInit+1:end))-yTestMS(nxInit+1:end)');
eRmsTestArr = rms(ySimTot(iTestArr(nxInit+1:end))-yTestArr(nxInit+1:end)');
eRmsTestArrNoExtra = rms(ySimTot(iTestArrNoExtra(nxInit+1:end))-yTestArrNoExtra(nxInit+1:end)');
time = toc;

disp(' ')
disp('Silverbox Results')
disp(['Train: ' num2str(eRmsTrain*1000) ' mV'])
disp(['Test MS:  ' num2str(eRmsTestMS*1000) ' mV'])
disp(['Test Arr:  ' num2str(eRmsTestArr*1000) ' mV'])
disp(['Test Arr No Extra:  ' num2str(eRmsTestArrNoExtra*1000) ' mV'])
disp(['time taken during training and testing: ' num2str(time/60) ' min'])

% plot
figure; hold on;
plot(yTot)
plot(iTrain,yTrain'-ySimTot(iTrain))
plot(iVal,yVal'-ySimTot(iVal))
plot(iTestMS,yTestMS'-ySimTot(iTestMS))
plot(iTestArr,yTestArr'-ySimTot(iTestArr))
plot(iTestArrNoExtra,yTestArrNoExtra'-ySimTot(iTestArrNoExtra))

% save results
A = BLA_SB.A;
B = BLA_SB.B;
C = BLA_SB.C;
D = BLA_SB.D;
save BLA_SB BLA_SB A B C D 

%% Cascaded Tanks
clear all

nx = 2; % state dimension - provided in benchmark documentation  
nxInit = 50;

% load and split data
dataCT = load('CascadedTank_Benchmark2016\CascadedTanksFiles\dataBenchmark.mat');

uTrain = dataCT.uEst;
uVal = [];
uTest = dataCT.uVal;

yTrain = dataCT.yEst;
yVal = [];
yTest = dataCT.yVal;

% substract means
um = mean(uTrain); uTrainBLA = uTrain - um;
ym = mean(yTrain); yTrainBLA = yTrain - ym;

% estimate OE SS model
dataTrain = iddata(yTrainBLA(:),uTrainBLA(:),1);
tic
BLA_CT = ssest(dataTrain, nx,'Ts',1,'Feedthrough',1,'Focus','simulation','InitialState','estimate');
% initial state is estimated as well, and used during the simulations

% evaluate performance
ySimTrain = lsim(BLA_CT,uTrain-um,[],BLA_CT.x0)+ym;
% ySimVal = lsim(BLA_WH,uVal-um)+ym;
ySimTest = lsim(BLA_CT,uTest-um,[],BLA_CT.x0)+ym;

eRmsTrain = rms(ySimTrain-yTrain);
% eRmsVal = rms(ySimVal-yVal);
eRmsTestDown = rms(ySimTest(nxInit+1:end)-yTest(nxInit+1:end));
time = toc;

disp(' ')
disp('Cascaded Tanks Results')
disp(['Train: ' num2str(eRmsTrain) ' V'])
disp(['Test:  ' num2str(eRmsTestDown) ' V'])
disp(['time taken during training and testing: ' num2str(time/60) ' min'])

% plot
figure; tiledlayout(2,1); 
nexttile(); hold on;
plot(yTrain)
plot(ySimTrain)
plot(yTrain-ySimTrain)
legend('data','model','error')
title('Train')

nexttile(); hold on;
plot(yTest)
plot(ySimTest)
plot(yTest-ySimTest)
legend('data','model','error')
title('Test')

% save results
A = BLA_CT.A;
B = BLA_CT.B;
C = BLA_CT.C;
D = BLA_CT.D;
x0 = BLA_CT.x0;
save BLA_CT BLA_CT A B C D x0 


%% Coupled Electric Drives
clear all

nx = 3; % state dimension - does not take AA filter at output into account (see documentation)
% https://www.it.uu.se/research/publications/reports/2017-024/2017-024-nc.pdf
nxInit = 10;

% load and split data
dataCED = load('CoupledElectricDrive\DATAUNIF.MAT');

iTrain = 1:400;
iTest = 401:500;

uTot1 = dataCED.u11;
uTrain1 = dataCED.u11(iTrain);
uVal1 = [];
uTest1 = dataCED.u11(iTest);
uTot2 = dataCED.u11;
uTrain2 = dataCED.u12(iTrain);
uVal2 = [];
uTest2 = dataCED.u12(iTest);

yTot1 = dataCED.z11;
yTrain1 = dataCED.z11(iTrain);
yVal1 = [];
yTest1 = dataCED.z11(iTest);
yTot2 = dataCED.z12;
yTrain2 = dataCED.z12(iTrain);
yVal2 = [];
yTest2 = dataCED.z12(iTest);

% substract means
um = mean([uTrain1; uTrain2]); uTrainBLA1 = uTrain1 - um; uTrainBLA2 = uTrain2 - um;
ym = mean([yTrain1; yTrain2]); yTrainBLA1 = yTrain1 - ym; yTrainBLA2 = yTrain2 - ym;

% estimate OE SS model
dataTrain1 = iddata(yTrainBLA1(:),uTrainBLA1(:),1);
dataTrain2 = iddata(yTrainBLA2(:),uTrainBLA2(:),1);
dataTrain = merge(dataTrain1,dataTrain2);
tic
BLA_CED = ssest(dataTrain, nx,'Ts',1,'Feedthrough',1,'Focus','simulation','InitialState','estimate');
% initial state is estimated as well, and used during the simulations



% evaluate performance
ySimTot1 = lsim(BLA_CED,uTot1-um,[],BLA_CED.x0)+ym;
ySimTot2 = lsim(BLA_CED,uTot2-um,[],BLA_CED.x0)+ym;

eRmsTrain1 = rms(ySimTot1(iTrain)-yTrain1);
eRmsTrain2 = rms(ySimTot2(iTrain)-yTrain2);
eRmsTest1 = rms(ySimTot1(iTest(nxInit+1:end))-yTest1(nxInit+1:end));
eRmsTest2 = rms(ySimTot2(iTest(nxInit+1:end))-yTest2(nxInit+1:end));

time = toc;

disp(' ')
disp('Coupled Electric Drives Results')
disp(['Train: ' num2str([eRmsTrain1 eRmsTrain2]) ' '])
disp(['Test:  ' num2str([eRmsTest1 eRmsTest2]) ' '])
disp(['time taken during training and testing: ' num2str(time/60) ' min'])

% plot
figure; tiledlayout(2,1); 
nexttile(); hold on;
plot(yTot1)
plot(ySimTot1)
plot(iTrain,yTrain1-ySimTot1(iTrain))
plot(iTest,yTest1-ySimTot1(iTest))
legend('data','model','error train','error test')
title('Signal 1')

nexttile(); hold on;
plot(yTot2)
plot(ySimTot2)
plot(iTrain,yTrain2-ySimTot2(iTrain))
plot(iTest,yTest2-ySimTot2(iTest))
legend('data','model','error train','error test')
title('Signal 2')

% save results
A = BLA_CED.A;
B = BLA_CED.B;
C = BLA_CED.C;
D = BLA_CED.D;
x0 = BLA_CED.x0;
save BLA_CED BLA_CED A B C D x0 

%% EMPS
clear all

nx = 4; % state dimension, different from nx=2 suggested in the reference document. Obtained through validation, final estimate on full training+val dataset 
nxInit = 20;

% load and split data
dataEMPSTrain = load('2018-11 EMPS\DATA_EMPS.mat');
dataEMPSTest = load('2018-11 EMPS\DATA_EMPS_PULSES.mat');

iTrain = 1:24841;
iVal = [];
% iTrain = 1:12400;
% iVal = 12401:24841;
uTrainVal = dataEMPSTrain.vir;
yTrainVal = dataEMPSTrain.qm;
uTrain = dataEMPSTrain.vir(iTrain);
yTrain = dataEMPSTrain.qm(iTrain);
uVal = dataEMPSTrain.vir(iVal);
yVal = dataEMPSTrain.qm(iVal);

uTest = dataEMPSTest.vir;
yTest = dataEMPSTest.qm;

% substract means
um = mean(uTrain); uTrainBLA= uTrain - um;
ym = mean(yTrain); yTrainBLA= yTrain - ym;

% estimate OE SS model
dataTrain = iddata(yTrainBLA(:),uTrainBLA(:),1);
tic
BLA_EMPS = ssest(dataTrain, nx,'Ts',1,'Feedthrough',1,'Focus','simulation','InitialState','estimate');
% initial state is estimated as well, and used during the simulations

% evaluate performance
ySimTrainVal = lsim(BLA_EMPS,uTrainVal-um,[],BLA_EMPS.x0)+ym;
ySimTest = lsim(BLA_EMPS,uTest-um,[],BLA_EMPS.x0)+ym;

eRmsTrain = rms(ySimTrainVal(iTrain)-yTrain);
% eRmsVal = rms(ySimTrainVal(iVal)-yVal);
eRmsTestDown = rms(ySimTest(nxInit+1:end)-yTest(nxInit+1:end));

time = toc;

disp(' ')
disp('EMPS Results')
disp(['Train: ' num2str(eRmsTrain*1e3) ' mm'])
% disp(['Val:   ' num2str(eRmsVal*1e3) ' mm'])
disp(['Test:  ' num2str(eRmsTestDown*1e3) ' mm'])
disp(['time taken during training and testing: ' num2str(time/60) ' min'])

% plot
figure; tiledlayout(2,1); 
nexttile(); hold on;
plot(yTrain)
plot(ySimTrainVal)
plot(iTrain,yTrain-ySimTrainVal(iTrain))
% plot(iVal,yVal-ySimTrainVal(iVal))
legend('data','model','error train')
title('Train')

nexttile(); hold on;
plot(yTest)
plot(ySimTest)
plot(yTest-ySimTest)
legend('data','model','error')
title('Test')

% save results
A = BLA_EMPS.A;
B = BLA_EMPS.B;
C = BLA_EMPS.C;
D = BLA_EMPS.D;
x0 = BLA_EMPS.x0;
save BLA_EMPS BLA_EMPS A B C D x0 


%% parallel WH

clear all

nx = 12; % state dimension, different from nx=2 suggested in the reference document. Obtained through validation, final estimate on full training+val dataset 
nxInit = 50;

% load and split data
dataParWH = load('Parallel Wiener-Hammerstein Benchmark\ParWHData.mat');

uTrain = dataParWH.uEst;
yTrain = dataParWH.yEst;

uTest = dataParWH.uVal; uTest = reshape(uTest,32768,5);
yTest = dataParWH.yVal; yTest = reshape(yTest,32768,5);

% substract means
um = mean(mean(mean(mean(uTrain,4),3),2),1); uTrainBLA= uTrain - um;
ym = mean(mean(mean(mean(yTrain,4),3),2),1); yTrainBLA= yTrain - ym;
dataTrain = iddata(yTrainBLA(:,1,1,1),uTrainBLA(:,1,1,1),1);
for ia=1:size(uTrain,4)
    for im=1:1 %size(uTrain,3)
        dataTrainTemp = iddata(yTrainBLA(:,1,im,ia),uTrainBLA(:,1,im,ia),1);
        if im~=1 || ia~=1
            dataTrain = merge(dataTrain,dataTrainTemp);
        end
    end
end

% estimate OE SS model
tic
BLA_ParWH = ssest(dataTrain, nx,'Ts',1,'Feedthrough',1,'Focus','simulation','InitialState','estimate');

% % evaluate performance
% ySimTrainVal = lsim(BLA_EMPS,uTrainVal-um,[],BLA_EMPS.x0)+ym;
ySimTest = zeros(size(yTest));
for ia=1:5
    ySimTestTemp = lsim(BLA_ParWH,uTest(:,ia)-um)+ym;
    ySimTest(:,ia) = ySimTestTemp;

    eRmsTestDown(ia) = rms(ySimTestTemp(nxInit:end)-yTest(nxInit:end,ia));
end

% eRmsTrain = rms(ySimTrainVal(iTrain)-yTrain);
% eRmsTest = rms(ySimTest-yTest);
% 

time = toc;
disp(' ')
disp('ParWH Results')
% disp(['Train: ' num2str(eRmsTrain) ' V'])
for ia=1:5
    disp(['Test amplitude ' num2str(ia) ':  ' num2str(eRmsTestDown(ia)*1000) ' mV'])
end
disp(['time taken during training and testing: ' num2str(time/60) ' min'])


%% F16

clear all

nx = 20; % state dimension, different from nx=2 suggested in the reference document. Obtained through validation, final estimate on full training+val dataset 
nxInit = 50;
iy = 1;
nTrain = 8;
nTest = 6;

% load and split data
dataMSF16_1 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_FullMSine_Level1.mat');
dataMSF16_2 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_FullMSine_Level2_Validation.mat');
dataMSF16_3 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_FullMSine_Level3.mat');
dataMSF16_4 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_FullMSine_Level4_Validation.mat');
dataMSF16_5 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_FullMSine_Level5.mat');
dataMSF16_6 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_FullMSine_Level6_Validation.mat');
dataMSF16_7 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_FullMSine_Level7.mat');
dataSSF16_1 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_SineSw_Level1.mat');
dataSSF16_2 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_SineSw_Level2_Validation.mat');
dataSSF16_3 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_SineSw_Level3.mat');
dataSSF16_4 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_SineSw_Level4_Validation.mat');
dataSSF16_5 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_SineSw_Level5.mat');
dataSSF16_6 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_SineSw_Level6_Validation.mat');
dataSSF16_7 = load('F16GVT Benchmark Data\BenchmarkData\F16Data_SineSw_Level7.mat');


NMS = 8192; PMS = 9;
NSS = 108477; PSS=1;
downFactor = 8;
MSIndex = 1:PMS*NMS;
uTrainMS = zeros(length(MSIndex)/downFactor,4);
yTrainMS = zeros(length(MSIndex)/downFactor,4);


uTrainMS(:,1) = downsample(dataMSF16_1.Force(MSIndex),downFactor);
uTrainMS(:,2) = downsample(dataMSF16_3.Force(MSIndex),downFactor);
uTrainMS(:,3) = downsample(dataMSF16_5.Force(MSIndex),downFactor);
uTrainMS(:,4) = downsample(dataMSF16_7.Force(MSIndex),downFactor);

yTrainMS(:,1) = downsample(squeeze(dataMSF16_1.Acceleration(iy,MSIndex)),downFactor);
yTrainMS(:,2) = downsample(squeeze(dataMSF16_3.Acceleration(iy,MSIndex)),downFactor);
yTrainMS(:,3) = downsample(squeeze(dataMSF16_5.Acceleration(iy,MSIndex)),downFactor);
yTrainMS(:,4) = downsample(squeeze(dataMSF16_7.Acceleration(iy,MSIndex)),downFactor);
um = mean(mean(uTrainMS,2),1); 
ym = mean(mean(yTrainMS,2),1); 

 
% construct training data objects
ii=1;
uTrainBLA{ii} = uTrainMS(:,ii) - um;
yTrainBLA{ii} = yTrainMS(:,ii) - ym;
dataTrain = iddata(yTrainBLA{ii},uTrainBLA{ii},1);

ii=ii+1;
uTrainBLA{ii} = uTrainMS(:,ii) - um;
yTrainBLA{ii} = yTrainMS(:,ii) - ym;
dataTrainTemp = iddata(yTrainBLA{ii},uTrainBLA{ii},1);
dataTrain = merge(dataTrain,dataTrainTemp);

ii=ii+1;
uTrainBLA{ii} = uTrainMS(:,ii) - um;
yTrainBLA{ii} = yTrainMS(:,ii) - ym;
dataTrainTemp = iddata(yTrainBLA{ii},uTrainBLA{ii},1);
dataTrain = merge(dataTrain,dataTrainTemp);

ii=ii+1;
uTrainBLA{ii} = uTrainMS(:,ii) - um;
yTrainBLA{ii} = yTrainMS(:,ii) - ym;
dataTrainTemp = iddata(yTrainBLA{ii},uTrainBLA{ii},1);
dataTrain = merge(dataTrain,dataTrainTemp);

ii=ii+1;
uTrainSS = downsample(dataSSF16_1.Force,downFactor);
yTrainSS = downsample(squeeze(dataSSF16_1.Acceleration(iy,:)),downFactor);
uTrainBLA{ii}= uTrainSS' - um;
yTrainBLA{ii}= yTrainSS' - ym;
dataTrainTemp = iddata(yTrainBLA{ii},uTrainBLA{ii},1);
dataTrain = merge(dataTrain,dataTrainTemp);

ii=ii+1;
uTrainSS = downsample(dataSSF16_3.Force,downFactor);
yTrainSS = downsample(squeeze(dataSSF16_3.Acceleration(iy,:)),downFactor);
uTrainBLA{ii}= uTrainSS' - um;
yTrainBLA{ii}= yTrainSS' - ym;
dataTrainTemp = iddata(yTrainBLA{ii},uTrainBLA{ii},1);
dataTrain = merge(dataTrain,dataTrainTemp);

ii=ii+1;
uTrainSS = downsample(dataSSF16_5.Force,downFactor);
yTrainSS = downsample(squeeze(dataSSF16_5.Acceleration(iy,:)),downFactor);
uTrainBLA{ii}= uTrainSS' - um;
yTrainBLA{ii}= yTrainSS' - ym;
dataTrainTemp = iddata(yTrainBLA{ii},uTrainBLA{ii},1);
dataTrain = merge(dataTrain,dataTrainTemp);

ii=ii+1;
uTrainSS = downsample(dataSSF16_7.Force,downFactor);
yTrainSS = downsample(squeeze(dataSSF16_7.Acceleration(iy,:)),downFactor);
uTrainBLA{ii}= uTrainSS' - um;
yTrainBLA{ii}= yTrainSS' - ym;
dataTrainTemp = iddata(yTrainBLA{ii},uTrainBLA{ii},1);
dataTrain = merge(dataTrain,dataTrainTemp);

% construct testing data objects
ii = 1;
uTestSS = downsample(dataMSF16_2.Force,downFactor);
yTestSS = downsample(squeeze(dataMSF16_2.Acceleration(iy,:)),downFactor);
uTestBLADown{ii}= uTestSS' - um;
yTestBLADown{ii}= yTestSS' - ym;
uTestSS = dataMSF16_2.Force;
yTestSS = squeeze(dataMSF16_2.Acceleration(iy,:));
uTestBLA{ii}= uTestSS' - um;
yTestBLA{ii}= yTestSS' - ym;

ii = ii+1;
uTestSS = downsample(dataMSF16_4.Force,downFactor);
yTestSS = downsample(squeeze(dataMSF16_4.Acceleration(iy,:)),downFactor);
uTestBLADown{ii}= uTestSS' - um;
yTestBLADown{ii}= yTestSS' - ym;
uTestSS = dataMSF16_4.Force;
yTestSS = squeeze(dataMSF16_4.Acceleration(iy,:));
uTestBLA{ii}= uTestSS' - um;
yTestBLA{ii}= yTestSS' - ym;

ii = ii+1;
uTestSS = downsample(dataMSF16_6.Force,downFactor);
yTestSS = downsample(squeeze(dataMSF16_6.Acceleration(iy,:)),downFactor);
uTestBLADown{ii}= uTestSS' - um;
yTestBLADown{ii}= yTestSS' - ym;
uTestSS = dataMSF16_6.Force;
yTestSS = squeeze(dataMSF16_6.Acceleration(iy,:));
uTestBLA{ii}= uTestSS' - um;
yTestBLA{ii}= yTestSS' - ym;


ii = ii+1;
uTestSS = downsample(dataSSF16_2.Force,downFactor);
yTestSS = downsample(squeeze(dataSSF16_2.Acceleration(iy,:)),downFactor);
uTestBLADown{ii}= uTestSS' - um;
yTestBLADown{ii}= yTestSS' - ym;
uTestSS = dataSSF16_2.Force;
yTestSS = squeeze(dataSSF16_2.Acceleration(iy,:));
uTestBLA{ii}= uTestSS' - um;
yTestBLA{ii}= yTestSS' - ym;

ii = ii+1;
uTestSS = downsample(dataSSF16_4.Force,downFactor);
yTestSS = downsample(squeeze(dataSSF16_4.Acceleration(iy,:)),downFactor);
uTestBLADown{ii}= uTestSS' - um;
yTestBLADown{ii}= yTestSS' - ym;
uTestSS = dataSSF16_4.Force;
yTestSS = squeeze(dataSSF16_4.Acceleration(iy,:));
uTestBLA{ii}= uTestSS' - um;
yTestBLA{ii}= yTestSS' - ym;

ii = ii+1;
uTestSS = downsample(dataSSF16_6.Force,downFactor);
yTestSS = downsample(squeeze(dataSSF16_6.Acceleration(iy,:)),downFactor);
uTestBLADown{ii}= uTestSS' - um;
yTestBLADown{ii}= yTestSS' - ym;
uTestSS = dataSSF16_6.Force;
yTestSS = squeeze(dataSSF16_6.Acceleration(iy,:));
uTestBLA{ii}= uTestSS' - um;
yTestBLA{ii}= yTestSS' - ym;

% % estimate OE SS model
tic
BLA_F16 = ssest(dataTrain, nx,'Ts',1,'Feedthrough',1,'Focus','simulation','InitialState','estimate');

% simulating test responses
for ii=1:nTest
    SimTestTempDown = lsim(BLA_F16,[uTestBLADown{ii}; uTestBLADown{ii}; uTestBLADown{ii}]'-um)+ym;
    SimTestDown{ii} = SimTestTempDown(end/3*2+1:end);
    eRmsTestDown(ii) = rms(SimTestDown{ii}(nxInit+1:end)-yTestBLADown{ii}(nxInit+1:end));

    SimTestTemp = interp(SimTestTempDown,downFactor);
    SimTest{ii} = SimTestTemp(end/3*2+1:end);
    diff = length(SimTest{ii}) - length(yTestBLA{ii});
    eRmsTest(ii) = rms(SimTest{ii}(nxInit+1:end-diff)-yTestBLA{ii}(nxInit+1:end));
end

time = toc;
disp(' ')
disp('F19 Results')
for ii=1:nTest
%     disp(['Test signal down' num2str(ii) ':  ' num2str(eRmsTestDown(ii))])
    disp(['Test signal up  ' num2str(ii) ':  ' num2str(eRmsTest(ii))])
end
disp(['Mean Test RMS:  ' num2str(mean(eRmsTest))])
disp(['time taken during training and testing: ' num2str(time/60) ' min'])


% plotting
for ii=1:nTest
    figure; hold on;
    plot(db(fft(SimTestDown{ii})))
    plot(db(fft(yTestBLADown{ii})))
    plot(db(fft(yTestBLADown{ii}-SimTestDown{ii})))
end