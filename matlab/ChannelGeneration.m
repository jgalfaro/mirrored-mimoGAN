%==========================================================================
% This code generates training data and validation data for GAN
%==========================================================================
close all;
clear;

%------------------------ Simulation parameters----------------------------
M = 64;                                %Number of BS antennas
K=8;                                   %Number of users
ASD = 10;                              %Angular standard deviation in the local scattering model (in degrees)
varphiRange = linspace(-pi,+pi,50);    %Define the range of nominal angles of arrival
antennaSpacing = 1/2;                  %Half wavelength distance
Dataset_size=4000;
Attack_Data=1000;
snr=0;
%--------------------------- Pilot generation------------------------------
Pilot= hadamard(K);

% ------------------------Trainin data generation--------------------------

% Generate correlation matrix
for r = 1:length(varphiRange)
    %Output simulation progress
    disp([num2str(r) ' angles out of ' num2str(length(varphiRange))]);    
    %Compute the spatial correlation matrix
    R = functionRlocalscattering(M,varphiRange(r),ASD,antennaSpacing);
    %Compute square root of the spatial correlation matrix
    Rsqrt = sqrtm(R);
end

%Generate uncorrelated channel realizations
uncorrelatedRealizations1 = (randn(M,K,Dataset_size)+1i*randn(M,K,Dataset_size))/sqrt(2);  
% generate real data
Channels=zeros(Dataset_size,M,K);
Y=zeros(Dataset_size,M,K);
Y_Data=zeros(Dataset_size,M,K);
for i=1:Dataset_size            
    Channels(i,:,:) = normalize(Rsqrt*uncorrelatedRealizations1(:,:,i),'scale'); %  normalize by its standard deviation
    CH(:,:)=Channels(i,:,:);
    Y(i,:,:)=CH(:,:)*Pilot;
    Y_Data(i,:,:)=awgn(Y(i,:,:),snr,'measured');
    
end

% generate abnormal data
Y_att=zeros(Attack_Data,M,K);
Channels_att=zeros(Attack_Data,M,1);
Y_abnormal=zeros(Attack_Data,M,K);
uncorrelatedRealizations2 = (randn(M,K,Attack_Data)+1i*randn(M,K,Attack_Data))/sqrt(2);
Channels_test=zeros(Attack_Data,M,K);
Y_real=zeros(Attack_Data,M,K);

for j=1:Attack_Data
    Channels_test(j,:,:) = normalize(Rsqrt*uncorrelatedRealizations2(:,:,j),'scale'); %  normalize by its standard deviation
    CH2(:,:)=Channels_test(j,:,:);
    Y_real(j,:,:)=CH2(:,:)*Pilot;
    Channels_att(j,:,1) = normalize(Rsqrt*uncorrelatedRealizations2(:,1,j),'scale');
    CH_att(:,1)=Channels_att(j,:,1);
    Y_att(j,:,:)=CH(:,1)*Pilot(1,:);
    Y_abnormal(j,:,:)=Y_real(j,:,:)+Y_att(j,:,:);
    Y_abnormal(j,:,:)=awgn(Y_abnormal(j,:,:),snr,'measured');
end


% ------------Extract real and imaginary part of real data---------------------- 
In_Data(:,:,:,1) = real(Y_Data); % real part of Y
In_Data(:,:,:,2) = imag(Y_Data); % imag papt of Y

Out_Data(:,:,:,1) = real(Channels); % real part of H
Out_Data(:,:,:,2) = imag(Channels); % imag part of H

Test_Channel(:,:,:,1) = real(Channels_test); % real part of H
Test_Channel(:,:,:,2) = imag(Channels_test); % imag part of H


Test_Data(:,:,:,1) = real(Y_abnormal); % real part of H
Test_Data(:,:,:,2) = imag(Y_abnormal); % imag part of H

% Shuffle data 
%Y_Data = shuffle(randperm(length(Y_Data)));
%Channels_Data = shuffle(randperm(length(Channels_Data)));

% --------------------------save data--------------------------------------
 
% Visualization of Y and H
figure
imshow(squeeze(In_Data(1,:,:,1)))
title('Visualization of Y')
figure
imshow(squeeze(Out_Data(1,:,:,1)))
title('Visualization of H')
figure
imshow(squeeze(Test_Data(1,:,:,1)))
title('Visualization of T')
 
save('In_Data.mat','In_Data');
save('Out_Data.mat','Out_Data');
save('Test_Data.mat','Test_Data');
save('Test_Channel.mat','Test_Channel');
