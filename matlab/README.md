Supplementary Material for Pilot Contamination Attack Detection in Massive
MIMO Using Generative Adversarial Networks [Matlab code]
===

=== ChannelGeneration.m

```Matlab
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

```

=== functionRlocalscattering.m

```Matlab
%==========================================================================
% functionRlocalscattering
%==========================================================================
function R = functionRlocalscattering(M,theta,ASDdeg,antennaSpacing,distribution)
%Generate the spatial correlation matrix for the local scattering model,
%defined in (2.23) for different angular distributions.
%
%INPUT:
%M              = Number of antennas
%theta          = Nominal angle
%ASDdeg         = Angular standard deviation around the nominal angle
%                 (measured in degrees)
%antennaSpacing = (Optional) Spacing between antennas (in wavelengths)
%distribution   = (Optional) Choose between 'Gaussian', 'Uniform', and
%                'Laplace' angular distribution. Gaussian is default
%
%OUTPUT:
%R              = M x M spatial correlation matrix
%
%
%This Matlab function was developed to generate simulation results to:
%
%Emil Bjornson, Jakob Hoydis and Luca Sanguinetti (2017),
%"Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency",
%Foundations and Trends in Signal Processing: Vol. 11, No. 3-4,
%pp. 154-655. DOI: 10.1561/2000000093.
%
%For further information, visit: https://www.massivemimobook.com
%
%This is version 1.1 (Last edited: 2017-11-16)
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%monograph as described above.
%Set the antenna spacing if not specified by input
if  nargin < 4

    %Half a wavelength distance
    antennaSpacing = 1/2;

end
%Set angular distribution to Gaussian if not specified by input
if nargin<5
    distribution = 'Gaussian';
end
%Compute the ASD in radians based on input
ASD = ASDdeg*pi/180;
%The correlation matrix has a Toeplitz structure, so we only need to
%compute the first row of the matrix
firstRow = zeros(M,1);
%Go through all the columns of the first row
for column = 1:M

    %Distance from the first antenna
    distance = antennaSpacing*(column-1);


    %For Gaussian angular distribution
    if strcmp(distribution,'Gaussian')

        %Define integrand of (2.23)
        F = @(Delta)exp(1i*2*pi*distance*sin(theta+Delta)).*exp(-Delta.^2/(2*ASD^2))/(sqrt(2*pi)*ASD);

        %Compute the integral in (2.23) by including 20 standard deviations
        firstRow(column) = integral(F,-20*ASD,20*ASD);


    %For uniform angular distribution
    elseif strcmp(distribution,'Uniform')

        %Set the upper and lower limit of the uniform distribution
        limits = sqrt(3)*ASD;

        %Define integrand of (2.23)
        F = @(Delta)exp(1i*2*pi*distance*sin(theta+Delta))/(2*limits);

        %Compute the integral in (2.23) over the entire interval
        firstRow(column) = integral(F,-limits,limits);


    %For Laplace angular distribution
    elseif strcmp(distribution,'Laplace')

        %Set the scale parameter of the Laplace distribution
        LaplaceScale = ASD/sqrt(2);

        %Define integrand of (2.23)
        F = @(Delta)exp(1i*2*pi*distance*sin(theta+Delta)).*exp(-abs(Delta)/LaplaceScale)/(2*LaplaceScale);

        %Compute the integral in (2.23) by including 20 standard deviations
        firstRow(column) = integral(F,-20*ASD,20*ASD);

    end

end
%Compute the spatial correlation matrix by utilizing the Toeplitz structure
R = toeplitz(firstRow);
%==========================================================================
```
