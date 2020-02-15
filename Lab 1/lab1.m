%E205 Lab 1
% Jane Watts and Tim Player
% 2 February 2020

%$ 1. import data
load('lab1_data.mat')

%% 2. histograms
figure(1)
subplot(1,3,3)
histogram(lab1azimuth90.Rangem, 28)
title('2c) Azimuth 90 deg Histogram')
xlabel('Range Measurement (m)')
ylabel('Number of Measurements (counts)')

subplot(1,3,2)
histogram(lab1azimuth00.Rangem, 28)
title('2b) Azimuth 00 deg Histogram')
xlabel('Range Measurement (m)')
ylabel('Number of Measurements (counts)')

subplot(1,3,1)
histogram(lab1azimuthneg90.Rangem, 28)

%% 4. Create Model
subplot(1,3,1)
h = histfit(lab1azimuthneg90.Rangem, 28);
title('2a) Azimuth -90 deg Histogram')
xlabel('Range Measurement (m)')
ylabel('Number of Measurements (counts)')
pd = fitdist(lab1azimuthneg90.Rangem,'Normal')
%pd = makedist('Normal','mu',mean(lab1azimuthneg90.Rangem),'sigma',std(lab1azimuthneg90.Rangem));
%pdf(pd,lab1azimuthneg90.Rangem)
legend('Range Data','Histogram Fit')

%% 5. Transform & Plot GPS
Rearth = 6.371*10^6; %meters
LatitudeRad = lab1azimuth00.Latitude * 2*pi/360; %dataset in degrees ==> radians
LongitudeRad = lab1azimuth00.Longitude * 2*pi/360; 

% Forward Equirectangular Projection Method from E80 Lab 7
% https://sites.google.com/g.hmc.edu/e80/labs/lab-7-navigation?authuser=0
Ymeters = Rearth * (LatitudeRad - mean(LatitudeRad));
Xmeters = Rearth * cos(mean(LatitudeRad)) * (LongitudeRad - mean(LongitudeRad)); 

figure(2)
scatter(Xmeters,Ymeters, 'filled')
title('5) Azimuth 0 deg GPS Measurements')
xlabel('X (East) (m)')
ylabel('Y (North) (m)')
Xvar = var(Xmeters)
Yvar = var(Ymeters)