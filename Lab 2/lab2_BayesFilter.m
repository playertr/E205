%E205 Lab 2
% Jane Watts and Tim Player
% 19 February 2020

%$ 1. import data
load('lab2_data.mat')

%% 2. vehicle speed histograms
figure(1)
% Vehicle 4
subplot(2,2,1)
t_4 = [15:41];
t_prev_4 = [14:40];
delta_t = 0.5; %seconds
S_4 = speed(E205Lab2NuScenesData.X_4, E205Lab2NuScenesData.Y_4, t_4, t_prev_4,delta_t);
histfit(S_4,6);
title('Vehicle 4 Speed Histogram')
xlabel('Speed (m/s)')
ylabel('Number of Measurements (counts)')
legend('Speed Data','Histogram Fit')
p_s_x_stopped = fitdist(S_4,'Normal');

% Vehicle 2
subplot(2,2,2)
t_2 = [4:28];
t_prev_2 = [3:27];
S_2 = speed(E205Lab2NuScenesData.X_2, E205Lab2NuScenesData.Y_2, t_2, t_prev_2,delta_t);
histfit(S_2,9);
title('Vehicle 2 Speed Histogram')
xlabel('Speed (m/s)')
ylabel('Number of Measurements (counts)')
legend('Speed Data','Histogram Fit')

% Vehicle 3
subplot(2,2,3)
t_3 = [8:39];
t_prev_3 = [7:38];
S_3 = speed(E205Lab2NuScenesData.X_3, E205Lab2NuScenesData.Y_3, t_3, t_prev_3,delta_t);
histfit(S_3,6);
title('Vehicle 3 Speed Histogram')
xlabel('Speed (m/s)')
ylabel('Number of Measurements (counts)')
legend('Speed Data','Histogram Fit')

% Vehicle 5
subplot(2,2,4)
t_5 = [7:41];
t_prev_5 = [6:40];
S_5 = speed(E205Lab2NuScenesData.X_5, E205Lab2NuScenesData.Y_5, t_5, t_prev_5,delta_t);
histfit(S_5,8);
title('Vehicle 5 Speed Histogram')
xlabel('Speed (m/s)')
ylabel('Number of Measurements (counts)')
legend('Speed Data','Histogram Fit')

% All Moving Vehicles Speed Data
S_Moving = [S_2; S_3; S_5];
figure(2)
subplot(2,1,1)
histfit(S_4,6);
title('Stopped Vehicle Speed Histogram')
xlabel('Speed (m/s)')
ylabel('Number of Measurements (counts)')
legend('Speed Data','Histogram Fit')

subplot(2,1,2)
histfit(S_Moving,12);
title('Moving Vehicles (Vehicles 2,3,5) Speed Histogram')
xlabel('Speed (m/s)')
ylabel('Number of Measurements (counts)')
legend('Speed Data','Histogram Fit')
p_s_x_notstopped = fitdist(S_Moving,'Normal');

% Ego Vehicle
t_ego = [3:41];
t_prev_ego = [2:40];
S_ego = speed(E205Lab2NuScenesData.X_ego, E205Lab2NuScenesData.Y_ego, t_ego, t_prev_ego,delta_t);

% Vehicle 1
t_1 = [3:13];
t_prev_1 = [2:12];
S_1 = speed(E205Lab2NuScenesData.X_1, E205Lab2NuScenesData.Y_1, t_1, t_prev_1,delta_t);

% Vehicle 6
t_6 = [20:41];
t_prev_6 = [19:40];
S_6 = speed(E205Lab2NuScenesData.X_1, E205Lab2NuScenesData.Y_1, t_1, t_prev_1,delta_t);
%% Bayes Filter
% initialization
bel_x_prev = [0.5; 0.5];
p_x_xprev = [0.6, 0.25; 0.4, 0.75]; % transition matrix; values given in lab 2 manual
myspeed = S_ego; % speed vector for vehicle selected for filter
estimate = [];

for t = 1:length(myspeed)
estimate = [estimate, bel_x];

% prediction step
bel_bar_x = p_x_xprev * bel_x_prev; % 2x1 mat = 2x2 mat + 2x1 mat

% correction step
bel_x = p_s_x(p_s_x_stopped, p_s_x_notstopped, myspeed(t)) .* bel_bar_x;
bel_x = bel_x ./ sum(bel_x); % eta = 1/sum(bel_x)

% step foward
bel_x_prev = bel_x;
end

figure(3)
%time = [1:length(myspeed)] .*0.5;
time = (t_ego - 2) ./2;
plot(time, myspeed)
hold on
plot(time, estimate(1,:),'o-')
plot(time, estimate(2,:), 'o-')
title('Bayes Filter Output and Speed for Car Ego')
xlabel('Time')
ylabel('Certainty or Speed (m/s)');
legend('Vehicle Speed','Stopped Probability Estimate', 'Not Stopped Probability Estimate')


%% speed calculating function
function [Sdata] = speed(Xdata,Ydata, t_array, tprev_array, deltat)
%Calculates the speed data from vehical X Y data
%t array and t prev array are the lines in the Excel file we want to read
%from
    Sdata = (((Xdata(t_array)-Xdata(tprev_array)).^2 + (Ydata(t_array)-Ydata(tprev_array)).^2).^0.5) ./ deltat;

end

%% speed calculating function
function [measurementlikelihood] = p_s_x(p_s_x_stopped, p_s_x_notstopped, speed_t)
%Description
%informed by PDFs of Vehicle 4 and Moving Vehicles (Veh 2,3,5)
    p_stopped = pdf(p_s_x_stopped, speed_t);
    p_notstopped = pdf(p_s_x_notstopped,speed_t);
    
    % belief vector is formatted [stopped, notstopped]
    measurementlikelihood = [p_stopped; p_notstopped];
end


