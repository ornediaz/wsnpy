%% Obtain lifetime graphs of the RT-Link paper
% We considers two cases: the B-MAC protocol and RT-Link
% TODO: Units are incorrect
clear,clc,close all

%% Define power consumptions
P_radio_Tx = 52.2; %mW
P_radio_Rx = 59.1; %mW
P_radio_idle = 1.28; %mW
P_radio_sleep = 3e-3; %mW
P_CPU_active = 3.3; %mW
P_CPU_sleep = 3e-3; %mW
P_sync = 15; %mW

%% Define time parameters

rho= 10e-5; % The clocks drift at a rate of 10µs per second
N_slots = 32;
T_idle = 0.5; % How can I compute this?
T_cca = 2.5; % ms
T_max_payload = 4;
T_sync = 100e-3;
T_slot= 5e-3;
T_frame= N_slots * T_slot;
T_sync_setup = 20e-3 + rho * T_frame;
T_GRx = 300e-3; % ms
T_GTx = 100e-3; % ms
T_ISS = 500e-3; % ms
T_active = T_sync_setup + T_sync + N_slots * ( T_max_payload * T_ISS );
 
%% Energy parameters -- involve time and power
E_sync = P_sync * (T_sync + T_sync_setup);
E_CPU_active = P_CPU_active * T_active;
E_CPU_sleep = P_CPU_sleep * T_idle;
E_radio_Tx = P_radio_Tx * (T_max_payload + T_GTx);
E_radio_Rx = P_radio_Rx * T_max_payload;
E_radio_idle = P_radio_idle * T_active;
E_radio_sleep = P_radio_sleep * T_idle;
E_GRX = P_radio_Rx * T_GRx;



%% Energy consumption in B-MAC

%P_sleep = P_radio_sleep+P_CPU_sleep;
P_sleep=3e-3;%in mW-->possibly wrong!

C_bat = 2500; % mAh
E_sample = 150; %mJ
E_Rx = E_radio_Rx;
P_Tx = P_radio_Tx;
E_CPU = E_CPU_active; %FIXME: this is for RT-Link, not for B-MAC
% TODO: Some of these parameters are wrong. One thing is for sure: the
% power consumption in sleep mode must be smaller than in transmit mode,
% but this does not hold in the paper.

C_bat = 2500;% mAh
V = 3; %Voltage in Volts

EnerConsumpPerSample = @(T_s,T_c)                       ...
	  T_s ./ T_c * E_sample ...
    +  P_radio_Rx * ( (T_s ./ T_c) * T_cca + T_max_payload )   ... % Energy receiving (mJ)
    + P_radio_Tx * ( T_c + T_max_payload )                    ... % Energy transmitting (mJ)
    + P_CPU_active * T_s/15  ... % FIXME: I have no clue about the active time
    + P_sleep * ( T_s - (T_cca * T_s./T_c) );       % Energy spent sleeping
    %In mJ = mA * Volt * seg
    % The power consumption of obtaining samples (ADC, etc) is missing
    % This model assumes that T_max_payload is negligible
AverageCurrentConsumpt = @(T_s,T_c) EnerConsumpPerSample(T_s,T_c) /T_s/V; % In mA

% *Lifetime*
Lhours=@(T_s,T_c) C_bat./ AverageCurrentConsumpt(T_s,T_c); % Lifetime in hours
Lyears = @(T_s,T_c) Lhours(T_s,T_c) / 24 / 365;   % Lifetime in years

%% Plot the results
T_c = linspace(0,15) ; % Check interval in seconds
T_s = [20 30 40 60]; % Sample time in ms
life = zeros( length(T_c) , length(T_s) );
for kk = 1: length(T_s)
    T_s_in_ms = T_s(kk) * 60 * 1e3;
    life(:,kk) = Lyears(T_s_in_ms , T_c * 1000);
    opt(kk) = sqrt(T_cca*T_s_in_ms);
    lopt(kk)=Lyears(T_s_in_ms,opt(kk));
    leg{kk,:} = [ 'Check interval ' num2str(T_s(kk)) ' min' ];
end  
plot(T_c,life,opt/1000,lopt,'*')
legend(leg)
xlabel('Check interval (s)')
ylabel('Lifetime (yrs)')