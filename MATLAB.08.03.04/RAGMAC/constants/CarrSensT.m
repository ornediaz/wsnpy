function h = CarrSensT
h = 0.1*ContentionT;
% Time 2 detect an ongoing tx. Two nodes that decide to transmit during
% this time will generate a collision if the recipient is within both
% nodes' range.

