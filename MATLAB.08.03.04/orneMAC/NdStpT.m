function h = NdStpT
% Setup time per tier used in OrneMAC
h = FxdBkffT + 2* ContentionT + RelReqT + B4AckT + RelOffT;
