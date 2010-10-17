% WITHOUTPRNTRDY
%
% Files
%   dmac        - DMAC. It only has transmission phase.
%   fat         - Returns the aggregation tree obtained with FAT
%   genData     - Gen traffic in nS points furthest frm sink.
%   plt         - Plot tree marking nodes & links
%   steiner     - Appr Steiner tree (Dijkstra + my intuition)
%   steiner_tst - Short driver function to test steiner.m
%   topology    - Gen rndm topol & get tier & tx & ix neighbrs
%   txPhase     - Tx phase without ack loss and without stop and go
%   txPhase_tst - Runs txPhase.m in a simple predefined topology
%   compare     - Old driver that I used to compare DMAC and OrneMAC
%   nPkt        - New pkt using inputs: frm,dst,drtn,ty,Nr,src
