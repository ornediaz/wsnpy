%% Distributed Bellman-Ford simulation
% In a randomly generated topology, through message exchange with its
% neighbors, each node finds which of its neighbors is the next hop in the
% best path to the gateway and the aggregated cost through this path.
%
% Each node keeps the following information:
% 
% * About its neighbors: their ID and the cost
% * About the best known path to the sink: cost and next hop
%
% Contributions:
% * It is the first distributed algorithm I simulate
% 
% Limitations:
%
% * It does not model collisions. In each timeslot, all transmissions
% succeed.
% * It does not model energy consumption.
% * It does not 
%% Generate randomly distributed nodes, plot them and label them
clear,clc,close all
tic
H=40; %Number of nodes
rand('twister',3); % Initialize the random number generator

%%
% Generate |H| nodes with |x| and |y| coordinates randomly distributed in
% [-0.5 0.5]
P=[0 0;rand(H-1,2)-.5];

%%
% Plot the nodes' location and label them with their node number
plot(P(:,1),P(:,2),'o')
text(P(:,1)+0.01,P(:,2),num2str((1:H)'));
axis([-.5 .5 -.5 .5])
xlabel('X-coordinate of the nodes')
ylabel('Y-coordinate of the nodes')
hold on

%% Initialize data structures used by the nodes

%%
% |x{i}| contains node i's information about its neighbors in a structure
% array with fields:
%%
% * |ID| is the neighbor ID
% * |q| is the cost of the link to its neighbor (not accumulated)
x=cell(H,1);

%%
% Node |i| stores the next hop to the gateway in a structure |nh(i)| (next hop)
% with fields:
%% 
% * |ID| is the ID of the next hop
% * |Q| is the aggregated cost from the source of the packet to the sink
nh=repmat(struct('Q',Inf,'ID',-1),H,1); % Initialize cost to Inf and next hop ID to -1
tier = Inf(H,1); %Indicates the tier of each node
tier(1) = 0; % The gateway knows that it is the gateway

%% 
% Node |i| stores its incoming packets in a structure array |pkt{i,1}| with the following fields:
%% 
% * |ID| is the ID of the source of the packet
% * |Q| is the aggregated cost from the source of the packet to the sink
pkt = cell(H,1); 

%% Initialize each node's information about its neighbors
for i=2:H % For each node 'i' except the gateway
	for j=[1:i-1 i+1:H] % For each node 'j' including gateway
		D = norm(P(i,:)-P(j,:)); % Compute the distance to node 'j'
		if D < .2 % If distance < threshold, they are neighbors
			% Node 'i' stores the information of the just discovered
			% neighbor:
			x{i,1} = [ x{i,1} struct('ID',j,'q',D) ];
			% If the new neighbor is the gateway (node 1):
			if j == 1
				tier(i) = 1; % Node 'i' is in tier 1
				% The aggregated cost is D and the next hop is the gateway
				nh(i).Q = D; 
				nh(i).ID = 1;
			end
		end
	end
end
%% Create initial messages
% Every node in tier 1 sends a packet to each of its neighbors (except the
% first neighbor, which by construction is the gateway).  The packet is a
% structure containing the source's ID (|ID|) and the aggregated cost to
% the gateway (|Q|).
for i = find(tier == 1)' % Index of every node in tier 1
	for k = 2:length([x{i}]) % For all the neighbors of node i(which is in tier 1) 
		% We skip the first link because it leads to the sink
		pkt{x{i}(k).ID} = [ pkt{x{i}(k).ID} struct('ID',i,'Q',x{i}(1).q) ];
	end
end
%% Find shortest path to the gateway using Bellman-Ford algorithm
 for numIt = 1:10 % Repeat a number of times
	 for i = 2:H % For each node except the gateway, check if it needs to do something
		 if ~isempty(pkt{i}) % Process incoming packets if there are any
			 changes = false;
			 n = -1; % The source of the last change
			 % |p| takes the value of all packets received by node |i|
			 for p = pkt{i} 
				 % |n| indicates the position of the source of the packet
				    % in |i|'s neighbor list.
				 n = find(p.ID == [x{i,1}.ID]);
				 newCost = p.Q + x{i,1}(n).q; % Cost through the new node
				 if newCost < nh(i).Q % If the new path is better
					 nh(i).Q = newCost;
					 nh(i).ID = p.ID; % Update routing information
					 changes = 1;
				 end
			 end
			 if changes
				 % Send the new cost to all neighbors except the one that
				 % triggered the change
				 for k=[1:n-1 n+1:length(x{i,1})]
					 pkt{x{i}(k).ID} = [ pkt{x{i}(k).ID} ...
						 struct('ID',i,'Q',newCost) ];
				 end
			 end
		 end
	 end
 end

%% Plot the tree obtained with Bellman-Ford algorithm
% For each node whose cost to the gateway is not |Inf|, plot a straight
% line to the next hop
for i = find(~isinf([nh.Q]))
	plot( P([i nh(i).ID],1) , P( [i nh(i).ID ] , 2 ) )
end
title('Path to the gateway computed by the Bellman-Ford algorithm')

toc
