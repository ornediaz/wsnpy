using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace perdida
{
    class Packet
    {
        public int t;
        public int k;
        public Packet(int t, int k)
        {
            this.t = t;
            this.k = k;
        }
    }
    class Node
    {
        public int ID;
        public int discard_type = 0;
        public int select_type = 0;
        public List<Packet> pkts;
        public List<Node> ancestors = new List<Node>(); // does not include sink
        public List<Node> ch = new List<Node>();
        public List<int> gen = new List<int>();
        public Node f = null;
        public double ps;
        public int q = 0;
        // maximum number of packets that a node can transmit per hyperframe. 
        public int size;
        public Node(int ID)
        {
            this.ID = ID;
        }
    }
}