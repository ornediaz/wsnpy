using System;
using System.Collections.Generic;
namespace perdida
{
    class LossTree
    {
        public Node[] nodes;
        public List<Node> postorder = new List<Node>(); // does not include sink
        public List<int> count = new List<int>();
        // How many measurements are provided for each sensing slot.
        private void add_postorder(Node x)
        {
            foreach (Node n in x.ch)
            {
                add_postorder(n);
            }
            postorder.Add(x);
        }
        public LossTree(int[] fv, double[] ps, int size)
        {
            nodes = new Node[fv.Length];
            for (int i = 0; i < fv.Length; i++)
            {
                nodes[i] = new Node(i);
                nodes[i].ps = ps[i];
                nodes[i].size = size;
            }
            for (int i = 1; i < fv.Length; i++)
            {
                Node ancestor = nodes[fv[i]];
                nodes[i].f = ancestor;
                ancestor.ch.Add(nodes[i]);
                while (ancestor != null && ancestor.ID != 0)
                {
                    nodes[i].ancestors.Add(ancestor);
                    ancestor = nodes[fv[ancestor.ID]];
                }
            }
            foreach (Node n in nodes[0].ch)
            {
                add_postorder(n);
            }
            if (Principal.VB)
            {
                Console.Write("Postorder list: ");
                foreach (Node n in postorder)
                {
                    Console.Write(n.ID + ", ");
                }
                Console.Write(";\n");
            }
        }
        private void discard(Node n)
        {
            if (n.ID == 0)
            {
                throw new Exception();
            }
            while (n.pkts.Count > n.size)
            {
                int position = 0;
                if (n.discard_type == 0)
                {
                    position = 0;
                    for (int i = 0; i < n.pkts.Count; i++)
                    {
                        if (n.pkts[i].t < n.pkts[position].t)
                        {
                            position = i;
                        }
                    }
                }
                else if (n.discard_type == 1)
                {
                    int kmin = 999999;
                    foreach (Packet p in n.pkts)
                    {
                        kmin = Math.Min(kmin, p.k);
                    }
                    List<int> l = new List<int>();
                    for (int i = 0; i < n.pkts.Count; i++)
                    {
                        if (n.pkts[i].k == kmin)
                        {
                            l.Add(i);
                        }
                    }
                    position = l[0];
                    foreach (int i in l)
                    {
                        if (n.pkts[i].t < n.pkts[position].t)
                        {
                            position = i;
                        }
                    }
                }
                else if (n.discard_type == 2)
                {
                    List<int> l = new List<int>();
                    for (int i = 0; i < n.pkts.Count; i++)
                    {
                        if (!n.gen.Contains(n.pkts[i].t % count.Count))
                        {
                            l.Add(i);
                        }
                    }
                    if (l.Count == 0)
                    {
                        for (int i = 0; i < n.pkts.Count; i++)
                        {
                            l.Add(i);
                        }
                    }
                    position = l[0];
                    foreach (int i in l)
                    {
                        if (n.pkts[i].t < n.pkts[position].t)
                        {
                            position = i;
                        }
                    }
                }
                else
                {
                    throw new Exception("Incorrect discard_type");
                }
                n.pkts.RemoveAt(position);
            }
        }
        public int[] simulate_it(int n_tx_frames, double rate, int type, int seed)
        {
            foreach (Node n in nodes)
            {
                n.pkts = new List<Packet>(n.size + 4);
                if (type == 0)
                {
                    n.discard_type = 0;
                    n.select_type = 0;
                }
                else if (type == 1)
                {
                    n.discard_type = 0;
                    n.select_type = 1;
                }
                else if (type == 2)
                {
                    n.discard_type = 1;
                    n.select_type = 1;
                }
                else if (type == 3)
                {
                    n.discard_type = 2;
                    n.select_type = 2;
                }
                else if (type == 4)
                {
                    double m = 1.0;
                    foreach (Node y in n.ancestors)
                    {
                        m = Math.Min(m, y.ps);
                    }
                    if (rate < m)
                    {
                        n.discard_type = 0;
                        n.select_type = 0;
                    }
                    else
                    {
                        n.discard_type = 1;
                        n.select_type = 1;
                    }
                }
                else
                {
                    throw new ArgumentException("Incorrect type");
                }
            }
            Principal.rgen = new Random(seed);
            int[] results = new int[(int)Math.Floor((n_tx_frames - 1) * rate + 1)];
            int sensing_interval = 0;
            for (int frame = 0; frame < n_tx_frames; frame++)
            {
                Principal.prnt("*** Simulating frame " + frame);
                for (; sensing_interval <= frame * rate; sensing_interval++)
                {
                    foreach (Node u in postorder)
                    {
                        u.pkts.Add(new Packet(sensing_interval, 1));
                        discard(u);
                    }
                }
                foreach (Node n in postorder)
                {
                    Principal.prnt("Processing node " + n.ID);
                    double random_number = Principal.rgen.NextDouble();
                    //Console.WriteLine(random_number);
                    if (n.pkts.Count == 0 || random_number >= n.ps)
                    {
                        continue;
                    }
                    int position = 0;
                    if (n.select_type == 0)
                    {
                        for (int i = 0; i < n.pkts.Count; i++)
                        {
                            if (n.pkts[i].t < n.pkts[position].t)
                            {
                                position = i;
                            }
                        }
                    }
                    else if (n.select_type == 1)
                    {
                        int maxcount = 0;
                        foreach (Packet p in n.pkts)
                        {
                            maxcount = Math.Max(maxcount, p.k);
                        }
                        List<int> l = new List<int>();
                        for (int i = 0; i < n.pkts.Count; i++)
                        {
                            if (n.pkts[i].k == maxcount)
                            {
                                l.Add(i);
                            }
                        }
                        position = l[0];
                        foreach (int i in l)
                        {
                            if (n.pkts[i].t < n.pkts[position].t)
                            {
                                position = i;
                            }
                        }
                    }
                    else if (n.select_type == 2)
                    {
                        List<int> l = new List<int>();
                        for (int i = 0; i < n.pkts.Count; i++)
                        {
                            if (n.gen.Contains(n.pkts[i].t % count.Count))
                            {
                                l.Add(i);
                            }
                        }
                        if (l.Count == 0)
                        {
                            for (int i = 0; i < n.pkts.Count; i++)
                            {
                                l.Add(i);
                            }
                        }
                        position = l[0];
                        foreach (int i in l)
                        {
                            if (n.pkts[i].t < n.pkts[position].t)
                            {
                                position = i;
                            }
                        }
                    }
                    Principal.prnt("Frame " + frame + "; Node " + n.ID + " position = " + position);
                    Packet pkt = n.pkts[position];
                    n.pkts.RemoveAt(position);
                    //Console.WriteLine("Node {0:d}; tx: ({1:d}, {2:d})", n.ID, pkt.t, pkt.k);
                    if (n.f.ID == 0)
                    {
                        results[pkt.t] += pkt.k;
                    }
                    else
                    {
                        bool added = false;
                        foreach (Packet j in n.f.pkts)
                        {
                            if (j.t == pkt.t)
                            {
                                j.k += pkt.k;
                                added = true;
                                break;
                            }
                        }
                        if (added == false)
                        {
                            n.f.pkts.Add(pkt);
                        }
                        discard(n.f);
                    }
                }
            }
            return results;
        }
        public void find_schedule(int frames, int source_min)
        {
            foreach (Node x in postorder)
            {
                x.q = (int)Math.Floor(x.ps * frames);
                foreach (Node y in x.ancestors)
                {
                    x.q = Math.Min(x.q, (int)Math.Floor(y.ps * frames));
                }
            }
            int max_frames = 99999;
            for (int frame = 0; frame < max_frames; frame++)
            {
                List<List<Node>> tree_list = new List<List<Node>>(nodes.Length);
                foreach (Node n in postorder)
                {
                    List<Node> tree = new List<Node>(nodes.Length);
                    tree.Add(n);
                    tree.AddRange(n.ancestors);
                    bool unsuitable = false;
                    foreach (Node x in tree)
                    {
                        if (x.q <= 0)
                        {
                            unsuitable = true;
                        }
                    }
                    if (unsuitable)
                    {
                        continue;
                    }
                    Stack<Node> stack = new Stack<Node>();
                    foreach (Node x in n.ch)
                    {
                        stack.Push(x);
                    }
                    while (stack.Count > 0 && tree.Count < source_min)
                    {
                        Node y = stack.Pop();
                        if (y.q > 0)
                        {
                            tree.Add(y);
                            foreach (Node z in y.ch)
                            {
                                stack.Push(z);
                            }
                        }
                    }
                    if (tree.Count >= source_min)
                    {
                        tree_list.Add(tree);
                    }
                }
                if (tree_list.Count == 0)
                {
                    break;
                }
                List<Node> best_tree = tree_list[0];
                foreach (List<Node> auxtree in tree_list)
                {
                    if (auxtree[0].ancestors.Count > best_tree[0].ancestors.Count)
                    {
                        best_tree = auxtree;
                    }
                }
                Principal.prnt("Selected subtree of node " + best_tree[0].ID);
                foreach (Node n in best_tree)
                {
                    add_frame(n, frame);
                }
                nodes[0].gen.Add(frame);
            }
            if (count.Count == max_frames - 1)
            {
                throw new Exception();
            }
            if (count.Count == 0)
            {
                throw new Exception("source_min is too small");
            }
            Stack<Node> unprocessed = new Stack<Node>();
            foreach (Node n in nodes[0].ch)
            {
                unprocessed.Push(n);
            }
            while (unprocessed.Count > 0)
            {
                Node x = unprocessed.Pop();
                foreach (Node y in x.ch)
                {
                    unprocessed.Push(y);
                }
                List<int> available = new List<int>();
                foreach (int q in x.f.gen)
                {
                    if (!x.gen.Contains(q))
                    {
                        available.Add(q);
                    }
                }
                while (x.q > 0 && available.Count > 0)
                {
                    int frm = available[0];
                    foreach (int i in available)
                    {
                        if (count[i] < count[frm])
                        {
                            frm = i;
                        }
                    }
                    add_frame(x, frm);
                    available.Remove(frm);
                }
            }
            foreach (Node x in postorder)
            {
                if (x.q > 0)
                {
                    throw new Exception();
                }
            }
            show_schedule();
        }
        public void add_frame(Node n, int frame)
        {
            if (n.q < 1)
            {
                throw new Exception();
            }
            n.q -= 1;
            n.gen.Add(frame);
            if (frame >= count.Count)
            {
                count.Add(0);
            }
            count[frame] += 1;
            Principal.prnt("Node " + n.ID + "gained frame " + frame);
        }
        public void show_schedule()
        {
            if (Principal.VB)
            {
                Console.WriteLine("********Showing the computed schedule*******");
                for (int i = 1; i < nodes.Length; i++)
                {
                    nodes[i].gen.Sort();
                    Console.Write("Node " + i + "; ");
                    foreach (int j in nodes[i].gen)
                    {
                        Console.Write(j + ", ");
                    }
                    Console.WriteLine();
                }
            }
        }
    }
}