// python run.py ProdGlb.graphRate1 15 1 1
// ProdGlb is the class with the main functions
//
// The functions have to be run with:




using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

class LossTree
{
    public Node[] nodes;
    public List<Node> postorder = new List<Node>(); // does not include sink
    public List<int> count = new List<int>();
    // count[i] is the number of measurements provided for sensing time i
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
        for (int i = 0; i < fv.Length; i++)
        {
            int ancestor_ID = fv[i];
            if (ancestor_ID == -1)
            {
                nodes[i].f = null;
                continue;
            }
            nodes[i].f = nodes[ancestor_ID];
            nodes[ancestor_ID].ch.Add(nodes[i]);
            while (ancestor_ID != -1)
            {
                nodes[i].ancestors.Add(nodes[ancestor_ID]);
                ancestor_ID = fv[ancestor_ID];
            }
        }
        foreach (Node n in nodes[0].ch)
        {
            add_postorder(n);
        }
        if (G.VB)
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
                // Discard the oldest packet
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
                // Discard the packet with the minimum node count k
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
                // Discard the oldest high-priority packet

                // High priority packets are in the n.gen list
                //
                // The list l contains all the high priority ppackets.
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
                    // If there are no high-priority packets, treat all
                    // nodes as high priority.
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
        G.rgen = new Random(seed);
        int[] results = new int[(int)Math.Floor((n_tx_frames - 1) * rate + 1)];
        int sensing_interval = 0;
        for (int frame = 0; frame < n_tx_frames; frame++)
        {
            G.prnt("*** Simulating frame " + frame);
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
                G.prnt("Processing node " + n.ID);
                if (n.pkts.Count == 0)
                {
                    G.prnt("Node " + n.ID + "has empty buffer");
                    continue;
                }
                double random_number = G.rgen.NextDouble();
                //Console.WriteLine(random_number);
                if (random_number >= n.ps)
                {
                    G.prnt("Packet transmission of node " + n.ID +
                            "failed");
                    continue;
                }
                int position = 0;
                if (n.select_type == 0)
                {
                    // Select the oldest packet.
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
                    // Among all the packets with the biggest node count k,
                    // select the oldest packet.
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
                    // Among all the packets with a high priority, select
                    // the oldest.  If no packets have high priority, select
                    // the oldest packet.
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
                        // If there are no high-priority packets, consider
                        // them equally.
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
                if (G.VB)
                {
                    Console.WriteLine("Frame {0}; Node {1}; position {3}",
                            frame, n.ID, position);
                }
                Packet pkt = n.pkts[position];
                n.pkts.RemoveAt(position);
                //Console.WriteLine("Node {0:d}; tx: ({1:d}, {2:d})", n.ID,
                //pkt.t, pkt.k);
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
        this.nodes[0].ps = 1.0;
        foreach (Node x in nodes)
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
            foreach (Node n in nodes)
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
                // In the following lines the <= in "tree.Count<=source_min"
                // is correct because tree includes ancestors, which in turn
                // includes the data sink.  The sink should not be counted
                // in source_min.
                while (stack.Count > 0 && tree.Count <= source_min)
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
                if (tree.Count > source_min)
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
            G.prnt("Selected subtree of node " + best_tree[0].ID);
            foreach (Node n in best_tree)
            {
                if (n.ID != 0)
                {
                    add_frame(n, frame);
                }
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
        G.prnt("Node " + n.ID + "gained frame " + frame);
    }
    public void show_schedule()
    {
        if (G.VB)
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
    public static void tst_find_schedule()
    {
        int[] fv = new int[] {-1, 0, 1, 1, 1, 0, 5, 6, 7, 8};
        double[] ps = new double[fv.Length];
        for (int i = 0; i < ps.Length; i++)
        {
            if (i < 5)
            {
                ps[i] = 0.4;
            }
            else
            {
                ps[i] = 0.8;
            }
        }
        LossTree t = new LossTree(fv, ps, 8);
        G.VB = true;
        t.find_schedule(5, 2);
        foreach (Node n in t.nodes)
        {
            Console.WriteLine("Node {0}: q = {1}", n.ID, n.q);
        }
    }
    public static void tst_find_schedule2()
    {
        int[] fv = new int[] {-1, 0, 1, 1, 0};
        double[] ps = new double[] {1, 0.8, 0.6, 0.2, 0.2};
        LossTree t = new LossTree(fv, ps, 8);
        G.VB = true;
        t.find_schedule(5, 3);
        foreach (Node n in t.nodes)
        {
            Console.WriteLine("Node {0}: q = {1}", n.ID, n.q);
        }
    }
}
class Node
{
    public int ID;
    public int discard_type = 0;
    public int select_type = 0;
    public List<Packet> pkts;
    public List<Node> ancestors = new List<Node>(); // includes sink
    public List<Node> ch = new List<Node>();
    public List<int> gen = new List<int>();
    public Node f = null;
    public double ps;
    public int q = 0;
    // maximum number of packets that a node can transmit per hyperframe. 
    public int size;
    public Node(int ID) {
        this.ID = ID;
    }
}
class Packet
{
    public int t;
    public int k;
    public Packet(int t, int k) {
        this.t = t;
        this.k = k; }
}
class PgfAxis
{
    public List<string> buf = new List<string>();
    public List<string> options = new List<string>();
    public List<string> legend = new List<string>();
    public PgfAxis(string xlabel, string ylabel)
    {
        options.Add("xlabel = { " + xlabel + "}");
        options.Add("ylabel = { " + ylabel + "}");
    }
    public void plot(double[] xv, double[] yv, string leg)
    {
        if (xv.Length != yv.Length)
        {
            throw new ArgumentException();
        }
        buf.Add("      \\addplot coordinates {\n");
        for (int i = 0; i < xv.Length; i++)
        {
            if (i < xv.Length - 1)
            {
                buf.Add("        (" + xv[i] + ", " + yv[i] + ")\n");
            }
            else
            {
                buf.Add("        (" + xv[i] + ", " + yv[i] + ")};\n");
            }
        }
        legend.Add(leg);
    }
    public void mplot(double[] xv, double[,] ym, string[] legv)
    {
        for (int j = 0; j < ym.GetLength(1); j++)
        {
            double[] yv = new double[ym.GetLength(0)];
            for (int i = 0; i < ym.GetLength(0); i++)
            {
                yv[i] = ym[i, j];
            }
            plot(xv, yv, legv[j]);
        }
    }
    // Inverse multiplot.  It is inverse in the sense that I swap the x and
    // y axis.  It is multi in the sense that
    public void implot(double[] xv, double[,] ym, string[] legv)
    {
        for (int j = 0; j < ym.GetLength(1); j++)
        {
            double[] yv = new double[ym.GetLength(0)];
            for (int i = 0; i < ym.GetLength(0); i++)
            {
                yv[i] = ym[i, j];
            }
            plot(yv, xv, legv[j]);
        }
    }
}
class PltGlb
{
    public static void plot_logical3(int[] fv, double[] ps, int plot)
    {
        if (plot == 0)
        {
            return;
        }
        string fname = "ztree";
        using (StreamWriter sw = File.CreateText(fname + ".dot"))
        {
            sw.Write("digraph tree {\n");
            for (int i = 1; i < fv.Length; i++)
            {
                sw.Write("{0:d} -> {1:d} [label = {2:f2}];\n", i, fv[i], ps[i]);
            }
            sw.Write("}\n");
            sw.Close();
        }

        ProcessStartInfo pi = new ProcessStartInfo("dot", String.Format(" -Tpdf {0}.dot -o {0}.pdf", fname));
        if (plot == 1)
        {
            pi.CreateNoWindow = true;
            pi.WindowStyle = ProcessWindowStyle.Hidden;
        }
        Process p1 = Process.Start(pi);
        p1.WaitForExit();
        if (plot == 2)
        {
            Process p2 = Process.Start("dot", String.Format(" -Tpng {0}.dot -o {0}.png", fname));
            p2.WaitForExit();
            Process p3 = Process.Start(fname + ".png");
        }
    }
}
//  Main plotting interface
class Pgf
{
    public List<PgfAxis> body = new List<PgfAxis>();
    public List<string> extra_body = new List<string>();
    public List<string> extra_preamble = new List<string>();
    public Pgf()
    {
        extra_preamble.Add("\\usepackage{plotjour1}");
    }
    public void add(string xlabel, string ylabel)
    {
        body.Add(new PgfAxis(xlabel, ylabel));
    }
    public void plot(double[] xv, double[] yv, string leg)
    {
        body[body.Count - 1].plot(xv, yv, leg);
    }
    public void mplot(double[] xv, double[,] ym, string[] legv)
    {
        body[body.Count - 1].mplot(xv, ym, legv);
    }
    public void implot(double[] xv, double[,] ym, string[] legv)
    {
        body[body.Count - 1].implot(xv, ym, legv);
    }
    public void save(string filename, int plot)
    {
        List<string> lst = new List<string>();
        lst.Add("\\documentclass{article}\n");
        lst.Add("\\usepackage[margin=0in]{geometry}\n");
        lst.Add("\\usepackage{orne1}\n");
        foreach (string s in extra_preamble)
        {
            lst.Add(s + "\n");
        }
        lst.Add("\\begin{document}\n");
        for (int i = 0; i < body.Count; i++)
        {
            lst.Add("  \\begin{tikzpicture}\n");
            lst.Add("    \\begin{axis} [\n");
            for (int j = 0; j < body[i].options.Count; j++)
            {
                if (j < body[i].options.Count - 1)
                {
                    lst.Add("      " + body[i].options[j] + ", \n");
                }
                else
                {
                    lst.Add("      " + body[i].options[j] + "\n");
                }
            }
            lst.Add("      ]\n");
            foreach (string s in body[i].buf)
            {
                lst.Add(s);
            }
            lst.Add("      " + "\\legend{{");
            for (int j = 0; j < body[i].legend.Count; j++)
            {
                if (j < body[i].legend.Count - 1)
                {
                    lst.Add(body[i].legend[j] + "}, {");
                }
                else
                {
                    lst.Add(body[i].legend[j] + "}}%\n");
                }
            }
            lst.Add("    \\end{axis}\n");
            lst.Add("  \\end{tikzpicture}\n");
            if ((i % 2) == 1)
            {
                lst.Add("\n");
            }
        }
        foreach (string s in extra_body)
        {
            lst.Add(s);
        }
        lst.Add("\\end{document}");
        using (StreamWriter sw = File.CreateText(filename + ".tex"))
        {
            foreach (string s in lst)
            {
                sw.Write(s);
            }
            sw.Close();
        }
        Console.WriteLine("File {0}.tex written", filename);
        if (plot > 0)
        {
            Console.WriteLine("We are here");
            ProcessStartInfo i = new ProcessStartInfo("pdflatex", filename +
                    ".tex");
            Process p1 = Process.Start(i);
            if (plot == 2)
            { 
                p1.WaitForExit();
                if (p1.ExitCode == 0)
                {
                    System.Diagnostics.Process.Start("acrord32", filename + ".pdf");
                }
            }
        }
    }
}
class ProdGlb
{
    // Simulate a deterministic topology for different rates.  
    //
    // The topology depends on parameter tst_nr
    //
    // The goal is to see how the different methods work with different
    // topologies. 
    public static void graphRate1(int tst_nr, int n_averages, int plot)
    {
        Console.WriteLine("Executing {0}({1:d2},{2:d6},{3}). Total {4}",
                G.current(), tst_nr, n_averages, plot, G.elapsed());
        int frames = 10;
        int[] fv;
        int n_tx_frames = 2000;
        double opt = 0;
        double[] ps;
        double[] rate_v = G.linspace(0.1, 1.5, 28);
        int source_min = 3;
        int size = 30;
        int[] types = new int[] {0, 1, 2, 3, 4};
        G.VB = false;
        // Shows the advantage of the scheduled approach, particularly in
        // terms of 
        if (tst_nr == 0)
        {
            fv = new int[] { -1, 0, 1, 1, 1, 1 };
            ps = new double[] { 1, 0.8, 0.4, 0.4, 0.4, 0.4 };
        }
        // No advantage in the scheduled approach
        else if (tst_nr == 1)
        {
            fv = new int[] { -1, 0, 1, 1, 1, 1 };
            ps = new double[] { 1, 0.4, 0.4, 0.4, 0.4, 0.4 };
        }
        // No advantage in scheduled approach
        else if (tst_nr == 2)
        {
            fv = new int[] { -1, 0, 1, 1, 1, 1 };
            ps = new double[] { 1, 0.4, 0.8, 0.8, 0.8, 0.8 };
        }
        // Slight advantage of scheduled approach, all select/discard
        // policies are similar.
        else if (tst_nr == 3)
        {
            fv = new int[] { -1, 0, 1, 1, 1, 1 };
            ps = new double[] { 1, 0.6, 0.6, 0.6, 0.2, 0.2 };
        }
        // Advantage of scheduled approach for low rate, great variety
        // between select/discard policies, but 0 is overall the best.
        else if (tst_nr == 4)
        {
            fv = new int[] { -1, 0, 1, 1, 1, 1 };
            ps = new double[] { 1, 0.3, 0.6, 0.6, 0.2, 0.2 };
        }
        // No schedule advantage and moderate advantage of packet
        // selection.  Strong peak at the optimal.
        else if (tst_nr == 5)
        {
            fv = new int[] { -1, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5 };
            ps = new double[12];
            for (int i = 0; i < fv.Length; i++)
            {
                ps[i] = 0.3;
            }
        }
        // No schedule advantage and moderate advantage of packet selection.
        else if (tst_nr == 6)
        {
            fv = new int[] { -1, 0, 1, 2, 3, 4, 5 };
            ps = new double[fv.Length];
            for (int i = 0; i < ps.Length; i++)
            {
                ps[i] = 0.3;
            }
        }
        // Unsuccessful attempt to make the hybrid select/discard topology
        // work.
        else if (tst_nr == 7)
        {
            fv = new int[] {-1, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
            ps = new double[fv.Length];
            for (int i = 0; i < ps.Length; i++)
            {
                ps[i] = 0.8;
            }
        }
        else if (tst_nr == 8)
        {
            fv = new int[] {-1, 0, 0, 1, 1, 1, 2, 2, 2};
            ps = new double[fv.Length];
            for (int i = 0; i < ps.Length; i++)
            {
                ps[i] = 0.8;
            }
        }
        else if (tst_nr == 9)
        {
            fv = new int[] {-1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 9, 10, 11, 12};
            ps = new double[fv.Length];
            for (int i = 0; i < ps.Length; i++)
            {
                if (i < 9)
                {
                    ps[i] = 0.8;
                }
                else
                {
                    ps[i] = 0.4;
                }
            }
        }
        // These parameters are chosen to show the advantage of the
        // scheduled approach vs the unscheduled approach.
        else if (tst_nr == 10)
        {
            fv = new int[] {-1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 9, 10, 11, 12};
            ps = new double[fv.Length];
            for (int i = 0; i < ps.Length; i++)
            {
                if (i < 9)
                {
                    ps[i] = 0.4;
                }
                else
                {
                    ps[i] = 0.8;
                }
            }
        }
        else if (tst_nr == 11)
        {
            fv = new int[] {-1, 0, 1, 1, 1, 0, 5, 6, 7, 8};
            ps = new double[fv.Length];
            for (int i = 0; i < ps.Length; i++)
            {
                if (i < 5)
                {
                    ps[i] = 0.4;
                }
                else
                {
                    ps[i] = 0.8;
                }
            }
        }
        // These parameters were chosen to show that the optimal rate is
        // hard to determine.  In this case, the total is higher for 0.83,
        // despite the fact that it means that node 1 has spare capacity
        // that it could be using to transmit more packets.
        else if (tst_nr == 12)
        {
            fv = new int[] {-1, 0, 1, 1, 1, 1, 1};
            ps = new double[] {1, 0.88, 0.83, 0.83, 0.83, 0.83, 0.83};
        }
        // These parameters were chosen to show the usefulness of the
        // scheduling algorithm in a small network.
        else if (tst_nr == 13)
        {
            fv = new int[] {-1, 0, 1, 1, 0};
            ps = new double[] {1, 0.8, 0.6, 0.2, 0.2};
        }
        // These parameters show that under a linear topology with all links
        // equally good, all select/discard policies perform very similarly.
        else if (tst_nr == 14)
        {
            fv = new int[] {-1, 0, 1, 2, 3, 4, 5};
            ps = new double[fv.Length];
            for (int i = 0; i < fv.Length; i++)
            {
                ps[i] = 0.5;
            }
        }
        // These parameters show a linear topology whose last node has many
        // children.  This is a good example of a situation in which the
        // select/discard type 2 performs much better than the other types.
        else if (tst_nr == 15)
        {
            fv = new int[] { -1, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5 };
            ps = new double[12];
            for (int i = 0; i < fv.Length; i++)
            {
                if (i < 6)
                {
                    ps[i] = 0.3;
                }
                else
                {
                    ps[i] = 1;
                }
            }
        }
        // These parameters show that under a linear topology with unequal
        // link qualities, the priority traffic may perform best.
        else if (tst_nr == 16)
        {
            fv = new int[] {-1, 0, 1, 2, 3, 4, 5, 6, 7};
            ps = new double[fv.Length];
            for (int i = 0; i < fv.Length; i++)
            {
                if (i < 4)
                    ps[i] = 0.4;
                else
                    ps[i] = 1;
            }
        }
        else if (tst_nr == 17)
        {
            fv = new int[] {-1, 0, 1, 1};
            ps = new double[] {1, 0.8, 0.4, 0.4};
            source_min = 2;
        }
        else if (tst_nr == 18)
        {
            fv = new int[] {-1, 0, 1, 1, 0};
            ps = new double[] {1, 0.8, 0.6, 0.2, 0.2};
            source_min = 2;
        }
        else if (tst_nr == 19)
        {
            fv = new int[] {-1, 0, 1, 1, 0};
            ps = new double[] {1, 0.66, 0.49, 0.16, 0.16};
            source_min = 2;
            frames = 5;
        }
        else
        {
            throw new ArgumentException("Inappropriate tst_nr");
        }
        PltGlb.plot_logical3(fv, ps, plot);
        double[,] tota = new double[rate_v.Length, types.Length];
        double[,] mean = new double[rate_v.Length, types.Length];
        double[,] pmin = new double[rate_v.Length, types.Length];
        for (int k = 0; k < n_averages; k++)
        {
            Console.WriteLine("Repetition {0} of {1}({2:d2},{3:d6},{4}) {5}",
                    k, G.current(), tst_nr, n_averages, plot, G.elapsed());
            for (int j = 0; j < rate_v.Length; j++)
            {
                for (int i = 0; i < types.Length; i++)
                {
                    LossTree t = new LossTree(fv, ps, size);
                    if (types[i] == 3)
                    {
                        t.find_schedule(frames, source_min);
                        if (k == 0 && j == 0)
                        {
                            opt = ((double)t.count.Count / frames);
                        }
                    }
                    int[] results = t.simulate_it(n_tx_frames, rate_v[j],
                            types[i], k);
                    foreach (int h in results)
                    {
                        tota[j, i] += (double)h / n_averages / n_tx_frames;
                        mean[j, i] += (double)h / n_averages / results.Length;
                        if (h < source_min)
                        {
                            pmin[j, i] += (double)1 / n_averages /
                                results.Length;
                        }
                    }
                }
            }
        }
        Console.WriteLine("**** Printing results *****");
        string[] legv = new string[] { "0", "1", "2",
            String.Format("3={0:F3}", opt), "4" };
        Pgf g = new Pgf();
        g.add("rate", "total");
        g.mplot(rate_v, tota, legv);
        g.add("rate", "mean");
        g.mplot(rate_v, mean, legv);
        g.add("rate", "pmin");
        g.mplot(rate_v, pmin, legv);
        g.add("missing data tolerance", "rate efficiency");
        g.implot(rate_v, pmin, legv);
        g.extra_body.Add("\n\\includegraphics[scale=0.4]{ztree.pdf}\n");
        string filename = String.Format("{0}_{1:d2}_{2:d6}", G.current(),
                tst_nr, n_averages);
        g.save(filename, plot);
    }
    // This function computes the average metrics at different rates for
    // different topologies.  This is quite useless, because different
    // metrics have different operation points.  It unfairly shows poor
    // performance of the scheduled approach.  The only useful part may be
    // comparing the unscheduled approaches.
    public static void graphRateRandom(int tst_nr, int n_averages, int plot)
    {
        Console.WriteLine("Executing {0}({1:d2},{2:d6},{3})", G.current(),
                tst_nr, n_averages, plot);
        double tx_rg = 2;
        double x = 5 * tx_rg;
        double y = 5 * tx_rg;
        double rho = 9;
        int n = (int)(rho * x * y / Math.PI / tx_rg / tx_rg);
        int frames = 10;
        int n_tx_frames = 2000;
        double opt = 0;
        double[] rate_v = G.linspace(0.1, 1.5, 28);
        int source_min = 3;
        int size = 30;
        int[] types = new int[] {0, 1, 2, 3, 4};
        G.VB = false;
        double[,] tota = new double[rate_v.Length, types.Length];
        double[,] mean = new double[rate_v.Length, types.Length];
        double[,] pmin = new double[rate_v.Length, types.Length];
        for (int k = 0; k < n_averages; k++)
        {
            Console.WriteLine("Repetition {0,4:D}. Total {1}",
                        k, G.elapsed());
            G.rgen = new Random(k);
            int[] fv = RandomTree.parents(n, x, y, tx_rg);
            double[] ps = new double[n];
            for (int u = 0; u < n; u++)
            {
                ps[u] = 0.5 + G.rgen.NextDouble() / 2;
            }
            for (int j = 0; j < rate_v.Length; j++)
            {
                for (int i = 0; i < types.Length; i++)
                {
                    LossTree t = new LossTree(fv, ps, size);
                    if (types[i] == 3)
                    {
                        t.find_schedule(frames, source_min);
                        if (k == 0 && j == 0)
                        {
                            opt = ((double)t.count.Count / frames);
                        }
                    }
                    int[] results = t.simulate_it(n_tx_frames, rate_v[j],
                            types[i], k);
                    foreach (int h in results)
                    {
                        tota[j, i] += (double)h / n_averages / n_tx_frames;
                        mean[j, i] += (double)h / n_averages / results.Length;
                        if (h < source_min)
                        {
                            pmin[j, i] += (double)1 / n_averages /
                                results.Length;
                        }
                    }
                }
            }
        }
        Console.WriteLine("**** Printing results *****");
        string[] legv = new string[] { "0", "1", "2",
            String.Format("3={0:F3}", opt), "4" };
        Pgf g = new Pgf();
        g.add("rate", "total");
        g.mplot(rate_v, tota, legv);
        g.add("rate", "mean");
        g.mplot(rate_v, mean, legv);
        g.add("rate", "pmin");
        g.mplot(rate_v, pmin, legv);
        string filename = String.Format("{0}_{1:d2}_{2:d6}", G.current(),
                tst_nr, n_averages);
        g.save(filename, plot);
    }
    // Compare sheduled vs unscheduled approaches operating at their optimal
    // point as a function of the node density and for different topologies.
    public static void graphRateSize(int tst_nr, int n_averages, int plot)
    {
        Console.WriteLine("Executing {0}({1:d2},{2:d6},{3})", G.current(),
                tst_nr, n_averages, plot);
        double tx_rg = 2;
        double[] xv = null;
        double[] yv = null;
        if (tst_nr == 0)
        {
            xv = new double[] {tx_rg, 2 * tx_rg};
            yv = new double[] {tx_rg,     tx_rg};
        }
        else if (tst_nr == 1)
        {
            xv = G.linspace(tx_rg, 5 * tx_rg, 5);
            yv = G.linspace(2 * tx_rg, 2.01 * tx_rg, 5);
        }
        if (xv.GetLength(0) != yv.GetLength(0))
        {
            throw new Exception("xv and yv should have equal length");
        }
        double rho = 15;
        int frames = 10;
        int n_tx_frames = 2000;
        double opt = 0;
        int[] types = new int[] {0, 1, 2, 3, 4};
        double[] rate_v = G.linspace(0.5, 3, 28);
        int source_min = 3;
        int size = 30;
        double[,] tota = new double[xv.GetLength(0), types.Length];
        double[,] mean = new double[xv.GetLength(0), types.Length];
        double[,] pmin = new double[xv.GetLength(0), types.Length];
        double[,] ropt = new double[xv.GetLength(0), types.Length];
        for (int k = 0; k < n_averages; k++)
        {
            Console.WriteLine("Repetition {0,4:D}. Total {1}",
                        k, G.elapsed());
            for (int s = 0; s < xv.GetLength(0); s++)
            {
                Console.WriteLine("s={0,2}, x/t={1,4:F}.  Total {2}", s,
				  xv[s] / tx_rg, G.elapsed());
                int n = (int)(rho* xv[s] * yv[s] / Math.PI / tx_rg / tx_rg);
                G.rgen = new Random(k);
                int[] fv = RandomTree.parents(n, xv[s], yv[s], tx_rg);
                double[] ps = new double[n];
                for (int u = 0; u < n; u++)
                {
                    ps[u] = 0.5 + G.rgen.NextDouble() / 2;
                }
                for (int i = 0; i < types.Length; i++)
                {
                    double m_tota = 0;
                    double m_mean = 0;
                    double m_pmin = 0;
                    double m_ropt = rate_v[0];
                    for (int j = 0; j < rate_v.Length; j++)
                    {
                        LossTree t = new LossTree(fv, ps, size);
                        if (types[i] == 3)
                        {
                            t.find_schedule(frames, source_min);
                        }
                        int[] results = t.simulate_it(n_tx_frames, 
                                            rate_v[j], types[i], k);
                        double a_tota = 0;
                        double a_mean = 0;
                        double a_pmin = 0;
                        foreach (int h in results)
                        {
                            a_tota += (double) h / n_tx_frames;
                            a_mean += (double) h / results.Length;
                            if (h < source_min)
                            {
                                a_pmin += (double)1 / results.Length;
                            }
                        }
                        if (a_tota > m_tota)
                        {
                            m_tota = a_tota;
                            m_mean = a_mean;
                            m_pmin = a_pmin;
                            m_ropt = rate_v[j];
                        }
                    }
                    tota[s, i] += m_tota / n_averages;
                    mean[s, i] += m_mean / n_averages;
                    pmin[s, i] += m_pmin / n_averages;
                    ropt[s, i] += m_ropt / n_averages;
                }
            }
        }
        Console.WriteLine("**** Printing results *****");
        string[] legv = new string[] { "0", "1", "2",
            String.Format("3={0:F3}", opt), "4" };
        Pgf g = new Pgf();
        string xlab = "normalized x size";
        double[] xn = new double[xv.Length];
        for (int q = 0; q < xn.Length; q++)
        {
            xn[q] = xv[q] / tx_rg;
        }
        g.add(xlab, "total");
        g.mplot(xn, tota, legv);
        g.add(xlab, "mean");
        g.mplot(xn, mean, legv);
        g.add(xlab, "pmin");
        g.mplot(xn, pmin, legv);
        g.add(xlab, "ropt");
        g.mplot(xn, ropt, legv);
        string filename = String.Format("{0}_{1:d2}_{2:d6}", G.current(),
                tst_nr, n_averages);
        g.save(filename, plot);
    }
    public static void graphRateSize2(int tst_nr, int n_averages, int plot)
    {
        Console.WriteLine("Executing {0}({1:d2},{2:d6},{3})", G.current(),
                tst_nr, n_averages, plot);
        double tx_rg = 2;
        double[] xv = null;
        double[] yv = null;
        if (tst_nr == 0)
        {
            xv = new double[] {tx_rg, 2 * tx_rg};
            yv = new double[] {tx_rg,     tx_rg};
        }
        else if (tst_nr == 1)
        {
            xv = G.linspace(tx_rg, 5 * tx_rg, 5);
            yv = G.linspace(2 * tx_rg, 2.01 * tx_rg, 5);
        }
        if (xv.GetLength(0) != yv.GetLength(0))
        {
            throw new Exception("xv and yv should have equal length");
        }
        double rho = 15;
        int frames = 10;
        int n_tx_frames = 2000;
        double opt = 0;
        int[] types = new int[] {0, 1, 2, 3, 4};
        double[] rate_v = G.linspace(0.5, 3, 28);
        int size = 30;
        double[,] tota = new double[xv.GetLength(0), types.Length];
        double[,] mean = new double[xv.GetLength(0), types.Length];
        double[,] pmin = new double[xv.GetLength(0), types.Length];
        double[,] ropt = new double[xv.GetLength(0), types.Length];
        for (double frac_source_min = 0.2; frac_source_min < 1; 
             frac_source_min += 0.3)
        {
                for (int k = 0; k < n_averages; k++)
                {
                    Console.WriteLine("Repetition {0,4:D}. Total {1}",
                                k, G.elapsed());
                    for (int s = 0; s < xv.GetLength(0); s++)
                    {
                        Console.WriteLine("s={0,2}, x/t={1,4:F}.  Total {2}", s,
                          xv[s] / tx_rg, G.elapsed());
                        int n = (int)(rho* xv[s] * yv[s] / Math.PI / tx_rg / tx_rg);
                        int source_min = (int) (frac_source_min * n);
                        G.rgen = new Random(k);
                        int[] fv = RandomTree.parents(n, xv[s], yv[s], tx_rg);
                        double[] ps = new double[n];
                        for (int u = 0; u < n; u++)
                        {
                            ps[u] = 0.5 + G.rgen.NextDouble() / 2;
                        }
                        for (int i = 0; i < types.Length; i++)
                        {
                            double m_tota = 0;
                            double m_mean = 0;
                            double m_pmin = 0;
                            double m_ropt = rate_v[0];
                            for (int j = 0; j < rate_v.Length; j++)
                            {
                                LossTree t = new LossTree(fv, ps, size);
                                if (types[i] == 3)
                                {
                                    t.find_schedule(frames, source_min);
                                }
                                int[] results = t.simulate_it(n_tx_frames, 
                                                    rate_v[j], types[i], k);
                                double a_tota = 0;
                                double a_mean = 0;
                                double a_pmin = 0;
                                foreach (int h in results)
                                {
                                    a_tota += (double) h / n_tx_frames;
                                    a_mean += (double) h / results.Length;
                                    if (h < source_min)
                                    {
                                        a_pmin += (double)1 / results.Length;
                                    }
                                }
                                if (a_tota > m_tota)
                                {
                                    m_tota = a_tota;
                                    m_mean = a_mean;
                                    m_pmin = a_pmin;
                                    m_ropt = rate_v[j];
                                }
                            }
                            tota[s, i] += m_tota / n_averages;
                            mean[s, i] += m_mean / n_averages;
                            pmin[s, i] += m_pmin / n_averages;
                            ropt[s, i] += m_ropt / n_averages;
                        }
                    }
                }
                Console.WriteLine("**** Printing results *****");
                string[] legv = new string[] { "0", "1", "2",
                    String.Format("3={0:F3}", opt), "4" };
                Pgf g = new Pgf();
                string xlab = "normalized x size";
                double[] xn = new double[xv.Length];
                for (int q = 0; q < xn.Length; q++)
                {
                    xn[q] = xv[q] / tx_rg;
                }
                g.add(xlab, "total");
                g.mplot(xn, tota, legv);
                g.add(xlab, "mean");
                g.mplot(xn, mean, legv);
                g.add(xlab, "pmin");
                g.mplot(xn, pmin, legv);
                g.add(xlab, "ropt");
                g.mplot(xn, ropt, legv);
                string filename = String.Format("{0}_{1:d2}_{2:d6}_{3:f2}",
                                                G.current(), tst_nr, 
                                                n_averages, frac_source_min);
                Console.WriteLine(filename);
                g.save(filename, plot);
        }
    }
    public static void multiplot1(int n_averages)
    {
        for (int tst_nr = 0; tst_nr < 15; tst_nr++)
        {
            Console.WriteLine("Iteration {0} of multiplot", tst_nr);
            ProdGlb.graphRate1(tst_nr, n_averages, 1);
        }
    }
}
class RandomTree
{
    public static int[] parents(int n, double x, double y, double tx_rg)
    {
        // n is  the number of nodes
        // x, y are the geographical area
        // tx_rg
        if (n < 2)
        {
            throw new Exception("Number of nodes must exceed 2");
        }
        int[] fv = new int[n];
        double[,] xy = new double[n, 2];
        double[,] cost = new double[n, n];
        int MAX_TESTS = 15;
        double MAX_DISCONNECTED = 0.1;
        for (int h = 0; h < MAX_TESTS; h++)
        { 
            for (int i = 1; i < n; i++)
            {
                xy[i, 0] = G.rgen.NextDouble() * x;
                xy[i, 1] = G.rgen.NextDouble() * y;
            }
            //Console.WriteLine(n);
            xy[0, 0] = 0;
            xy[0, 1] = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double dist = Math.Sqrt(Math.Pow(xy[i, 0] - xy[j, 0], 2)
                           + Math.Pow(xy[i, 1] - xy[j, 1], 2));
                    if (dist < tx_rg)
                    {
                        cost[i,j] = dist;
                        cost[j,i] = dist;
                    }
                    else
                    {
                        cost[i, j] = G.INF;
                        cost[j, i] = G.INF;
                    }
                }
            }
            if (G.VB)
            {
                Console.WriteLine("****Iteration {0}", h);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        Console.Write("{0:f3}, ", cost[i,j]);
                    }
                    Console.WriteLine();
                }
            }
            fv = dijkstra(cost);
            if (G.VB)
            {
                for (int i = 0; i < n; i ++)
                {
                    Console.Write("{0:d}, ", fv[i]);

                }
            }
            int disconnected = 0;
            foreach (int f in fv)
            {
                if (f == -1)
                {
                    disconnected++;
                }
            }
            if ((double) (disconnected - 1) / (n - 1) <= MAX_DISCONNECTED)
            {
                return fv;
            }
        }
        Console.WriteLine(n);
        Console.WriteLine(x);
        Console.WriteLine(y);
        Console.WriteLine(tx_rg);
        throw new Exception("No sufficiently connected topology found");
    }
    public static int[] dijkstra(double[,] cost)
    {
        // Return every node's next hop to the sink using Dijkstra's algorithm.
        // Parameter:
        // cost -- NxN ndarray indicating cost of N nodes '''
        int N = cost.GetLength(0);
        //len(cost);
        // Each node's smallest distance to the sink
        double[] dst = new double[N];
        for (int i = 1; i < N; i++)
        {
            dst[i] = G.INF;
        }
        dst[0] = 0; // The source is at distance 0 from itz
        int[] previous = new int[N];
        List<int> unprocessed = new List<int>();
        for (int i = 0; i < N; i++)
        {
            previous[i] = -1;
            unprocessed.Add(i);
        }
        //If any node does not have neighbors, it will never be processed.
        while (unprocessed.Count > 0)
        {
            int x = unprocessed[0];
            foreach (int i in unprocessed)
            {
                if (dst[i] < dst[x])
                {
                    x = i;
                }
            }
            if (dst[x] == G.INF)
            {
                break;
            }
            unprocessed.Remove(x);
            foreach (int y in unprocessed)
            {
                double alt = dst[x] + cost[x, y];
                if (alt < dst[y])
                {
                    dst[y] = alt;
                    previous[y] = x;
                }
            }
        }
        return previous;
    }
    public static void tst_parents()
    {
        double x = 10;
        double y = 4;
        double tx_rg = 2;
        double rho = 8;
        int n = (int)(rho * x * y / Math.PI / tx_rg / tx_rg);
        int[] fv = parents(n, x, y, tx_rg);
        double[] ps = new double[n];
        PltGlb.plot_logical3(fv, ps, 2);
    }
}
class tLossTree
{
    public static void t_simulate_it1()
    {
        double[] ps;
        int[] fv;
        ps = new double[] { 1, 0.4, 0.4, 0.4, 0.4, 0.4 };
        fv = new int[] { -1, 0, 1, 1, 1, 1 };
        int size = 3;
        //int[] types = new int[] { 0, 1, 2, 3, 4 };
        int[] types = new int[] { 0 };
        int n_tx_frames = 2000;
        LossTree t = new LossTree(fv, ps, size);
        foreach (double rate in new double[] { 0.2, 0.4, 0.6, 1.0 })
        {
            int[] results = t.simulate_it(n_tx_frames, rate, 0, 4);
            int sum = 0;
            foreach (int x in results)
                sum += x;
            Console.WriteLine("Rate {0}: sum = {1}", rate, sum);
        }
    }
    public static void t_simulate_it2()
    {
        double[] ps;
        int[] fv;
        ps = new double[] { 1, 0.4, 0.4, 0.4, 0.4, 0.4 };
        fv = new int[] { -1, 0, 1, 1, 1, 1 };
        int size = 20;
        //int[] types = new int[] { 0, 1, 2, 3, 4 };
        int[] types = new int[] { 0 };
        int n_tx_frames = 1000;
        LossTree t = new LossTree(fv, ps, size);
        int averages = 1000;
        double[] rate_v = new double[] { 0.2, 0.4, 0.6, 0.8, 1.0 };
        foreach (double rate in rate_v)
        {
            double sum = 0.0;
            for (int i = 0; i < averages; i++)
            {
                int[] results = t.simulate_it(n_tx_frames, rate, 0, i);
                foreach (int x in results)
                {
                    double c = (double)x / n_tx_frames / averages;
                    sum += c;
                }
            }
            Console.WriteLine("Rate {0}: sum = {1}", rate, sum);
        }
    }
    public static void tst_find_schedule()
    {
        double[] ps;
        int[] fv;
        ps = new double[] { 1, 0.6, 0.3, 0.3, 0.3, 0.3 };
        fv = new int[] { -1, 0, 1, 1, 1, 1 };
        int size = 20;
        LossTree t = new LossTree(fv, ps, size);
        t.find_schedule(10, 3);
        G.VB = true;
        t.show_schedule();
        G.VB = false;
        int[] types = new int[] { 0, 2};
        int n_tx_frames = 10000;
        foreach (int type in types)
        {
            int[] results = t.simulate_it(n_tx_frames, 0.6, type, 0);
            double sum = 0.0;
            foreach (int i in results)
            {
                sum += (double)i / n_tx_frames;
            }
            Console.WriteLine("Type {0}, sum = {1}", type, sum);
        }
    }
}
class tPgf
{
    public static void tst_plot_logical3()
    {
        int[] fv = new int[] { -1, 0, 1, 1 };
        double[] ps = new double[] { 1, 0.4, 0.3, 0.2 };
        PltGlb.plot_logical3(fv, ps, 2);
    }
    public static void tst_plot()
    {
        double[] xv = new double[] { 0, 1, 2, 3, 4 };
        double[] yv = new double[] { 1, 2, 1, 4, 5 };
        Pgf p = new Pgf();
        p.add("x axis", "y axis");
        p.plot(xv, yv, "mountain");
        p.save("ztest", 2);
    }
    public static void tst_Pgf1()
    {
        double[] xv = G.linspace(0, 9, 10);
        double[,] y1 = new double[10, 2];
        double[,] y2 = new double[10, 2];
        for (int j = 0; j < y1.GetLength(1); j++)
        {
            for (int i = 0; i < y1.GetLength(0); i++)
            {
                y1[i, j] = i * j;
                y2[i, j] = i * (j + 1);
            }
        }
        Pgf p = new Pgf();
        p.add("x", "y");
        p.mplot(xv, y1, new string[] { "normal", "double" });
        p.add("x", "y");
        p.mplot(xv, y2, new string[] { "normal", "double" });
        p.save("zb", 2);
    }
}
class G
{
    public static Stopwatch stopwatch = new Stopwatch();
    public static double INF = 99;
    public static Random rgen = new Random();
    public static bool VB = false;
    public static string comando;
    public static void prnt(String s)
    {
        if (G.VB) { Console.WriteLine(s); }
    }
    // get current method name
    public static string current()
    {
        System.Diagnostics.StackFrame stackframe = 
            new System.Diagnostics.StackFrame(1, true);
        
        return stackframe.GetMethod().ReflectedType.Name 
            + "."
            + stackframe.GetMethod().Name ;
    }
    public static void tst_1()
    {
        int[] fv = new int[] { -1, 0, 0, 1, 1 };
        double[] ps = new double[] { 0.5, 0.5, 0.5, 0.5, 0.5 };
        LossTree t = new LossTree(fv, ps, 30);
        Console.WriteLine("Hello World!");
        t.simulate_it(4, 0.4, 0, 0);
        int y = 7;
        Console.WriteLine("Something else" + " Hi" + y);
        List<int> x = new List<int>();
        for (int i = 0; i < 4; i++)
        {
            x.Add(i);
        }
        x.Remove(3);
        //Console.WriteLine(x.Capacity);
        Console.WriteLine(x.Count);
        Console.WriteLine(true | false);
    }
    public static double[] linspace(double init, double end, int number)
    {
        double[] x = new double[number];
        double spacing = (end - init) / (double)(number - 1);
        double y = init;
        for (int i = 0; i < number; i++)
        {
            x[i] = y;
            y += spacing;
        }
        return x;
    }
    public static string elapsed()
    {
        TimeSpan ts = stopwatch.Elapsed;
        return String.Format("{0:00}D{1:00}H:{2:00}M:{3:00}S, {4}",
                ts.Days, ts.Hours, ts.Minutes, ts.Seconds, comando);
    }
    public static void Main(string[] args)
    {
        Console.WriteLine("Execution started on {0}", DateTime.Now.ToString("u"));
        stopwatch = new Stopwatch();
        stopwatch.Start();

        ProdGlb.graphRateSize(0, 1, 1); comando = "ProdGlb.graphRateSize(0, 1, 1);"; //TOREPLACE
        //System.Threading.Thread.Sleep(1000);
        //tst_plot_logical3();
        //Console.ReadLine();
        //Tst.tLossTree.tst_find_schedule();
        stopwatch.Stop();
        TimeSpan ts = stopwatch.Elapsed;
        // Format and display the TimeSpan value.
        string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
            ts.Hours, ts.Minutes, ts.Seconds, ts.Milliseconds / 10);
        Console.WriteLine("Runtime = " + elapsed());
        Console.WriteLine("Execution terminated on {0}", DateTime.Now.ToString("u"));
    }
}
