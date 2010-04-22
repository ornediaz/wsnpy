using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

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
}
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
        if (plot > 0)
        {
            Process p = Process.Start("pdflatex", filename);
            p.WaitForExit();
        }
        if (plot == 2)
        {
            System.Diagnostics.Process.Start("acrord32", filename + ".pdf");
        }
    }
}
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
        if (Glb.VB)
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
        Glb.rgen = new Random(seed);
        int [] results = new int[(int) Math.Floor((n_tx_frames - 1) * rate + 1)];
        int sensing_interval = 0;
        for (int frame = 0; frame < n_tx_frames; frame++)
        {
            Glb.prnt("*** Simulating frame " + frame);
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
                Glb.prnt("Processing node " + n.ID);
                double random_number = Glb.rgen.NextDouble();
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
                Glb.prnt("Frame " + frame + "; Node " + n.ID + " position = " + position);
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
            x.q = (int) Math.Floor(x.ps * frames);
            foreach (Node y in x.ancestors)
            {
                x.q = Math.Min(x.q, (int) Math.Floor(y.ps * frames));
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
            Glb.prnt("Selected subtree of node " + best_tree[0].ID);
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
        Glb.prnt("Node " + n.ID + "gained frame " + frame);
    }
    public void show_schedule()
    {
        if (Glb.VB)
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
            }
        }
    }
}
class Glb
{
    public static Random rgen = new Random();
    public static bool VB = false;
    public static void prnt(String s)
    {
        if (Glb.VB) { Console.WriteLine(s); }
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
    public static void tst_Pgf1()
    {
        double[] xv = Glb.linspace(0, 9, 10);
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
    public static void graphRate1()
    {
        int tst_nr = 1;
        int n_averages = 100;
        //int action = 1;
        int frames = 10;
        int plot = 2;
        double [] ps;
        double opt = 0;
        int [] fv;
        int source_min = 3;
        if (tst_nr == 0)
        {
            ps = new double[] { 1, 0.8, 0.4, 0.4, 0.4, 0.4 };
            fv = new int[] { -1, 0, 1, 1, 1, 1 };
        }
        else if (tst_nr == 1)
        {
            ps = new double[] {1, 0.4, 0.4, 0.4, 0.4, 0.4};
            fv = new int[] {-1, 0, 1, 1, 1 , 1};
        }
        else
        {
            throw new ArgumentException("Inappropriate tst_nr");
        }
        double[] rate_v = Glb.linspace(0.1, 1.5, 28);
        int size = 30;
        //int[] types = new int[] { 0, 1, 2, 3, 4 };
        int[] types = new int[] { 0 };
        int n_tx_frames = 2000;
        double[,] sum = new double[rate_v.Length, types.Length];
        double[,] mean = new double[rate_v.Length, types.Length];
        double[,] pmin = new double[rate_v.Length, types.Length];
        Glb.VB = false;
        for (int k = 0; k < n_averages; k++)
        {
            if (VB)
            {
                Console.WriteLine("Executing repetition " + k);
            }
            for (int j = 0; j < rate_v.Length; j++)
            {
                for (int i = 0; i < types.Length; i++)
                {
                    Glb.rgen = new Random(k);
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
                        sum[j, i] += (double)h / n_averages / n_tx_frames;
                        mean[j, i] += (double)h / n_averages / results.Length;
                        if (h < source_min)
                        {
                            pmin[j, i] += (double)1 / n_averages / results.Length;
                        }
                    }
                }
            }
        }
        Console.WriteLine("**** Printing results *****");
        string[] legv = new string[] { "0", "1", "2", "3=" + opt, "4" };
        Pgf g = new Pgf();
        g.add("rate", "total");
        g.mplot(rate_v, sum, legv);
        g.add("rate", "mean");
        g.mplot(rate_v, mean, legv);
        g.add("rate", "pmin");
        g.mplot(rate_v, pmin, legv);
        string filename = String.Format("graphRate1_{0:d2}_{1:d6}", tst_nr,
                n_averages);
        g.save(filename, plot);
    }
    public static void tst_simulate_it1()
    {
        int frames;
        int plot = 2;
        double[] ps;
        double opt = 0;
        int[] fv;
        int source_min = 3;
        ps = new double[] { 1, 0.4, 0.4, 0.4, 0.4, 0.4 };
        fv = new int[] { -1, 0, 1, 1, 1, 1 };
        frames = 30;
        int size = 3;
        //int[] types = new int[] { 0, 1, 2, 3, 4 };
        int[] types = new int[] { 0 };
        int n_tx_frames = 2000;
        LossTree t = new LossTree(fv, ps, size);
        foreach (double rate in new double [] {0.2, 0.4, 0.6, 1.0})
        {
            int[] results = t.simulate_it(n_tx_frames, rate, 0, 4);
            int sum = 0;
            foreach (int x in results)
                sum += x;
            Console.WriteLine("Rate {0}: sum = {1}", rate, sum);
        }
    }
    public static void tst_simulate_it2()
    {
        int frames;
        int plot = 2;
        double[] ps;
        double opt = 0;
        int[] fv;
        int source_min = 3;
        ps = new double[] { 1, 0.4, 0.4, 0.4, 0.4, 0.4 };
        fv = new int[] { -1, 0, 1, 1, 1, 1 };
        frames = 30;
        int size = 20;
        //int[] types = new int[] { 0, 1, 2, 3, 4 };
        int[] types = new int[] { 0 };
        int n_tx_frames = 10000;
        LossTree t = new LossTree(fv, ps, size);
        int averages = 10;
        double[] rate_v = new double[] { 0.4, 0.8 };
        foreach (double rate in rate_v)
        {
            double sum = 0.0;
            for (int i = 0; i < averages; i++)
            {
                int[] results = t.simulate_it(n_tx_frames, rate, 0, i);
                foreach (int x in results)
                {
                    sum += ((double) x) / ((double) n_tx_frames / averages);
                }
            }
            Console.WriteLine("Rate {0}: sum = {1}", rate, sum);
        }
    }
    public static void Main(string[] args)
    {
        //tst_Pgf1();
        tst_simulate_it2();
        //Console.ReadLine();
    }
}

