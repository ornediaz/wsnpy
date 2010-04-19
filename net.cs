using System;
using System.Collections.Generic;
using System.IO;

namespace ConsoleApplication1
{
class PgfAxis
{
    public List<string> buf = new List<string>();
    public List<string> options = new List<string>();
    public List<string> legend = new List<string>();
    public PgfAxis(string xlabel, string ylabel)
    {
        options.Add("xlabel = { " + xlabel + "}");
    }
    public void plot(double[] xv, double[] yv, string leg)
    {
        if (xv.Length != yv.Length)
        {
            throw new ArgumentException("The two vectors should have equal" +
                    " lengths.");
        }
        buf.Add("      \\addplot coordinates {\n");
        for (int i = 0; i < xv.Length; i++)
        {
            if (i < xv.Length - 1)
            {
                buf.Add("        (" + xv[i] + ")\n");
            }
            else
            {
                buf.Add("        (" + xv[i] + ")}};\n");
            }
        }
        legend.Add(leg);
    }
    public void mplot(double[] xv, double[][] ym, string[] legv)
    {
        Console.WriteLine("Hey");
    }
}
class Pgf
{
    public List<PgfAxis> body = new List<PgfAxis>();
    public List<string> extra_body = new List<string>();
    public List<string> extra_preamble = new List<string>();
    public Pgf()
    {
        extra_preamble.Add("\\usepackageplotjour1");
    }
    public void add(string xlabel, string ylabel)
    {
        body.Add(new PgfAxis(xlabel, ylabel));
    }
    public void plot(double[] xv, double[] yv, string leg)
    {
        body[body.Count - 1].plot(xv, yv, leg);
    }
    public void mplot(double[] xv, double[][] ym, string[] legv)
    {
        body[body.Count - 1].mplot(xv, ym, legv);
    }
    public void save(string filename)
    {
        Console.WriteLine("Hey you");
        List<string> lst = new List<string>();
        lst.Add("\\documentclass{article}\n");
        lst.Add("\\usepackage[margin=0in]{geometry>\n");
        lst.Add("\\usepackage{orne1}\n");
        foreach (string s in extra_preamble)
        {
            lst.Add(s);
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
        Console.WriteLine("Hey 1");
        foreach (string s in extra_body)
        {
            lst.Add(s);
        }
        Console.WriteLine("Hey 2");
        lst.Add("\\end{document}");
        foreach (string s in lst)
        {
            Console.WriteLine(s);
        }
        Console.WriteLine("Hey again");
        using (StreamWriter sw = File.CreateText(filename))
        {
            foreach (string s in lst)
            {
                sw.Write(s);
            }
            sw.Close();
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
    public List<Packet> pkts = new List<Packet>();
    public List<Node> ancestors = new List<Node>();
    public List<Node> ch = new List<Node>();
    public List<int> gen = new List<int>();
    public Node f;
    public double ps;
    public int size;
    public Node(int ID)
    {
        this.ID = ID;
        f = null;
    }
}
class LossTree
{
    public bool VB;
    public Node[] nodes;
    public List<Node> postorder;
    private void add_postorder(Node x)
    {
        foreach (Node n in x.ch)
        {
            add_postorder(n);
        }
        postorder.Add(x);
    }
    public void prnt(String s)
    {
        if (VB)
        {
            Console.WriteLine(s);
        }
    }

    public LossTree(int[] fv, double[] ps, int size, bool verbose)
    {
        VB = verbose;
        nodes = new Node[fv.Length];
        for (int i = 0; i < fv.Length; i++)
        {
            nodes[i] = new Node(i);
            nodes[i].ps = ps[i];
            nodes[i].size = size;
        }
        for (int i = 1; i < fv.Length; i++)
        {
            nodes[fv[i]].ch.Add(nodes[i]);
            nodes[i].f = nodes[fv[i]];
        }
        for (int i = 1; i < fv.Length; i++)
        {
            for (Node ancestor = nodes[fv[i]]; ancestor != null; ancestor = ancestor.f)
            {
                nodes[i].ancestors.Add(ancestor);
            }
        }
        foreach (Node n in nodes[2].ancestors)
        {
            Console.WriteLine(n.ID);
        }
        postorder = new List<Node>();
        foreach (Node n in nodes[0].ch)
        {
            add_postorder(n);
        }
        Console.WriteLine("Postorder list");
        foreach (Node n in postorder)
        {
            Console.WriteLine(n.ID);
        }
    }
    private void discard(Node n, int type, double rate)
    {
        int discard_type;
        int position = 0;
        if (n.ID == 0) { return; }
        if (type == 0)
        {
            discard_type = 0;
        }
        else if (type == 1)
        {
            discard_type = 0;
        }
        else if (type == 2)
        {
            discard_type = 1;
        }
        else if (type == 3)
        {
            discard_type = 2;
        }
        else if (type == 4)
        {
            double m = 1.0;
            for (int i = 0; i < n.ancestors.Count - 1; i++)
            {
                m = Math.Min(m, n.ancestors[i].ps);
            }
            if (rate < m) { discard_type = 0; }
            else { discard_type = 1; }
        }
        else
        {
            throw new ArgumentException("Incorrect type");
        }
        while (n.pkts.Count > n.size)
        {
            if (discard_type == 0)
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
            else if (discard_type == 1)
            {
                int nmin = 99999;
                foreach (Packet p in n.pkts)
                {
                    nmin = Math.Min(nmin, p.k);
                }
                List<int> l = new List<int>();
                for (int i = 0; i < n.pkts.Count; i++)
                {
                    if (n.pkts[i].k == nmin)
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
            else if (discard_type == 2)
            {
                List<int> l = new List<int>();
                for (int i = 0; i < n.pkts.Count; i++)
                {
                    if (n.gen.Contains(n.pkts[i].t))
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
        }
        n.pkts.RemoveAt(position);
    }
    public int[] simulate_it(int iterations, double rate, int type, int seed)
    {
        int[] results = new int[iterations];
        Packet pkt;
        int position = 0;
        int select_type = 0;
        Random rgen = new Random();
        if (type > 4)
        {
            throw new ArgumentException("Inappropriate type");
        }
        int old = 0;
        for (int frame = 0; frame < iterations; frame++)
        {
            if (VB)
            {
                Console.WriteLine("*** Simulating frame " + frame);
            }
            while (old <= frame * rate)
            {
                for (int i = 1; i < nodes.Length; i++)
                {
                    nodes[i].pkts.Add(new Packet(old, 1));
                }
                old += 1;
            }
            foreach (Node n in postorder)
            {
                if (VB)
                {
                    Console.WriteLine("Processing node " + n.ID);
                }
                double r = rgen.NextDouble();
                Console.WriteLine(nodes[0].pkts.Count);
                if (n.pkts.Count == 0 || r >= n.ps)
                {
                    continue;
                }
                if (type == 0)
                {
                    select_type = 0;
                }
                else if (type == 1)
                {
                    select_type = 1;
                }
                else if (type == 2)
                {
                    select_type = 1;
                }
                else if (type == 3)
                {
                    select_type = 2;
                }
                if (type == 4)
                {
                    double m = 1.0;
                    for (int i = 0; i < n.ancestors.Count - 1; i++)
                    {
                        m = Math.Min(m, n.ancestors[i].ps);
                    }
                    if (rate < m)
                    {
                        select_type = 0;
                    }
                    else
                    {
                        select_type = 1;
                    }
                }
                if (select_type == 0)
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
                else if (select_type == 1)
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
                else if (select_type == 2)
                {
                    List<int> l = new List<int>();
                    for (int i = 0; i < n.pkts.Count; i++)
                    {
                        if (n.gen.Contains(n.pkts[i].t))
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
                Console.WriteLine("Iteration " + frame + "; Node " + n.ID + " position = " + position);
                pkt = n.pkts[position];
                n.pkts.RemoveAt(position);
                if (this.VB)
                {
                    Console.WriteLine("Node " + n.ID + " transmitted: " + pkt.t + ", " + pkt.k);
                }
                bool added = false;
                for (int i = 0; i < n.f.pkts.Count; i++)
                {
                    if (n.f.pkts[i].t == pkt.t)
                    {
                        n.f.pkts[i].k += pkt.k;
                        added = true;
                    }
                }
                if (added == false)
                {
                    n.f.pkts.Add(pkt);
                }
            }
        }
        return results;
    }
}
class Program
{
    public static void Main(string[] args)
    {
        tst_plot();
    }
    public static void tst_plot()
    {
        double[] xv = new double[] { 0, 1, 2, 3, 4 };
        double[] yv = new double[] { 1, 2, 1, 4, 5 };
        Pgf p = new Pgf();
        p.add("x axis", "y axis");
        p.plot(xv, yv, "mountain");
        p.save("test.txt");
    }
    public static void tst_1()
    {
        int[] fv = new int[] { -1, 0, 0, 1, 1 };
        double[] ps = new double[] { 0.5, 0.5, 0.5, 0.5, 0.5 };
        LossTree t = new LossTree(fv, ps, 30, true);
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
}
}
