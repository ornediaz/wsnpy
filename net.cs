using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace ConsoleApplication1 {
class PgfAxis {
    public List<string> buf = new List<string>();
    public List<string> options = new List<string>();
    public List<string> legend = new List<string>();
    public PgfAxis(string xlabel, string ylabel) {
        options.Add("xlabel = { " + xlabel + "}");
        options.Add("ylabel = { " + ylabel + "}");}
    public void plot(double[] xv, double[] yv, string leg) {
        if (xv.Length != yv.Length) {
            throw new ArgumentException(); }
        buf.Add("      \\addplot coordinates {\n");
        for (int i = 0; i < xv.Length; i++) {
            if (i < xv.Length - 1) {
                buf.Add("        (" + xv[i] + ", " + yv[i] + ")\n"); }
            else {
                buf.Add("        (" + xv[i] + ", " + yv[i] + ")};\n");}}
        legend.Add(leg);}
    public void mplot(double[] xv, double[,] ym, string[] legv) {
        for (int j=0; j < ym.GetLength(1); j++) {
            double [] yv = new double [ym.GetLength(0)];
            for (int i = 0; i < ym.GetLength(0); i++) {
                yv[i] = ym[i, j]; }
            plot(xv, yv, legv[j]);}}}
class Pgf {
    public List<PgfAxis> body = new List<PgfAxis>();
    public List<string> extra_body = new List<string>();
    public List<string> extra_preamble = new List<string>();
    public Pgf() {
        extra_preamble.Add("\\usepackage{plotjour1}"); }
    public void add(string xlabel, string ylabel) {
        body.Add(new PgfAxis(xlabel, ylabel)); }
    public void plot(double[] xv, double[] yv, string leg) {
        body[body.Count - 1].plot(xv, yv, leg); }
    public void mplot(double[] xv, double[,] ym, string[] legv) {
        body[body.Count - 1].mplot(xv, ym, legv); }
    public void save(string filename) {
        List<string> lst = new List<string>();
        lst.Add("\\documentclass{article}\n");
        lst.Add("\\usepackage[margin=0in]{geometry}\n");
        lst.Add("\\usepackage{orne1}\n");
        foreach (string s in extra_preamble) {
            lst.Add(s + "\n"); }
        lst.Add("\\begin{document}\n");
        for (int i = 0; i < body.Count; i++) {
            lst.Add("  \\begin{tikzpicture}\n");
            lst.Add("    \\begin{axis} [\n");
            for (int j = 0; j < body[i].options.Count; j++) {
                if (j < body[i].options.Count - 1) {
                    lst.Add("      " + body[i].options[j] + ", \n"); }
                else {
                    lst.Add("      " + body[i].options[j] + "\n"); } }
            lst.Add("      ]\n");
            foreach (string s in body[i].buf) {
                lst.Add(s); }
            lst.Add("      " + "\\legend{{");
            for (int j = 0; j < body[i].legend.Count; j++) {
                if (j < body[i].legend.Count - 1) {
                    lst.Add(body[i].legend[j] + "}, {"); }
                else {
                    lst.Add(body[i].legend[j] + "}}%\n"); } }
            lst.Add("    \\end{axis}\n");
            lst.Add("  \\end{tikzpicture}\n");
            if ((i % 2) == 1) {
                lst.Add("\n"); } }
        foreach (string s in extra_body) {
            lst.Add(s); }
        lst.Add("\\end{document}");
        using (StreamWriter sw = File.CreateText(filename + ".tex")) {
            foreach (string s in lst) {
                sw.Write(s); }
            sw.Close(); }
        Process p = Process.Start("pdflatex", filename);
        p.WaitForExit();
        System.Diagnostics.Process.Start("acrord32", filename + ".pdf"); } }
class Packet {
    public int t;
    public int k;
    public Packet(int t, int k) {
        this.t = t;
        this.k = k; } }
class Node {
    public int ID;
    public int discard_type = 0;
    public int select_type = 0;
    public List<Packet> pkts = new List<Packet>();
    public List<Node> ancestors = new List<Node>(); // does not include sink
    public List<Node> ch = new List<Node>();
    public List<int> gen = new List<int>();
    public Node f = null;
    public double ps;
    public int q = 0; 
    // maximum number of packets that a node can transmit per hyperframe. 
    public int size;
    public Node(int ID) {
        this.ID = ID; } }
class LossTree {
    public Node[] nodes;
    public List<Node> postorder = new List<Node>(); // does not include sink
    public List<int> count = new List<int>(); 
    // How many measurements are provided for each sensing slot.
    private void add_postorder(Node x) {
        foreach (Node n in x.ch) {
            add_postorder(n); }
        postorder.Add(x); }
    public LossTree(int[] fv, double[] ps, int size) {
        nodes = new Node[fv.Length];
        for (int i = 0; i < fv.Length; i++) {
            nodes[i] = new Node(i);
            nodes[i].ps = ps[i];
            nodes[i].size = size; }
        for (int i = 1; i < fv.Length; i++) {
            nodes[i].f = nodes[fv[i]]; 
            nodes[fv[i]].ch.Add(nodes[i]);}
        for (int i = 1; i < fv.Length; i++) {
            Node ancestor = nodes[fv[i]];
            while (ancestor != null && ancestor.ID !=0) {
                nodes[i].ancestors.Add(ancestor);
                ancestor = ancestor.f; } }
        foreach (Node n in nodes[2].ancestors) {
            Console.WriteLine(n.ID); }
        foreach (Node n in nodes[0].ch) {
            add_postorder(n); }
        Console.WriteLine("Postorder list");
        foreach (Node n in postorder) {
            Console.WriteLine(n.ID); } }
    private void discard(Node n, int type, double rate) {
        int position = 0;
        List<int> l = new List<int>();
        while (n.pkts.Count > n.size) {
            if (n.discard_type == 0) {
                position = 0;
                for (int i = 0; i < n.pkts.Count; i++) {
                    if (n.pkts[i].t < n.pkts[position].t) {
                        position = i; } } }
            else if (n.discard_type == 1) {
                int kmin = 999999;
                foreach (Packet p in n.pkts) {
                    kmin = Math.Min(kmin, p.k); }
                l.Clear();
                for (int i = 0; i < n.pkts.Count; i++) {
                    if (n.pkts[i].k == kmin) {
                        l.Add(i); } }
                position = l[0];
                foreach (int i in l) {
                    if (n.pkts[i].t < n.pkts[position].t) {
                        position = i; } } }
            else if (n.discard_type == 2) {
                l.Clear();
                for (int i = 0; i < n.pkts.Count; i++) {
                    if (n.gen.Contains(n.pkts[i].t % count.Count)) {
                        l.Add(i); } }
                if (l.Count == 0) {
                    for (int i = 0; i < n.pkts.Count; i++) {
                        l.Add(i); } }
                position = l[0];
                foreach (int i in l) {
                    if (n.pkts[i].t < n.pkts[position].t) {
                        position = i; } } }
            else {
                throw new Exception("Incorrect discard_type"); } }
        n.pkts.RemoveAt(position); }
    public int[] simulate_it(int iterations, double rate, int type, int seed) {
        List<int> l = new List<int>();
        foreach (Node n in nodes) {
            if (type == 0)        { n.discard_type = 0; }
            else if (type == 1)   { n.discard_type = 0; }
            else if (type == 2)   { n.discard_type = 1; }
            else if (type == 3)   { n.discard_type = 2; }
            else if (type == 4) {
                double m = 1.0;
                for (int i = 0; i < n.ancestors.Count - 1; i++) {
                    m = Math.Min(m, n.ancestors[i].ps); }
                if (rate < m) { n.discard_type = 0; }
                else          { n.discard_type = 1; } }
            else {
                throw new ArgumentException("Incorrect type"); } 
            if (type == 0) {
                n.select_type = 0; }
            else if (type == 1) {
                n.select_type = 1; }
            else if (type == 2) {
                n.select_type = 1; }
            else if (type == 3) {
                n.select_type = 2; }
            else if (type == 4) {
                double m = 1.0;
                for (int i = 0; i < n.ancestors.Count - 1; i++) {
                    m = Math.Min(m, n.ancestors[i].ps); }
                if (rate < m) {
                    n.select_type = 0; }
                else {
                    n.select_type = 1; } }
            else if (type > 4) {
                throw new ArgumentException("Incorrect type"); } }
        int[] results = new int[iterations];
        Packet pkt;
        int position = 0;
        int maxcount;
        double random_number;
        int old = 0;
        int select_type = 0;
        //Random rgen = new Random();
        for (int frame = 0; frame < iterations; frame++) {
            Glb.prnt("*** Simulating frame " + frame);
            while (old <= frame * rate) {
                for (int i = 1; i < nodes.Length; i++) {
                    nodes[i].pkts.Add(new Packet(old, 1)); }
                old += 1; }
            foreach (Node n in postorder) {
                Glb.prnt("Processing node " + n.ID);
                random_number = Glb.rgen.NextDouble();
                if (n.pkts.Count == 0 || random_number >= n.ps) {
                    continue; }
                if (select_type == 0) {
                    position = 0;
                    for (int i = 0; i < n.pkts.Count; i++) {
                        if (n.pkts[i].t < n.pkts[position].t) {
                            position = i; } } }
                else if (select_type == 1) {
                    maxcount = 0;
                    foreach (Packet p in n.pkts) {
                        maxcount = Math.Max(maxcount, p.k); }
                    l.Clear();
                    for (int i = 0; i < n.pkts.Count; i++) {
                        if (n.pkts[i].k == maxcount) {
                            l.Add(i); } }
                    position = l[0];
                    foreach (int i in l) {
                        if (n.pkts[i].t < n.pkts[position].t) {
                            position = i; } } }
                else if (select_type == 2) {
                    l.Clear();
                    for (int i = 0; i < n.pkts.Count; i++) {
                        if (n.gen.Contains(n.pkts[i].t % count.Count)) {
                            l.Add(i); } }
                    if (l.Count == 0) {
                        for (int i = 0; i < n.pkts.Count; i++) {
                            l.Add(i); } }
                    position = l[0];
                    foreach (int i in l) {
                        if (n.pkts[i].t < n.pkts[position].t) {
                            position = i; } } }
                Glb.prnt("Frame " + frame + "; Node " + n.ID + 
                        " position = " + position);
                pkt = n.pkts[position];
                n.pkts.RemoveAt(position);
                Glb.prnt("Node "+n.ID+" tx: "+pkt.t + ", " + pkt.k);
                bool added = false;
                for (int i = 0; i < n.f.pkts.Count; i++) {
                    if (n.f.pkts[i].t == pkt.t) {
                        n.f.pkts[i].k += pkt.k;
                        added = true; } }
                if (added == false) {
                    n.f.pkts.Add(pkt); } } }
        return results; } 
    public void find_schedule(int frames, int source_min){
        int q = frames;
        foreach (Node x in postorder) {
            q = (int) (x.ps * frames);
            foreach (Node y in x.ancestors) {
                q = Math.Min(q, (int) (y.ps * frames))}}
        Node y;
        int max_ancestors;
        List<Node> tree = new List<Node>;
        Stack<Node> stack = new Stack<Node>;
        List<List<Node>> tree_list = new List<List<Node>>();
        int max_frames = 9999;
        for (int frame = 0; frame < max_frames; frame++) {
            tree_list.Clear();
            foreach (Node n in postorder) {
                bool unsuitable = false;
                tree.Clear();
                tree.Add(n);
                tree.AddRange(n.ancestors);
                foreach (Node x in tree) {
                    if (x.q <= 0) {
                        unsuitable = true;}}
                stack.Clear();
                foreach (Node x in n.ch) {
                    stack.Push(x);}
                while (stack.Count > 0 && tree.Count < source_min) {
                    y = stack.Pop();
                    if (y.q > 0) {
                        tree.Add(y);
                        foreach (z in y.ch) {
                            tree.Add(z);}}}
                if (tree.Count >= source_min) {
                    tree_list.Add(tree);}}
            if (tree_list.Count == 0) {
                break;}
            tree = tree_list[0];
            max_ancestors = tree[0].ancestors.Count;
            foreach (List<Node> auxtree in tree_list) {
                if (auxtree[0].ancestors.Count >  tree[0].ancestors.Count) {
                    tree = auxtree;}}
            Glb.prnt("Selected subtree of node " + tree[0].ID);
            foreach (Node n in tree) {
                add_frame(n, frame);}}
            nodes[0].gen.Add(frame);
        if (frame == max_frames - 1) {
            throw new Exception();}
        if (count.Count == 0) {
            throw new Exception("source_min is too small");}
        stack.Clear();
        foreach (Node n in nodes[0].ch) {
            stack.




                
                

            



}
class Glb {
public static Random rgen = new Random();
public static bool VB = false;
public static void prnt(String s) {
    if (Glb.VB) { Console.WriteLine(s); } }
public static void tst_plot() {
    double[] xv = new double[] { 0, 1, 2, 3, 4 };
    double[] yv = new double[] { 1, 2, 1, 4, 5 };
    Pgf p = new Pgf();
    p.add("x axis", "y axis");
    p.plot(xv, yv, "mountain");
    p.save("ztest"); }
public static void tst_1() {
    int[] fv = new int[] { -1, 0, 0, 1, 1 };
    double[] ps = new double[] { 0.5, 0.5, 0.5, 0.5, 0.5 };
    LossTree t = new LossTree(fv, ps, 30);
    Console.WriteLine("Hello World!");
    t.simulate_it(4, 0.4, 0, 0);
    int y = 7;
    Console.WriteLine("Something else" + " Hi" + y);
    List<int> x = new List<int>();
    for (int i = 0; i < 4; i++) {
        x.Add(i); }
    x.Remove(3);
    //Console.WriteLine(x.Capacity);
    Console.WriteLine(x.Count);
    Console.WriteLine(true | false); }
public static void tst_Pgf1() {
    double [] xv = Glb.linspace(0, 9, 10);
    double [,] y1 = new double[10, 2];
    double [,] y2 = new double[10, 2];
    for (int j = 0; j < y1.GetLength(1); j++) {
        for (int i = 0; i < y1.GetLength(0); i++) {
            y1[i, j] = i * j;
            y2[i, j] = i * (j + 1); } }
    Pgf p = new Pgf();
    p.add("x", "y");
    p.mplot(xv, y1, new string [] {"normal", "double"});
    p.add("x", "y");
    p.mplot(xv, y2, new string [] {"normal", "double"});
    p.save("zb"); }
public static double [] linspace(double init, double end, int number) {
    double[] x = new double[number];
    double spacing = (end - init) / (double) (number - 1);
    double y = init;
    for (int i = 0; i < number; i++) {
        x[i] = y;
        y += spacing; }
    return x; }
public static void graphRate1() {
    int tst_nr = 0;
    int repetitions = 2;
    int action = 1;
    if (tst_nr == 0) {
        double[] ps = new double[] {1, 0.8, 0.4, 0.4, 0.4, 0.4};
        int[] fv = new int[] {-1, 0, 1, 1, 1, 1}; } 
    else {
        throw new ArgumentException("Inappropriate tst_nr"); }
    double[] rate_v = Glb.linspace(0.5, 1, 4);
    foreach (double t in rate_v) {
        Console.WriteLine(t); }
    int size = 30;
    int [] types = new int[] {0, 2};
    int iterations = 10;
    double [,,] mean = new double [repetitions,rate_v.Length,types.Length];
    Glb.VB = true;
    for (int k = 0; k < repetitions; k++) {
        if (VB) {
            Console.WriteLine("Executing repetition " + k); }
        for (int j = 0; j < rate_v.Length; j++) {
            for (int i = 0; i < rate_v.Length; i++) {
                Glb.rgen = new Random(k); }}}}
public static void Main(string[] args) {
    tst_Pgf1();
    //graphRate1();
}
}
}
