using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using System.IO;


namespace perdida
{
    class Principal
    {
        public static Random rgen = new Random();
        public static bool VB = false;
        public static void prnt(String s)
        {
            if (Principal.VB) { Console.WriteLine(s); }
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


        public static void Main(string[] args)
        {
            //tst_Pgf1();
            //tst_simulate_it2();
            Prod.Glb.graphRate1();
            //tst_plot_logical3();
            //Console.ReadLine();
        }
    }
}
