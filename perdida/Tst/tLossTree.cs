using System;
using System.Collections.Generic;
using System.Text;

namespace perdida.Tst
{
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
            perdida.Principal.VB = true;
            t.show_schedule();
            perdida.Principal.VB = false;
            int[] types = new int[] { 0, 2};
            int n_tx_frames = 10000;
            int averages = 100;
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
}
