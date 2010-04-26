using System;

namespace perdida.Prod
{
    class ProdGlb
    {
        public static void graphRate1(int tst_nr, int n_averages, int plot)
        {
            int frames = 10;
            int[] fv;
            int n_tx_frames = 2000;
            double opt = 0;
            double[] ps;
            double[] rate_v = perdida.Principal.linspace(0.1, 1.5, 28);
            int source_min = 3;
            int size = 30;
            int[] types = new int[] { 0, 1, 2, 3, 4 };
            perdida.Principal.VB = false;
            if (tst_nr == 0)
            {
                fv = new int[] { -1, 0, 1, 1, 1, 1 };
                ps = new double[] { 1, 0.8, 0.4, 0.4, 0.4, 0.4 };
            }
            else if (tst_nr == 1)
            {
                fv = new int[] { -1, 0, 1, 1, 1, 1 };
                ps = new double[] { 1, 0.4, 0.4, 0.4, 0.4, 0.4 };
            }
            else if (tst_nr == 2)
            {
                fv = new int[] { -1, 0, 1, 1, 1, 1 };
                ps = new double[] { 1, 0.4, 0.8, 0.8, 0.8, 0.8 };
            }
            else if (tst_nr == 3)
            {
                fv = new int[] { -1, 0, 1, 1, 1, 1 };
                ps = new double[] { 1, 0.6, 0.6, 0.6, 0.2, 0.2 };
            }
            else if (tst_nr == 4)
            {
                fv = new int[] { -1, 0, 1, 1, 1, 1 };
                ps = new double[] { 1, 0.3, 0.6, 0.6, 0.2, 0.2 };
            }
            else if (tst_nr == 5)
            {
                fv = new int[] { -1, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5 };
                ps = new double[12];
                for (int i = 0; i < fv.Length; i++)
                {
                    ps[i] = 0.3;
                }
            }
            else if (tst_nr == 6)
            {
                fv = new int[] { -1, 0, 1, 2, 3, 4, 5 };
                ps = new double[fv.Length];
                for (int i = 0; i < ps.Length; i++)
                {
                    ps[i] = 0.3;
                }
            }
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
            else
            {
                throw new ArgumentException("Inappropriate tst_nr");
            }
            Plot.PlgGlb.plot_logical3(fv, ps, plot);
            double[,] sum = new double[rate_v.Length, types.Length];
            double[,] mean = new double[rate_v.Length, types.Length];
            double[,] pmin = new double[rate_v.Length, types.Length];
            for (int k = 0; k < n_averages; k++)
            {
                if (perdida.Principal.VB)
                {
                    Console.WriteLine("Executing repetition " + k);
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
            string[] legv = new string[] { "0", "1", "2", String.Format("3={0:F3}", opt), "4" };
            Plot.Pgf g = new Plot.Pgf();
            g.add("rate", "total");
            g.mplot(rate_v, sum, legv);
            g.add("rate", "mean");
            g.mplot(rate_v, mean, legv);
            g.add("rate", "pmin");
            g.mplot(rate_v, pmin, legv);
            g.extra_body.Add("\n\\includegraphics[scale=0.4]{ztree.pdf}\n");
            string filename = String.Format("graphRate1_{0:d2}_{1:d6}", tst_nr,
                    n_averages);
            g.save(filename, plot);
        }
        public static void multiplot1()
        {
            int n_averages = 20;
            for (int tst_nr = 0; tst_nr < 12; tst_nr++)
            {
                Console.WriteLine("Iteration {0} of multiplot", tst_nr);
                ProdGlb.graphRate1(tst_nr, n_averages, 1);
            }

        }
    }
}
