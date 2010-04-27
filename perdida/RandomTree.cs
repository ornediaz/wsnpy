using System;
using System.Collections.Generic;
using System.Text;

namespace perdida
{
    class RandomTree
    {
        public static int[] parents(int n, double x, double y, double tx_rg)
        {
            // n is  the number of nodes
            // x, y are the geographical area
            // tx_rg
            int[] fv = new int[n];
            double[,] xy = new double[n,2];
            double[,] cost = new double[n,n];
            for (int i = 1; i < n; i++)
            {
                xy[i, 0] = perdida.Principal.rgen.NextDouble() * x;
                xy[i, 1] = perdida.Principal.rgen.NextDouble() * y;
            }
            xy[0, 0] = 0;
            xy[0, 1] = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    cost[i, j] = Math.Sqrt(Math.Pow(xy[i, 0] - xy[j, 0], 2) + Math.Pow(xy[i, 1] - xy[j, 1], 2));
                }
            }
            return dijkstra(cost);
        }
        public static int[] dijkstra(double[,] cost)
        {
            // Return every node's next hop to the sink using Dijkstra's algorithm.
            // Parameter:
            // cost -- NxN ndarray indicating cost of N nodes '''
            int N = cost.GetLength(0);
            //len(cost);

            double[] dst = new double[N];// Each node's smallest distance to the sink
            for (int i = 1; i < N; i++)
            {
                dst[i] = perdida.Principal.INF;
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
                if (dst[x] == perdida.Principal.INF)
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
            double y = 10;
            double tx_rg = 2;
            double rho = 8;
            int n = (int) (x * y / Math.PI / rho / tx_rg / tx_rg);
            int[] fv = parents(n, x, y, tx_rg);
            double[] ps = new double[n];
            Plot.PlgGlb.plot_logical3(fv, ps, n);
        }
    }
}
