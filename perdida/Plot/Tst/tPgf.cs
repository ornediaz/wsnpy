using System;
using System.Collections.Generic;
using System.Text;

namespace Plot.Tst
{
    class tPgf
    {
        public static void tst_plot_logical3()
        {
            int[] fv = new int[] { -1, 0, 1, 1 };
            double[] ps = new double[] { 1, 0.4, 0.3, 0.2 };
            PlgGlb.plot_logical3(fv, ps, 2);
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
            double[] xv = perdida.Principal.linspace(0, 9, 10);
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
}
