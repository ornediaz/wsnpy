using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace Plot
{
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
}