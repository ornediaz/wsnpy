using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace Plot
{
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
}