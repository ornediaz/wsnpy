using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using System.IO;
namespace Plot
{
    class Glb
    {
        public static void plot_logical3(int[] fv, double[] ps, int plot)
        {
            if (plot == 0)
            {
                return;
            }
            string fname = "ztree";
            using (StreamWriter sw = File.CreateText(fname + ".dot"))
            {
                sw.Write("digraph tree {\n");
                for (int i = 0; i < fv.Length; i++)
                {
                    sw.Write("{0:d} -> {1:d} [label = {2:f2}];\n", i, fv[i], ps[i]);
                }
                sw.Write("}\n");
                sw.Close();
            }
            Process p1 = Process.Start("dot", String.Format(" -Tpdf {0}.dot -o {0}.pdf", fname));
            p1.WaitForExit();
            if (plot == 2)
            {
                Process p2 = Process.Start("dot", String.Format(" -Tpng {0}.dot -o {0}.png", fname));
                p2.WaitForExit();
                Process p3 = Process.Start(fname + ".png");
            }
        }
    }
}

