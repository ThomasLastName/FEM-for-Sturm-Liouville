# FEM for Sturm Liouville
Use FEM to approximate a solution to the Sturm-Liouville problem with homogeneous boundary conditions in the span of some pointy bois (https://en.wikipedia.org/wiki/Triangular_function).
Slightly more information, though admittedly not a complete discussion, can be found in the corresponding writeup for the original problem (https://www.overleaf.com/read/wpbppmwyjnkw).

The algorithm fails ocasionally, but when it does so it will not lead to false results, as it will fail spectacularly. If that happens, literally just close python and re-run the code.
I don't understand why the same code works sometimes and fails others (and once it has been successfully run, it typically will no longer fail).
This behavior is somehow related to Numba (which I still do not completely understand), and if anyone has any insight on the matter, an email on the subject is welcome (my address found at https://sites.google.com/view/thomas-winckelman/introduction).
