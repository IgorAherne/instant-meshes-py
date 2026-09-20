/*
    globals.cpp: definitions for the handful of globals the algorithm core
    declares `extern`.

    These used to live in main.cpp, which made libraries built from the core
    unlinkable on their own.  Keeping them here means every front end -- the
    Python extension, the desktop GUI, a test harness -- links against the same
    single definition.

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

/* Worker threads for the task scheduler; -1 means "use every core".
   Read by Optimizer::run (src/field.cpp) and by main.cpp's -t option. */
int nprocs = -1;
