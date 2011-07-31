/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

/* CUda UTility Library */

// includes, file
#include <stopwatch_win.h>

////////////////////////////////////////////////////////////////////////////////
// Variables, static

//! tick frequency
/*static*/ double  StopWatchWin::freq;

//! flag if the frequency has been set
/*static*/  bool   StopWatchWin::freq_set;

////////////////////////////////////////////////////////////////////////////////
//! Constructor, default
////////////////////////////////////////////////////////////////////////////////
StopWatchWin::StopWatchWin() :
    start_time(),
    end_time(),
    diff_time( 0.0),
    total_time(0.0),
    running( false),
    clock_sessions(0)
{
    if( ! freq_set) 
    {
        // helper variable
        LARGE_INTEGER temp;

        // get the tick frequency from the OS
        QueryPerformanceFrequency((LARGE_INTEGER*) &temp);

        // convert to type in which it is needed
        freq = ((double) temp.QuadPart) / 1000.0;

        // rememeber query
        freq_set = true;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Destructor
////////////////////////////////////////////////////////////////////////////////
StopWatchWin::~StopWatchWin() { }

