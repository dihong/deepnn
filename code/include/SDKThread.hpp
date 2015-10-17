/**********************************************************************
Copyright ©2014 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
#ifndef _SDK_THREAD_H_
#define _SDK_THREAD_H_

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif

/**
 * Header Files
 */
#include "windows.h"
#include <deque>
#include "assert.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <process.h>
#endif
#define EXPORT __declspec(dllexport)

#else
#include "pthread.h"
#define EXPORT
#endif

/**
 * suppress the warning #810 if intel compiler is used.
 */
#if defined(__INTEL_COMPILER)
#pragma warning(disable : 810)
#endif

//#define PRINT_COND_VAR_ERROR_MSG
#ifdef PRINT_COND_VAR_ERROR_MSG
#define PRINT_ERROR_MSG(errorcode, msg) \
    if(errorcode != 0) \
        printf("%s \n", msg)
#else
#define PRINT_ERROR_MSG(errorcode, msg)
#endif // PRINT_COND_VAR_ERROR_MSG

/**
 * namespace appsdk
 */
namespace appsdk
{
/**
 * Entry point for the thread
 * prototype of the entry point in windows
 */
typedef void* (*threadFunc)(void*);

//! pack the function pointer and data inside this struct
typedef struct __argsToThreadFunc
{
    threadFunc func;
    void* data;

} argsToThreadFunc;



/**
 * class ThreadLock
 *  \brief Provides a wrapper for locking primitives used to
 *   synchronize _CPU_ threads.
 *
 *  Common usage would be:
 *
 *     CALLock lock;
 *
 *     // Critical section begins
 *
 *     lock.lock();
 *
 * Critical section ends
 *
 *    lock.unlock();
 */

/**
 * class ThreadLock
 */
class EXPORT ThreadLock
{
    public:

        /**
         * Constructor
        */
        ThreadLock()
        {
#ifdef _WIN32
            InitializeCriticalSection(&_cs);
#else
            pthread_mutex_init(&_lock, NULL);
#endif
        }

        /**
         * Destructor
        */
        ~ThreadLock()
        {
#ifdef _WIN32
            DeleteCriticalSection(&_cs);
#else
            pthread_mutex_destroy(&_lock);
#endif
        }

        /**
         * Returns true if the lock is already locked, false otherwise
        */
        bool isLocked()
        {
#ifdef _WIN32
            return (_cs.LockCount != ~0x0);
#else
            if(pthread_mutex_trylock(&_lock) != 0)
            {
                return true;
            }
            pthread_mutex_unlock(&_lock);
            return false;
#endif
        }

        /**
         * Try to acquire the lock, if available continue, else wait on the lock
        */
        void lock()
        {
#ifdef _WIN32
            EnterCriticalSection(&_cs);
#else
            pthread_mutex_lock(&_lock);
#endif
        }

        /**
         * Try to acquire the lock, if available, hold it, else continue doing something else
        */
        bool tryLock()
        {
#ifdef _WIN32
            return (TryEnterCriticalSection(&_cs) != 0);
#else
            return !((bool)pthread_mutex_trylock(&_lock));
#endif
        }

        /**
         * Unlock the lock and return
        */
        void unlock()
        {
#ifdef _WIN32
            LeaveCriticalSection(&_cs);
#else
            pthread_mutex_unlock(&_lock);
#endif
        }

    private:

        /**
         * Private data members and methods
         */

        /**
         * System specific synchronization primitive
        */
#ifdef _WIN32
        CRITICAL_SECTION _cs;
#else
        pthread_mutex_t _lock;
#endif
};


#ifdef _WIN32
unsigned _stdcall win32ThreadFunc(void* args);
#endif
/**
 * \class Thread
 * \brief Provides a wrapper for creating a _CPU_ thread.
 * This class provides a simple wrapper to a CPU thread/
 */
class EXPORT SDKThread
{
    public:
        /**
         * Thread constructor and destructor. Note that the thread is
         * NOT created in the constructor. The thread creation takes
         *  place in the create method
        */
        SDKThread() : _tid(0), _data(0)
        {
        }

        ~SDKThread()
        {
#ifdef _WIN32
            if(_tid)
            {
                CloseHandle(_tid);
                _tid = 0;
            }
#endif
        }

        /**
         * Wrapper for pthread_create. Pass the thread's entry
         * point and data to be passed to the routine
        */
        bool create(threadFunc func, void* arg)
        {
            // Save the data internally
            _data = arg;
#ifdef _WIN32
            // Setup the callback struct for thread function and pass to the
            // begin thread routine
            // xxx The following struct is allocated but never freed!!!!
            argsToThreadFunc *args =  new argsToThreadFunc;
            args->func = func;
            args->data = this;
            _tid = (HANDLE)_beginthreadex(NULL, 0, win32ThreadFunc, args, 0, NULL);
            if(_tid == 0)
            {
                return false;
            }
#else
            //! Now create the thread with pointer to self as the data
            int retVal = pthread_create(&_tid, NULL, func, arg);
            if(retVal != 0)
            {
                return false;
            }
#endif
            return true;
        }

        /**
         * Wrapper for pthread_join. The calling thread
        * will wait until _this_ thread exits
        */
        bool join()
        {
            if(_tid)
            {
#ifdef _WIN32
                DWORD rc = WaitForSingleObject(_tid, INFINITE);
                CloseHandle(_tid);
                if(rc == WAIT_FAILED)
                {
                    printf("Bad call to function(invalid handle?)\n");
                }
#else
                int rc = pthread_join(_tid, NULL);
#endif
                _tid = 0;
                if(rc != 0)
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * Get the thread data passed by the application
        */
        void* getData()
        {
            return _data;
        }

        /**
         * Get the thread ID
        */
        unsigned int getID()
        {
#if defined(__MINGW32__) && defined(__MINGW64_VERSION_MAJOR)
            //This is to fix compilation issue with MinGW64-w64
            return (unsigned int)(long long)_tid;
#else
            return (unsigned long long)_tid;
#endif //__MINGW32__  and __MINGW64_VERSION_MAJOR
        }

    private:

        /**
         * Private data members and methods
         */

#ifdef _WIN32
        /**
         * store the handle
        */
        HANDLE _tid;
#else
        pthread_t _tid;
#endif

        void *_data;

};

#ifdef _WIN32
//! Windows thread callback - invokes the callback set by
//! the application in Thread constructor
unsigned _stdcall win32ThreadFunc(void* args)
{
    argsToThreadFunc* ptr = (argsToThreadFunc*) args;
    SDKThread *obj = (SDKThread *) ptr->data;
    ptr->func(obj->getData());
    delete args;
    return 0;
}
#endif

}

#endif // _CPU_THREAD_H_
