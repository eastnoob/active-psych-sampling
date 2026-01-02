/*
* Copyright 2009-2016  NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* This software and the information contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and conditions
* of a form of NVIDIA software license agreement.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/

/* This header defines types which are used by the internal implementation
*  of NVTX and callback subscribers.  API clients do not use these types,
*  so they are defined here instead of in nvToolsExt.h to clarify they are
*  not part of the NVTX client API. */

#ifndef NVTX_IMPL_GUARD
#error Never include this file directly -- it is automatically included by nvToolsExt.h.
#endif

/* ------ Dependency-free types binary-compatible with real types ------- */

/* In order to avoid having the NVTX core API headers depend on non-NVTX
*  headers like cuda.h, NVTX defines binary-compatible types to use for
*  safely making the initialization versions of all NVTX functions without
*  needing to have definitions for the real types. */

typedef int   nvtx_CUdevice;
typedef void* nvtx_CUcontext;
typedef void* nvtx_CUstream;
typedef void* nvtx_CUevent;

typedef void* nvtx_cudaStream_t;
typedef void* nvtx_cudaEvent_t;

typedef void* nvtx_cl_platform_id;
typedef void* nvtx_cl_device_id;
typedef void* nvtx_cl_context;
typedef void* nvtx_cl_command_queue;
typedef void* nvtx_cl_mem;
typedef void* nvtx_cl_program;
typedef void* nvtx_cl_kernel;
typedef void* nvtx_cl_event;
typedef void* nvtx_cl_sampler;

typedef struct nvtxSyncUser* nvtxSyncUser_t;
struct nvtxSyncUserAttributes_v0;
typedef struct nvtxSyncUserAttributes_v0 nvtxSyncUserAttributes_t;

/* --------- Types for function pointers (with fake API types) ---------- */

typedef void (NVTX_API * nvtxMarkEx_impl_fntype)(const nvtxEventAttributes_t* eventAttrib);
typedef void (NVTX_API * nvtxMarkA_impl_fntype)(const char* message);
typedef void (NVTX_API * nvtxMarkW_impl_fntype)(const wchar_t* message);
typedef nvtxRangeId_t (NVTX_API * nvtxRangeStartEx_impl_fntype)(const nvtxEventAttributes_t* eventAttrib);
typedef nvtxRangeId_t (NVTX_API * nvtxRangeStartA_impl_fntype)(const char* message);
typedef nvtxRangeId_t (NVTX_API * nvtxRangeStartW_impl_fntype)(const wchar_t* message);
typedef void (NVTX_API * nvtxRangeEnd_impl_fntype)(nvtxRangeId_t id);
typedef int (NVTX_API * nvtxRangePushEx_impl_fntype)(const nvtxEventAttributes_t* eventAttrib);
typedef int (NVTX_API * nvtxRangePushA_impl_fntype)(const char* message);
typedef int (NVTX_API * nvtxRangePushW_impl_fntype)(const wchar_t* message);
typedef int (NVTX_API * nvtxRangePop_impl_fntype)(void);
typedef void (NVTX_API * nvtxNameCategoryA_impl_fntype)(uint32_t category, const char* name);
typedef void (NVTX_API * nvtxNameCategoryW_impl_fntype)(uint32_t category, const wchar_t* name);
typedef void (NVTX_API * nvtxNameOsThreadA_impl_fntype)(uint32_t threadId, const char* name);
typedef void (NVTX_API * nvtxNameOsThreadW_impl_fntype)(uint32_t threadId, const wchar_t* name);

/* Real impl types are defined in nvtxImplCuda_v3.h, where CUDA headers are included */
typedef void (NVTX_API * nvtxNameCuDeviceA_fakeimpl_fntype)(nvtx_CUdevice device, const char* name);
typedef void (NVTX_API * nvtxNameCuDeviceW_fakeimpl_fntype)(nvtx_CUdevice device, const wchar_t* name);
typedef void (NVTX_API * nvtxNameCuContextA_fakeimpl_fntype)(nvtx_CUcontext context, const char* name);
typedef void (NVTX_API * nvtxNameCuContextW_fakeimpl_fntype)(nvtx_CUcontext context, const wchar_t* name);
typedef void (NVTX_API * nvtxNameCuStreamA_fakeimpl_fntype)(nvtx_CUstream stream, const char* name);
typedef void (NVTX_API * nvtxNameCuStreamW_fakeimpl_fntype)(nvtx_CUstream stream, const wchar_t* name);
typedef void (NVTX_API * nvtxNameCuEventA_fakeimpl_fntype)(nvtx_CUevent event, const char* name);
typedef void (NVTX_API * nvtxNameCuEventW_fakeimpl_fntype)(nvtx_CUevent event, const wchar_t* name);

/* Real impl types are defined in nvtxImplOpenCL_v3.h, where OPENCL headers are included */
typedef void (NVTX_API * nvtxNameClDeviceA_fakeimpl_fntype)(nvtx_cl_device_id device, const char* name);
typedef void (NVTX_API * nvtxNameClDeviceW_fakeimpl_fntype)(nvtx_cl_device_id device, const wchar_t* name);
typedef void (NVTX_API * nvtxNameClContextA_fakeimpl_fntype)(nvtx_cl_context context, const char* name);
typedef void (NVTX_API * nvtxNameClContextW_fakeimpl_fntype)(nvtx_cl_context context, const wchar_t* name);
typedef void (NVTX_API * nvtxNameClCommandQueueA_fakeimpl_fntype)(nvtx_cl_command_queue command_queue, const char* name);
typedef void (NVTX_API * nvtxNameClCommandQueueW_fakeimpl_fntype)(nvtx_cl_command_queue command_queue, const wchar_t* name);
typedef void (NVTX_API * nvtxNameClMemObjectA_fakeimpl_fntype)(nvtx_cl_mem memobj, const char* name);
typedef void (NVTX_API * nvtxNameClMemObjectW_fakeimpl_fntype)(nvtx_cl_mem memobj, const wchar_t* name);
typedef void (NVTX_API * nvtxNameClSamplerA_fakeimpl_fntype)(nvtx_cl_sampler sampler, const char* name);
typedef void (NVTX_API * nvtxNameClSamplerW_fakeimpl_fntype)(nvtx_cl_sampler sampler, const wchar_t* name);
typedef void (NVTX_API * nvtxNameClProgramA_fakeimpl_fntype)(nvtx_cl_program program, const char* name);
typedef void (NVTX_API * nvtxNameClProgramW_fakeimpl_fntype)(nvtx_cl_program program, const wchar_t* name);
typedef void (NVTX_API * nvtxNameClEventA_fakeimpl_fntype)(nvtx_cl_event evnt, const char* name);
typedef void (NVTX_API * nvtxNameClEventW_fakeimpl_fntype)(nvtx_cl_event evnt, const wchar_t* name);

/* Real impl types are defined in nvtxImplCudaRt_v3.h, where CUDART headers are included */
typedef void (NVTX_API * nvtxNameCudaDeviceA_impl_fntype)(int device, const char* name);
typedef void (NVTX_API * nvtxNameCudaDeviceW_impl_fntype)(int device, const wchar_t* name);
typedef void (NVTX_API * nvtxNameCudaStreamA_fakeimpl_fntype)(nvtx_cudaStream_t stream, const char* name);
typedef void (NVTX_API * nvtxNameCudaStreamW_fakeimpl_fntype)(nvtx_cudaStream_t stream, const wchar_t* name);
typedef void (NVTX_API * nvtxNameCudaEventA_fakeimpl_fntype)(nvtx_cudaEvent_t event, const char* name);
typedef void (NVTX_API * nvtxNameCudaEventW_fakeimpl_fntype)(nvtx_cudaEvent_t event, const wchar_t* name);

typedef void (NVTX_API * nvtxDomainMarkEx_impl_fntype)(nvtxDomainHandle_t domain, const nvtxEventAttributes_t* eventAttrib);
typedef nvtxRangeId_t (NVTX_API * nvtxDomainRangeStartEx_impl_fntype)(nvtxDomainHandle_t domain, const nvtxEventAttributes_t* eventAttrib);
typedef void (NVTX_API * nvtxDomainRangeEnd_impl_fntype)(nvtxDomainHandle_t domain, nvtxRangeId_t id);
typedef int (NVTX_API * nvtxDomainRangePushEx_impl_fntype)(nvtxDomainHandle_t domain, const nvtxEventAttributes_t* eventAttrib);
typedef int (NVTX_API * nvtxDomainRangePop_impl_fntype)(nvtxDomainHandle_t domain);
typedef nvtxResourceHandle_t (NVTX_API * nvtxDomainResourceCreate_impl_fntype)(nvtxDomainHandle_t domain, nvtxResourceAttributes_t* attribs);
typedef void (NVTX_API * nvtxDomainResourceDestroy_impl_fntype)(nvtxResourceHandle_t resource);
typedef void (NVTX_API * nvtxDomainNameCategoryA_impl_fntype)(nvtxDomainHandle_t domain, uint32_t category, const char* name);
typedef void (NVTX_API * nvtxDomainNameCategoryW_impl_fntype)(nvtxDomainHandle_t domain, uint32_t category, const wchar_t* name);
typedef nvtxStringHandle_t (NVTX_API * nvtxDomainRegisterStringA_impl_fntype)(nvtxDomainHandle_t domain, const char* string);
typedef nvtxStringHandle_t (NVTX_API * nvtxDomainRegisterStringW_impl_fntype)(nvtxDomainHandle_t domain, const wchar_t* string);
typedef nvtxDomainHandle_t (NVTX_API * nvtxDomainCreateA_impl_fntype)(const char* message);
typedef nvtxDomainHandle_t (NVTX_API * nvtxDomainCreateW_impl_fntype)(const wchar_t* message);
typedef void (NVTX_API * nvtxDomainDestroy_impl_fntype)(nvtxDomainHandle_t domain);
typedef void (NVTX_API * nvtxInitialize_impl_fntype)(const void* reserved);

typedef nvtxSyncUser_t (NVTX_API * nvtxDomainSyncUserCreate_impl_fntype)(nvtxDomainHandle_t domain, const nvtxSyncUserAttributes_t* attribs);
typedef void (NVTX_API * nvtxDomainSyncUserDestroy_impl_fntype)(nvtxSyncUser_t handle);
typedef void (NVTX_API * nvtxDomainSyncUserAcquireStart_impl_fntype)(nvtxSyncUser_t handle);
typedef void (NVTX_API * nvtxDomainSyncUserAcquireFailed_impl_fntype)(nvtxSyncUser_t handle);
typedef void (NVTX_API * nvtxDomainSyncUserAcquireSuccess_impl_fntype)(nvtxSyncUser_t handle);
typedef void (NVTX_API * nvtxDomainSyncUserReleasing_impl_fntype)(nvtxSyncUser_t handle);

/* ---------------- Types for callback subscription --------------------- */

typedef const void *(NVTX_API * NvtxGetExportTableFunc_t)(uint32_t exportTableId);
typedef int (NVTX_API * NvtxInitializeInjectionNvtxFunc_t)(NvtxGetExportTableFunc_t exportTable);

typedef enum NvtxCallbackModule
{
    NVTX_CB_MODULE_INVALID                 = 0,
    NVTX_CB_MODULE_CORE                    = 1,
    NVTX_CB_MODULE_CUDA                    = 2,
    NVTX_CB_MODULE_OPENCL                  = 3,
    NVTX_CB_MODULE_CUDART                  = 4,
    NVTX_CB_MODULE_CORE2                   = 5,
    NVTX_CB_MODULE_SYNC                    = 6,
    /* --- New constants must only be added directly above this line --- */
    NVTX_CB_MODULE_SIZE,
    NVTX_CB_MODULE_FORCE_INT               = 0x7fffffff
} NvtxCallbackModule;

typedef enum NvtxCallbackIdCore
{
    NVTX_CBID_CORE_INVALID                 =  0,
    NVTX_CBID_CORE_MarkEx                  =  1,
    NVTX_CBID_CORE_MarkA                   =  2,
    NVTX_CBID_CORE_MarkW                   =  3,
    NVTX_CBID_CORE_RangeStartEx            =  4,
    NVTX_CBID_CORE_RangeStartA             =  5,
    NVTX_CBID_CORE_RangeStartW             =  6,
    NVTX_CBID_CORE_RangeEnd                =  7,
    NVTX_