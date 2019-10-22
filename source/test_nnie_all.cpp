/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2019, Open AI Lab
 * Author: cmeng@openailab.com
 */
#include <unistd.h>
#include <sys/stat.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <math.h>
#include "mpi_sys.h"
#include "mpi_vb.h"
#include <sys/time.h>
#include "tengine_c_api.h"
#include "tengine_nnie_plugin.h"
#include "test_nnie_all.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

#define DEFAULT_REPEAT_CNT 1
static int repeat_count = DEFAULT_REPEAT_CNT;

/*
*Malloc memory with cached
*/
HI_S32 TEST_COMM_MallocCached(const HI_CHAR *pszMmb, HI_CHAR *pszZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size)
{
    HI_S32 s32Ret = HI_SUCCESS;
    s32Ret = HI_MPI_SYS_MmzAlloc_Cached(pu64PhyAddr, ppvVirAddr, pszMmb, pszZone, u32Size);

    return s32Ret;
}

/*
*Fulsh cached
*/
HI_S32 TEST_COMM_FlushCache(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr, HI_U32 u32Size)
{
    HI_S32 s32Ret = HI_SUCCESS;
    s32Ret = HI_MPI_SYS_MmzFlushCache(u64PhyAddr, pvVirAddr, u32Size);
    return s32Ret;
}

void get_nnie_input_data(std::string &image_file, float *input_data, int img_h, int img_w)
{
    cv::Mat img = cv::imread(image_file, -1);
    if (img.empty())
    {
        std::cerr << "failed to read image file " << image_file << "\n";
        return;
    }
    cv::resize(img, img, cv::Size(img_h, img_w));
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3);
    // img_float = (img_float - 127.5) / 128;
    float *img_data = (float *)img.data;
    int hw = img_h * img_w;
    // float mean[3]={104.f,117.f,123.f};
    //float mean[3] = {127.5, 127.5, 127.5};
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                //input_data[c * hw + h * img_w + w] = (*img_data - mean[c]);
                input_data[c * hw + h * img_w + w] = (*img_data);
                img_data++;
            }
        }
    }
}
void get_input_data_from_org(const char *image_file, float *input_data, int img_h, int img_w, const float *mean, float scale)
{
    const string imagefile(image_file);
    cv::Mat img = cv::imread(imagefile);
    if (img.empty())
    {
        std::cerr << "Failed to read image file " << image_file << ".\n";
        return;
    }
    cv::resize(img, img, cv::Size(img_h, img_w));

    if (img.channels() == 4)
    {
        cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
    }
    else if (img.channels() == 1)
    {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }

    img.convertTo(img, CV_32FC3);
    float *img_data = (float *)img.data;
    int hw = img_h * img_w;
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale;
                img_data++;
            }
        }
    }
}

bool get_input_data(const char *image_file, void *input_data, int input_length)
{
    FILE *fp = fopen(image_file, "rb");
    if (fp == nullptr)
    {
        std::cout << "Open input data file failed: " << image_file << "\n";
        return false;
    }

    int res = fread(input_data, 1, input_length, fp);
    if (res != input_length)
    {
        std::cout << "Read input data file failed: " << image_file << "\n";
        return false;
    }
    fclose(fp);
    return true;
}

bool write_output_data(const char *file_name, void *output_data, int output_length)
{
    FILE *fp = fopen(file_name, "wb");
    if (fp == nullptr)
    {
        std::cout << "Open output data file failed: " << file_name << "\n";
        return false;
    }

    int res = fwrite(output_data, 1, output_length, fp);
    if (res != output_length)
    {
        std::cout << "Write output data file failed: " << file_name << "\n";
        return false;
    }
    fflush(fp);
    fclose(fp);
    return true;
}

typedef unsigned long long HI_U64;

/******************************************************************************
* function : show usage
******************************************************************************/
void TEST_Usage(char *pchPrgName)
{
    printf("Usage : %s <index> \n", pchPrgName);
    printf("index:\n");
    printf("\t 2) FasterRcnn(Read File).\n");
    printf("\t 3) Cnn(Read File).\n");
    printf("\t 4) SSD(Read File).\n");
    printf("\t 5) Yolov1(Read File).\n");
    printf("\t 6) Yolov2(Read File).\n");
    printf("\t 7) Yolov3(Read File).\n");
    printf("\t 8) LSTM(Read File).\n");
    printf("\t 9) Pvanet(Read File).\n");
}

HI_U32 TEST_NNIE_RpnTmpBufSize(HI_U32 u32NumRatioAnchors,
                               HI_U32 u32NumScaleAnchors, HI_U32 u32ConvHeight, HI_U32 u32ConvWidth)
{
    HI_U32 u32AnchorsNum = u32NumRatioAnchors * u32NumScaleAnchors * u32ConvHeight * u32ConvWidth;
    HI_U32 u32AnchorsSize = sizeof(HI_U32) * TEST_NNIE_COORDI_NUM * u32AnchorsNum;
    HI_U32 u32BboxDeltaSize = u32AnchorsSize;
    HI_U32 u32ProposalSize = sizeof(HI_U32) * TEST_NNIE_PROPOSAL_WIDTH * u32AnchorsNum;
    HI_U32 u32RatioAnchorsSize = sizeof(HI_FLOAT) * u32NumRatioAnchors * TEST_NNIE_COORDI_NUM;
    HI_U32 u32ScaleAnchorsSize = sizeof(HI_FLOAT) * u32NumRatioAnchors * u32NumScaleAnchors * TEST_NNIE_COORDI_NUM;
    HI_U32 u32ScoreSize = sizeof(HI_FLOAT) * u32AnchorsNum * 2;
    HI_U32 u32StackSize = sizeof(TEST_NNIE_STACK_S) * u32AnchorsNum;
    HI_U32 u32TotalSize = u32AnchorsSize + u32BboxDeltaSize + u32ProposalSize + u32RatioAnchorsSize + u32ScaleAnchorsSize + u32ScoreSize + u32StackSize;
    return u32TotalSize;
}

HI_U32 TEST_NNIE_Rfcn_GetResultTmpBuf(HI_U32 u32MaxRoiNum, HI_U32 u32ClassNum)
{
    HI_U32 u32ScoreSize = sizeof(HI_FLOAT) * u32MaxRoiNum * u32ClassNum;
    HI_U32 u32ProposalSize = sizeof(HI_U32) * u32MaxRoiNum * TEST_NNIE_PROPOSAL_WIDTH;
    HI_U32 u32BboxSize = sizeof(HI_U32) * u32MaxRoiNum * TEST_NNIE_COORDI_NUM;
    HI_U32 u32StackSize = sizeof(TEST_NNIE_STACK_S) * u32MaxRoiNum;
    HI_U32 u32TotalSize = u32ScoreSize + u32ProposalSize + u32BboxSize + u32StackSize;
    return u32TotalSize;
}

static HI_FLOAT s_af32ExpCoef[10][16] = {
    {1.0f, 1.00024f, 1.00049f, 1.00073f, 1.00098f, 1.00122f, 1.00147f, 1.00171f, 1.00196f, 1.0022f, 1.00244f, 1.00269f, 1.00293f, 1.00318f, 1.00342f, 1.00367f},
    {1.0f, 1.00391f, 1.00784f, 1.01179f, 1.01575f, 1.01972f, 1.02371f, 1.02772f, 1.03174f, 1.03578f, 1.03984f, 1.04391f, 1.04799f, 1.05209f, 1.05621f, 1.06034f},
    {1.0f, 1.06449f, 1.13315f, 1.20623f, 1.28403f, 1.36684f, 1.45499f, 1.54883f, 1.64872f, 1.75505f, 1.86825f, 1.98874f, 2.117f, 2.25353f, 2.39888f, 2.55359f},
    {1.0f, 2.71828f, 7.38906f, 20.0855f, 54.5981f, 148.413f, 403.429f, 1096.63f, 2980.96f, 8103.08f, 22026.5f, 59874.1f, 162755.0f, 442413.0f, 1.2026e+006f, 3.26902e+006f},
    {1.0f, 8.88611e+006f, 7.8963e+013f, 7.01674e+020f, 6.23515e+027f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f},
    {1.0f, 0.999756f, 0.999512f, 0.999268f, 0.999024f, 0.99878f, 0.998536f, 0.998292f, 0.998049f, 0.997805f, 0.997562f, 0.997318f, 0.997075f, 0.996831f, 0.996588f, 0.996345f},
    {1.0f, 0.996101f, 0.992218f, 0.98835f, 0.984496f, 0.980658f, 0.976835f, 0.973027f, 0.969233f, 0.965455f, 0.961691f, 0.957941f, 0.954207f, 0.950487f, 0.946781f, 0.94309f},
    {1.0f, 0.939413f, 0.882497f, 0.829029f, 0.778801f, 0.731616f, 0.687289f, 0.645649f, 0.606531f, 0.569783f, 0.535261f, 0.502832f, 0.472367f, 0.443747f, 0.416862f, 0.391606f},
    {1.0f, 0.367879f, 0.135335f, 0.0497871f, 0.0183156f, 0.00673795f, 0.00247875f, 0.000911882f, 0.000335463f, 0.00012341f, 4.53999e-005f, 1.67017e-005f, 6.14421e-006f, 2.26033e-006f, 8.31529e-007f, 3.05902e-007f},
    {1.0f, 1.12535e-007f, 1.26642e-014f, 1.42516e-021f, 1.60381e-028f, 1.80485e-035f, 2.03048e-042f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

static HI_FLOAT TEST_NNIE_QuickExp(HI_S32 s32Value)
{
    if (s32Value & 0x80000000)
    {
        s32Value = ~s32Value + 0x00000001;
        return s_af32ExpCoef[5][s32Value & 0x0000000F] * s_af32ExpCoef[6][(s32Value >> 4) & 0x0000000F] * s_af32ExpCoef[7][(s32Value >> 8) & 0x0000000F] * s_af32ExpCoef[8][(s32Value >> 12) & 0x0000000F] * s_af32ExpCoef[9][(s32Value >> 16) & 0x0000000F];
    }
    else
    {
        return s_af32ExpCoef[0][s32Value & 0x0000000F] * s_af32ExpCoef[1][(s32Value >> 4) & 0x0000000F] * s_af32ExpCoef[2][(s32Value >> 8) & 0x0000000F] * s_af32ExpCoef[3][(s32Value >> 12) & 0x0000000F] * s_af32ExpCoef[4][(s32Value >> 16) & 0x0000000F];
    }
}

static void TEST_NNIE_Argswap(HI_S32 *ps32Src1, HI_S32 *ps32Src2)
{
    HI_U32 i = 0;
    HI_S32 u32Tmp = 0;
    for (i = 0; i < TEST_NNIE_PROPOSAL_WIDTH; i++)
    {
        u32Tmp = ps32Src1[i];
        ps32Src1[i] = ps32Src2[i];
        ps32Src2[i] = u32Tmp;
    }
}

static HI_S32 TEST_NNIE_NonRecursiveArgQuickSort(HI_S32 *ps32Array,
                                                 HI_S32 s32Low, HI_S32 s32High, TEST_NNIE_STACK_S *pstStack, HI_U32 u32MaxNum)
{
    HI_S32 i = s32Low;
    HI_S32 j = s32High;
    HI_S32 s32Top = 0;
    HI_S32 s32KeyConfidence = ps32Array[TEST_NNIE_PROPOSAL_WIDTH * s32Low + 4];
    pstStack[s32Top].s32Min = s32Low;
    pstStack[s32Top].s32Max = s32High;

    while (s32Top > -1)
    {
        s32Low = pstStack[s32Top].s32Min;
        s32High = pstStack[s32Top].s32Max;
        i = s32Low;
        j = s32High;
        s32Top--;

        s32KeyConfidence = ps32Array[TEST_NNIE_PROPOSAL_WIDTH * s32Low + 4];

        while (i < j)
        {
            while ((i < j) && (s32KeyConfidence > ps32Array[j * TEST_NNIE_PROPOSAL_WIDTH + 4]))
            {
                j--;
            }
            if (i < j)
            {
                TEST_NNIE_Argswap(&ps32Array[i * TEST_NNIE_PROPOSAL_WIDTH], &ps32Array[j * TEST_NNIE_PROPOSAL_WIDTH]);
                i++;
            }

            while ((i < j) && (s32KeyConfidence < ps32Array[i * TEST_NNIE_PROPOSAL_WIDTH + 4]))
            {
                i++;
            }
            if (i < j)
            {
                TEST_NNIE_Argswap(&ps32Array[i * TEST_NNIE_PROPOSAL_WIDTH], &ps32Array[j * TEST_NNIE_PROPOSAL_WIDTH]);
                j--;
            }
        }

        if (s32Low <= (HI_S32)u32MaxNum)
        {
            if (s32Low < i - 1)
            {
                s32Top++;
                pstStack[s32Top].s32Min = s32Low;
                pstStack[s32Top].s32Max = i - 1;
            }

            if (s32High > i + 1)
            {
                s32Top++;
                pstStack[s32Top].s32Min = i + 1;
                pstStack[s32Top].s32Max = s32High;
            }
        }
    }
    return HI_SUCCESS;
}

static HI_S32 TEST_NNIE_Overlap(HI_S32 s32XMin1, HI_S32 s32YMin1, HI_S32 s32XMax1, HI_S32 s32YMax1, HI_S32 s32XMin2,
                                HI_S32 s32YMin2, HI_S32 s32XMax2, HI_S32 s32YMax2, HI_S32 *s32AreaSum, HI_S32 *s32AreaInter)
{
    /*** Check the input, and change the Return value  ***/
    HI_S32 s32Inter = 0;
    HI_S32 s32Total = 0;
    HI_S32 s32XMin = 0;
    HI_S32 s32YMin = 0;
    HI_S32 s32XMax = 0;
    HI_S32 s32YMax = 0;
    HI_S32 s32Area1 = 0;
    HI_S32 s32Area2 = 0;
    HI_S32 s32InterWidth = 0;
    HI_S32 s32InterHeight = 0;

    s32XMin = TEST_NNIE_MAX(s32XMin1, s32XMin2);
    s32YMin = TEST_NNIE_MAX(s32YMin1, s32YMin2);
    s32XMax = TEST_NNIE_MIN(s32XMax1, s32XMax2);
    s32YMax = TEST_NNIE_MIN(s32YMax1, s32YMax2);

    s32InterWidth = s32XMax - s32XMin + 1;
    s32InterHeight = s32YMax - s32YMin + 1;

    s32InterWidth = (s32InterWidth >= 0) ? s32InterWidth : 0;
    s32InterHeight = (s32InterHeight >= 0) ? s32InterHeight : 0;

    s32Inter = s32InterWidth * s32InterHeight;
    s32Area1 = (s32XMax1 - s32XMin1 + 1) * (s32YMax1 - s32YMin1 + 1);
    s32Area2 = (s32XMax2 - s32XMin2 + 1) * (s32YMax2 - s32YMin2 + 1);

    s32Total = s32Area1 + s32Area2 - s32Inter;

    *s32AreaSum = s32Total;
    *s32AreaInter = s32Inter;
    return HI_SUCCESS;
}

static HI_S32 TEST_NNIE_NonMaxSuppression(HI_S32 *ps32Proposals, HI_U32 u32AnchorsNum,
                                          HI_U32 u32NmsThresh, HI_U32 u32MaxRoiNum)
{
    /****** define variables *******/
    HI_S32 s32XMin1 = 0;
    HI_S32 s32YMin1 = 0;
    HI_S32 s32XMax1 = 0;
    HI_S32 s32YMax1 = 0;
    HI_S32 s32XMin2 = 0;
    HI_S32 s32YMin2 = 0;
    HI_S32 s32XMax2 = 0;
    HI_S32 s32YMax2 = 0;
    HI_S32 s32AreaTotal = 0;
    HI_S32 s32AreaInter = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 u32Num = 0;
    HI_BOOL bNoOverlap = HI_TRUE;

    for (i = 0; i < u32AnchorsNum && u32Num < u32MaxRoiNum; i++)
    {
        if (ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 5] == 0)
        {
            u32Num++;
            s32XMin1 = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i];
            s32YMin1 = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 1];
            s32XMax1 = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 2];
            s32YMax1 = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 3];
            for (j = i + 1; j < u32AnchorsNum; j++)
            {
                if (ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * j + 5] == 0)
                {
                    s32XMin2 = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * j];
                    s32YMin2 = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * j + 1];
                    s32XMax2 = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * j + 2];
                    s32YMax2 = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * j + 3];
                    bNoOverlap = (HI_BOOL)((s32XMin2 > s32XMax1) || (s32XMax2 < s32XMin1) || (s32YMin2 > s32YMax1) || (s32YMax2 < s32YMin1));
                    if (bNoOverlap)
                    {
                        continue;
                    }
                    (void)TEST_NNIE_Overlap(s32XMin1, s32YMin1, s32XMax1, s32YMax1, s32XMin2, s32YMin2, s32XMax2, s32YMax2, &s32AreaTotal, &s32AreaInter);
                    if (s32AreaInter * TEST_NNIE_QUANT_BASE > ((HI_S32)u32NmsThresh * s32AreaTotal))
                    {
                        if (ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 4] >= ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * j + 4])
                        {
                            ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * j + 5] = 1;
                        }
                        else
                        {
                            ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 5] = 1;
                        }
                    }
                }
            }
        }
    }
    return HI_SUCCESS;
}

static HI_S32 TEST_NNIE_FasterRcnn_SoftwareInit(graph_t graph, TEST_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_U32 i = 0;
    //HI_U32 j = 0;
    HI_U32 u32RpnTmpBufSize = 0;
    HI_U32 u32RpnBboxBufSize = 0;
    HI_U32 u32GetResultTmpBufSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;
    HI_U32 u32MaxRoiNum = 300;

    /*RPN parameter init*/
    pstSoftWareParam->u32MaxRoiNum = u32MaxRoiNum;
    // if(TEST_NNIE_VGG16_FASTER_RCNN == s_enNetType)
    // {
    //     pstSoftWareParam->u32ClassNum = 4;
    //     pstSoftWareParam->u32NumRatioAnchors = 3;
    //     pstSoftWareParam->u32NumScaleAnchors = 3;
    //     pstSoftWareParam->au32Scales[0] = 8 * TEST_NNIE_QUANT_BASE;
    //     pstSoftWareParam->au32Scales[1] = 16 * TEST_NNIE_QUANT_BASE;
    //     pstSoftWareParam->au32Scales[2] = 32 * TEST_NNIE_QUANT_BASE;
    //     pstSoftWareParam->au32Ratios[0] = 0.5 * TEST_NNIE_QUANT_BASE;
    //     pstSoftWareParam->au32Ratios[1] = 1 * TEST_NNIE_QUANT_BASE;
    //     pstSoftWareParam->au32Ratios[2] = 2 * TEST_NNIE_QUANT_BASE;
    // }
    // else
    {
        pstSoftWareParam->u32ClassNum = 2;
        pstSoftWareParam->u32NumRatioAnchors = 1;
        pstSoftWareParam->u32NumScaleAnchors = 9;
        pstSoftWareParam->au32Scales[0] = 1.5 * TEST_NNIE_QUANT_BASE;
        pstSoftWareParam->au32Scales[1] = 2.1 * TEST_NNIE_QUANT_BASE;
        pstSoftWareParam->au32Scales[2] = 2.9 * TEST_NNIE_QUANT_BASE;
        pstSoftWareParam->au32Scales[3] = 4.1 * TEST_NNIE_QUANT_BASE;
        pstSoftWareParam->au32Scales[4] = 5.8 * TEST_NNIE_QUANT_BASE;
        pstSoftWareParam->au32Scales[5] = 8.0 * TEST_NNIE_QUANT_BASE;
        pstSoftWareParam->au32Scales[6] = 11.3 * TEST_NNIE_QUANT_BASE;
        pstSoftWareParam->au32Scales[7] = 15.8 * TEST_NNIE_QUANT_BASE;
        pstSoftWareParam->au32Scales[8] = 22.1 * TEST_NNIE_QUANT_BASE;
        pstSoftWareParam->au32Ratios[0] = 2.44 * TEST_NNIE_QUANT_BASE;
    }

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    int dims[4]; //NCHW
    int dimssize = 4;
    get_tensor_shape(input_tensor, dims, dimssize); //NCHW
    printf("input tensor dims[%d:%d:%d:%d]\n", dims[0], dims[1], dims[2], dims[3]);

    pstSoftWareParam->u32OriImHeight = dims[2]; //pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = dims[3];  //pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32MinSize = 16;
    pstSoftWareParam->u32FilterThresh = 16;
    pstSoftWareParam->u32SpatialScale = (HI_U32)(0.0625 * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.7 * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->u32FilterThresh = 0;
    pstSoftWareParam->u32NumBeforeNms = 6000;
    for (i = 0; i < pstSoftWareParam->u32ClassNum; i++)
    {
        pstSoftWareParam->au32ConfThresh[i] = 1;
    }
    pstSoftWareParam->u32ValidNmsThresh = (HI_U32)(0.3 * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->stRpnBbox.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stRpnBbox.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stRpnBbox.unShape.stWhc.u32Height = u32MaxRoiNum;
    pstSoftWareParam->stRpnBbox.unShape.stWhc.u32Width = TEST_COORDI_NUM;
    pstSoftWareParam->stRpnBbox.u32Stride = TEST_NNIE_ALIGN16(TEST_COORDI_NUM * sizeof(HI_U32));
    pstSoftWareParam->stRpnBbox.u32Num = 1;

    // /*set rpn input data info, the input info is set according to RPN data layers' name*/
    // for(i = 0; i < 2; i++)
    // {
    //     node_t node = get_graph_node_by_idx(graph, 1);
    //     int outputCount = get_node_output_number(node);
    //     printf("[%s][%d]outputCount:%d\n",__FUNCTION__,__LINE__,outputCount);
    //     for(j = 0; j < outputCount; j++)
    //     {
    //         tensor_t tensor = get_node_output_tensor(node,j);
    //         const char * tensorname = get_tensor_name(tensor);
    //         string::size_type idx;
    //         std::string a = std::string(tensorname);
    //         std::string b = std::string(pstSoftWareParam->apcRpnDataLayerName[i]);
    //         // if(0 == strncmp(pstNnieParam->pstModel->astSeg[0].astDstNode[j].szName,
    //         //         pstSoftWareParam->apcRpnDataLayerName[i],
    //         //         SVP_NNIE_NODE_NAME_LEN))
    //         idx = a.find(b);
    //         if(idx != std::string::npos)
    //         {
    //             pstSoftWareParam->aps32Conv[i] =(HI_S32*)pstNnieParam->astSegData[0].astDst[j].u64VirAddr;
    //             pstSoftWareParam->au32ConvHeight[i] = pstNnieParam->pstModel->astSeg[0].astDstNode[j].unShape.stWhc.u32Height;
    //             pstSoftWareParam->au32ConvWidth[i] = pstNnieParam->pstModel->astSeg[0].astDstNode[j].unShape.stWhc.u32Width;
    //             pstSoftWareParam->au32ConvChannel[i] = pstNnieParam->pstModel->astSeg[0].astDstNode[j].unShape.stWhc.u32Chn;
    //             break;
    //         }
    //     }
    //     if(j == pstNnieParam->pstModel->astSeg[0].u16DstNum){
    //         printf("Error,failed to find report node %s!\n", pstSoftWareParam->apcRpnDataLayerName[i]);
    //     }
    //     if(0 == i)
    //     {
    //         pstSoftWareParam->u32ConvStride = pstNnieParam->astSegData[0].astDst[j].u32Stride;
    //     }
    // }

    /*malloc software mem*/
    // u32RpnTmpBufSize = TEST_NNIE_RpnTmpBufSize(pstSoftWareParam->u32NumRatioAnchors,
    //     pstSoftWareParam->u32NumScaleAnchors,pstSoftWareParam->au32ConvHeight[0],
    //     pstSoftWareParam->au32ConvWidth[0]);
    u32RpnTmpBufSize = TEST_NNIE_RpnTmpBufSize(pstSoftWareParam->u32NumRatioAnchors,
                                               pstSoftWareParam->u32NumScaleAnchors,
                                               23,
                                               77);
    u32RpnTmpBufSize = TEST_NNIE_ALIGN16(u32RpnTmpBufSize);
    u32RpnBboxBufSize = pstSoftWareParam->stRpnBbox.u32Num *
                        pstSoftWareParam->stRpnBbox.unShape.stWhc.u32Height * pstSoftWareParam->stRpnBbox.u32Stride;
    u32GetResultTmpBufSize = TEST_NNIE_Rfcn_GetResultTmpBuf(u32MaxRoiNum, pstSoftWareParam->u32ClassNum);
    u32GetResultTmpBufSize = TEST_NNIE_ALIGN16(u32GetResultTmpBufSize);
    u32ClassNum = pstSoftWareParam->u32ClassNum;
    u32DstRoiSize = TEST_NNIE_ALIGN16(u32ClassNum * u32MaxRoiNum * sizeof(HI_U32) * TEST_NNIE_COORDI_NUM);
    u32DstScoreSize = TEST_NNIE_ALIGN16(u32ClassNum * u32MaxRoiNum * sizeof(HI_U32));
    u32ClassRoiNumSize = TEST_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32RpnTmpBufSize + u32RpnBboxBufSize + u32GetResultTmpBufSize + u32DstRoiSize +
                   u32DstScoreSize + u32ClassRoiNumSize;

    s32Ret = TEST_COMM_MallocCached("SAMPLE_RFCN_INIT", NULL, (HI_U64 *)&u64PhyAddr,
                                    (void **)&pu8VirAddr, u32TotalSize);
    if (HI_SUCCESS != s32Ret)
        printf("Error,Malloc memory failed!\n");
    memset(pu8VirAddr, 0, u32TotalSize);
    TEST_COMM_FlushCache(u64PhyAddr, (void *)pu8VirAddr, u32TotalSize);

    pstSoftWareParam->stRpnTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stRpnTmpBuf.u64VirAddr = (HI_U64)(pu8VirAddr);
    pstSoftWareParam->stRpnTmpBuf.u32Size = u32RpnTmpBufSize;

    // pstSoftWareParam->stRpnBbox.u64PhyAddr = u64PhyAddr+u32RpnTmpBufSize;
    // pstSoftWareParam->stRpnBbox.u64VirAddr = (HI_U64)(pu8VirAddr)+u32RpnTmpBufSize;

    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr + u32RpnTmpBufSize + u32RpnBboxBufSize;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)(pu8VirAddr + u32RpnTmpBufSize + u32RpnBboxBufSize);
    pstSoftWareParam->stGetResultTmpBuf.u32Size = u32GetResultTmpBufSize;

    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32RpnTmpBufSize + u32RpnBboxBufSize + u32GetResultTmpBufSize;
    pstSoftWareParam->stDstRoi.u64VirAddr = (HI_U64)(pu8VirAddr + u32RpnTmpBufSize + u32RpnBboxBufSize + u32GetResultTmpBufSize);
    pstSoftWareParam->stDstRoi.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32) * TEST_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width = u32ClassNum * pstSoftWareParam->u32MaxRoiNum * TEST_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32RpnTmpBufSize + u32RpnBboxBufSize + u32GetResultTmpBufSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr = (HI_U64)(pu8VirAddr + u32RpnTmpBufSize + u32RpnBboxBufSize + u32GetResultTmpBufSize + u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum * pstSoftWareParam->u32MaxRoiNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32RpnTmpBufSize + u32RpnBboxBufSize + u32GetResultTmpBufSize + u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr = (HI_U64)(pu8VirAddr + u32RpnTmpBufSize + u32RpnBboxBufSize + u32GetResultTmpBufSize + u32DstRoiSize + u32DstScoreSize);
    pstSoftWareParam->stClassRoiNum.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;
    return s32Ret;
}

static HI_S32 TEST_NNIE_FasterRcnn_GetResult2(HI_S32 *ps32FcBbox, HI_U32 u32BboxStride,
                                              HI_S32 *ps32FcScore, HI_U32 u32ScoreStride, HI_S32 *ps32Proposal, HI_U32 u32RoiCnt,
                                              HI_U32 *pu32ConfThresh, HI_U32 u32NmsThresh, HI_U32 u32MaxRoi, HI_U32 u32ClassNum,
                                              HI_U32 u32OriImWidth, HI_U32 u32OriImHeight, HI_U32 *pu32MemPool, HI_S32 *ps32DstScore,
                                              HI_S32 *ps32DstBbox, HI_S32 *ps32ClassRoiNum)
{
    // printf("[%s][%d] %p %d %p %d %p %d %p u32NmsThresh:%d %d %d %d %d %p %p %p %p\n", __FUNCTION__, __LINE__, ps32FcBbox, u32BboxStride, ps32FcScore, u32ScoreStride,
    //     ps32Proposal, u32RoiCnt, pu32ConfThresh, u32NmsThresh, u32MaxRoi, u32ClassNum, u32OriImWidth, u32OriImHeight, pu32MemPool,
    //     ps32DstScore, ps32DstBbox, ps32ClassRoiNum);
    /************* define variables *****************/
    HI_U32 u32Size = 0;
    HI_U32 u32ClsScoreChannels = 0;
    HI_S32 *ps32Proposals = NULL;
    HI_FLOAT f32ProposalWidth = 0.0;
    HI_FLOAT f32ProposalHeight = 0.0;
    HI_FLOAT f32ProposalCenterX = 0.0;
    HI_FLOAT f32ProposalCenterY = 0.0;
    HI_FLOAT f32PredW = 0.0;
    HI_FLOAT f32PredH = 0.0;
    HI_FLOAT f32PredCenterX = 0.0;
    HI_FLOAT f32PredCenterY = 0.0;
    HI_FLOAT *pf32FcScoresMemPool = NULL;
    HI_S32 *ps32ProposalMemPool = NULL;
    HI_S32 *ps32ProposalTmp = NULL;
    HI_U32 u32FcBboxIndex = 0;
    HI_U32 u32ProposalMemPoolIndex = 0;
    HI_FLOAT *pf32Ptr = NULL;
    HI_S32 *ps32Ptr = NULL;
    HI_S32 *ps32Score = NULL;
    HI_S32 *ps32Bbox = NULL;
    HI_S32 *ps32RoiCnt = NULL;
    HI_U32 u32RoiOutCnt = 0;
    HI_U32 u32SrcIndex = 0;
    HI_U32 u32DstIndex = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 k = 0;
    TEST_NNIE_STACK_S *pstStack = NULL;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32OffSet = 0;

    /* {
        unsigned int i = 0;
        unsigned int *pu32Tmp = NULL;
        unsigned int u32TopN = 10;

        printf("==== The 1 tensor info====\n");
        pu32Tmp = (unsigned int *)((HI_U64)ps32FcBbox);
        for (i = 0; i < u32TopN; i++)
        {
            printf("%d:%d\n", i, pu32Tmp[i]);
        }
    }

    {
        unsigned int i = 0;
        unsigned int *pu32Tmp = NULL;
        unsigned int u32TopN = 10;

        printf("==== The 2 tensor info====\n");
        pu32Tmp = (unsigned int *)((HI_U64)ps32FcScore);
        for (i = 0; i < u32TopN; i++)
        {
            printf("%d:%d\n", i, pu32Tmp[i]);
        }
    } */

    /******************* Get or calculate parameters **********************/
    u32ClsScoreChannels = u32ClassNum; /*channel num is equal to class size, cls_score class*/
    /*************** Get Start Pointer of MemPool ******************/
    pf32FcScoresMemPool = (HI_FLOAT *)pu32MemPool;
    pf32Ptr = pf32FcScoresMemPool;
    u32Size = u32MaxRoi * u32ClsScoreChannels;
    printf("[%s][%d] u32MaxRoi:%d \n", __FUNCTION__, __LINE__, u32MaxRoi);
    pf32Ptr += u32Size;

    ps32ProposalMemPool = (HI_S32 *)pf32Ptr;
    ps32Ptr = ps32ProposalMemPool;
    u32Size = u32MaxRoi * TEST_NNIE_PROPOSAL_WIDTH;
    ps32Ptr += u32Size;
    pstStack = (TEST_NNIE_STACK_S *)ps32Ptr;

    u32DstIndex = 0;

    for (i = 0; i < u32RoiCnt; i++)
    {
        for (k = 0; k < u32ClsScoreChannels; k++)
        {
            u32SrcIndex = i * u32ClsScoreChannels + k;
            pf32FcScoresMemPool[u32DstIndex++] = (HI_FLOAT)((HI_S32)ps32FcScore[u32SrcIndex]) / TEST_NNIE_QUANT_BASE;
            // printf("[%s][%d] pf32FcScoresMemPool[%d]:%f u32SrcIndex:%d\n",__FUNCTION__, __LINE__,u32DstIndex - 1,pf32FcScoresMemPool[u32DstIndex - 1], u32SrcIndex);
        }
    }
    ps32Proposals = (HI_S32 *)ps32Proposal;

    /************** bbox tranform ************/
    for (j = 0; j < u32ClsScoreChannels; j++)
    {
        for (i = 0; i < u32RoiCnt; i++)
        {
            f32ProposalWidth = (HI_FLOAT)(ps32Proposals[TEST_NNIE_COORDI_NUM * i + 2] - ps32Proposals[TEST_NNIE_COORDI_NUM * i] + 1);
            f32ProposalHeight = (HI_FLOAT)(ps32Proposals[TEST_NNIE_COORDI_NUM * i + 3] - ps32Proposals[TEST_NNIE_COORDI_NUM * i + 1] + 1);
            f32ProposalCenterX = (HI_FLOAT)(ps32Proposals[TEST_NNIE_COORDI_NUM * i] + TEST_NNIE_HALF * f32ProposalWidth);
            f32ProposalCenterY = (HI_FLOAT)(ps32Proposals[TEST_NNIE_COORDI_NUM * i + 1] + TEST_NNIE_HALF * f32ProposalHeight);
            // printf("[%s][%d] f32ProposalWidth[%f:%f:%f:%f]\n",__FUNCTION__, __LINE__,f32ProposalWidth,f32ProposalHeight,f32ProposalCenterX,f32ProposalCenterY);

            u32FcBboxIndex = 8 * i + TEST_NNIE_COORDI_NUM * j;
            f32PredCenterX = ((HI_FLOAT)ps32FcBbox[u32FcBboxIndex] / TEST_NNIE_QUANT_BASE) * f32ProposalWidth + f32ProposalCenterX;
            f32PredCenterY = ((HI_FLOAT)ps32FcBbox[u32FcBboxIndex + 1] / TEST_NNIE_QUANT_BASE) * f32ProposalHeight + f32ProposalCenterY;
            f32PredW = f32ProposalWidth * TEST_NNIE_QuickExp((HI_S32)(ps32FcBbox[u32FcBboxIndex + 2]));
            f32PredH = f32ProposalHeight * TEST_NNIE_QuickExp((HI_S32)(ps32FcBbox[u32FcBboxIndex + 3]));
            // printf("[%s][%d] u32FcBboxIndex[%d][%f:%f:%f:%f]\n",__FUNCTION__, __LINE__,u32FcBboxIndex,
            //         f32PredCenterX,f32PredCenterY,f32PredW,f32PredH);

            u32ProposalMemPoolIndex = TEST_NNIE_PROPOSAL_WIDTH * i;
            ps32ProposalMemPool[u32ProposalMemPoolIndex] = (HI_S32)(f32PredCenterX - TEST_NNIE_HALF * f32PredW);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 1] = (HI_S32)(f32PredCenterY - TEST_NNIE_HALF * f32PredH);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 2] = (HI_S32)(f32PredCenterX + TEST_NNIE_HALF * f32PredW);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 3] = (HI_S32)(f32PredCenterY + TEST_NNIE_HALF * f32PredH);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 4] = (HI_S32)(pf32FcScoresMemPool[u32ClsScoreChannels * i + j] * TEST_NNIE_QUANT_BASE);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 5] = 0; /* suprressed flag */
        }

        /* clip bbox */
        for (i = 0; i < u32RoiCnt; i++)
        {
            u32ProposalMemPoolIndex = TEST_NNIE_PROPOSAL_WIDTH * i;
            ps32ProposalMemPool[u32ProposalMemPoolIndex] = ((ps32ProposalMemPool[u32ProposalMemPoolIndex]) > ((HI_S32)u32OriImWidth - 1) ? ((HI_S32)u32OriImWidth - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex])) > 0 ? ((ps32ProposalMemPool[u32ProposalMemPoolIndex]) > ((HI_S32)u32OriImWidth) ? (u32OriImWidth - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex])) : 0;
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 1] = ((ps32ProposalMemPool[u32ProposalMemPoolIndex + 1]) > ((HI_S32)u32OriImHeight - 1) ? ((HI_S32)u32OriImHeight - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex + 1])) > 0 ? ((ps32ProposalMemPool[u32ProposalMemPoolIndex + 1]) > ((HI_S32)u32OriImHeight) ? (u32OriImHeight - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex + 1])) : 0;
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 2] = ((ps32ProposalMemPool[u32ProposalMemPoolIndex + 2]) > ((HI_S32)u32OriImWidth - 1) ? ((HI_S32)u32OriImWidth - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex + 2])) > 0 ? ((ps32ProposalMemPool[u32ProposalMemPoolIndex + 2]) > ((HI_S32)u32OriImWidth) ? (u32OriImWidth - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex + 2])) : 0;
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 3] = ((ps32ProposalMemPool[u32ProposalMemPoolIndex + 3]) > ((HI_S32)u32OriImHeight - 1) ? ((HI_S32)u32OriImHeight - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex + 3])) > 0 ? ((ps32ProposalMemPool[u32ProposalMemPoolIndex + 3]) > ((HI_S32)u32OriImHeight) ? (u32OriImHeight - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex + 3])) : 0;
        }

        ps32ProposalTmp = ps32ProposalMemPool;

        (void)TEST_NNIE_NonRecursiveArgQuickSort(ps32ProposalTmp, 0, u32RoiCnt - 1, pstStack, u32RoiCnt);

        (void)TEST_NNIE_NonMaxSuppression(ps32ProposalTmp, u32RoiCnt, u32NmsThresh, u32RoiCnt);

        ps32Score = (HI_S32 *)ps32DstScore;
        ps32Bbox = (HI_S32 *)ps32DstBbox;
        ps32RoiCnt = (HI_S32 *)ps32ClassRoiNum;

        ps32Score += (HI_S32)(u32OffSet);
        ps32Bbox += (HI_S32)(TEST_NNIE_COORDI_NUM * u32OffSet);

        u32RoiOutCnt = 0;
        for (i = 0; i < u32RoiCnt; i++)
        {
            u32ProposalMemPoolIndex = TEST_NNIE_PROPOSAL_WIDTH * i;
            if (0 == ps32ProposalMemPool[u32ProposalMemPoolIndex + 5] && ps32ProposalMemPool[u32ProposalMemPoolIndex + 4] > (HI_S32)pu32ConfThresh[j]) //Suppression = 0; CONF_THRESH == 0.8
            {
                ps32Score[u32RoiOutCnt] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 4];
                ps32Bbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM] = ps32ProposalMemPool[u32ProposalMemPoolIndex];
                ps32Bbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM + 1] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 1];
                ps32Bbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM + 2] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 2];
                ps32Bbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM + 3] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 3];
                // printf("ps32Score[%d]:%d ps32Bbox[%d] %d:%d:%d:%d\n", u32RoiOutCnt, ps32Score[u32RoiOutCnt], u32RoiOutCnt * TEST_NNIE_COORDI_NUM ,
                //                 ps32Bbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM ], ps32Bbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM + 1 ],
                //                 ps32Bbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM + 2], ps32Bbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM + 3]);

                u32RoiOutCnt++;
            }
            if (u32RoiOutCnt >= u32RoiCnt)
                break;
        }
        ps32RoiCnt[j] = (HI_S32)u32RoiOutCnt;
        // printf("[%s][%d] ps32RoiCnt[%d]:%d u32RoiCnt:%d\n", __FUNCTION__, __LINE__, j, ps32RoiCnt[j], u32RoiCnt);
        u32OffSet += u32RoiOutCnt;
    }
    return s32Ret;
}

HI_S32 TEST_NNIE_FasterRcnn_GetResult(graph_t graph, TEST_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstSoftwareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0;
    HI_S32 *ps32Proposal = (HI_S32 *)pstSoftwareParam->stRpnBbox.u64VirAddr;
    for (i = 0; i < pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height; i++)
    {
        *(ps32Proposal + TEST_NNIE_COORDI_NUM * i) /= TEST_NNIE_QUANT_BASE;
        *(ps32Proposal + TEST_NNIE_COORDI_NUM * i + 1) /= TEST_NNIE_QUANT_BASE;
        *(ps32Proposal + TEST_NNIE_COORDI_NUM * i + 2) /= TEST_NNIE_QUANT_BASE;
        *(ps32Proposal + TEST_NNIE_COORDI_NUM * i + 3) /= TEST_NNIE_QUANT_BASE;
    }
    tensor_t output_tensor1 = get_graph_output_tensor(graph, 0, 0);
    tensor_t output_tensor2 = get_graph_output_tensor(graph, 0, 1);
    void *outputData1 = get_tensor_buffer(output_tensor1);
    void *outputData2 = get_tensor_buffer(output_tensor2);

    s32Ret = TEST_NNIE_FasterRcnn_GetResult2(
        (HI_S32 *)outputData1,
        0,
        (HI_S32 *)outputData2,
        0,
        (HI_S32 *)pstSoftwareParam->stRpnBbox.u64VirAddr,
        pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height,
        pstSoftwareParam->au32ConfThresh, pstSoftwareParam->u32ValidNmsThresh,
        pstSoftwareParam->u32MaxRoiNum, pstSoftwareParam->u32ClassNum,
        pstSoftwareParam->u32OriImWidth, pstSoftwareParam->u32OriImHeight,
        (HI_U32 *)pstSoftwareParam->stGetResultTmpBuf.u64VirAddr,
        (HI_S32 *)pstSoftwareParam->stDstScore.u64VirAddr,
        (HI_S32 *)pstSoftwareParam->stDstRoi.u64VirAddr,
        (HI_S32 *)pstSoftwareParam->stClassRoiNum.u64VirAddr);

    return s32Ret;
}

static HI_S32 TEST_NNIE_Detection_PrintResult(SVP_BLOB_S *pstDstScore,
                                              SVP_BLOB_S *pstDstRoi, SVP_BLOB_S *pstClassRoiNum, HI_FLOAT f32PrintResultThresh)
{
    HI_U32 i = 0, j = 0;
    HI_U32 u32RoiNumBias = 0;
    HI_U32 u32ScoreBias = 0;
    HI_U32 u32BboxBias = 0;
    HI_FLOAT f32Score = 0.0f;
    HI_S32 *ps32Score = (HI_S32 *)pstDstScore->u64VirAddr;
    HI_S32 *ps32Roi = (HI_S32 *)pstDstRoi->u64VirAddr;
    HI_S32 *ps32ClassRoiNum = (HI_S32 *)pstClassRoiNum->u64VirAddr;
    HI_U32 u32ClassNum = pstClassRoiNum->unShape.stWhc.u32Width;
    HI_S32 s32XMin = 0, s32YMin = 0, s32XMax = 0, s32YMax = 0;
    u32RoiNumBias += ps32ClassRoiNum[0];
    for (i = 1; i < u32ClassNum; i++)
    {
        u32ScoreBias = u32RoiNumBias;
        u32BboxBias = u32RoiNumBias * TEST_NNIE_COORDI_NUM;
        /*if the confidence score greater than result threshold, the result will be printed*/
        if ((HI_FLOAT)ps32Score[u32ScoreBias] / TEST_NNIE_QUANT_BASE >=
                f32PrintResultThresh &&
            ps32ClassRoiNum[i] != 0)
        {
            printf("==== The %dth class box info====\n", i);
        }
        // printf("[%s][%d] ps32ClassRoiNum[%d]:%d \n", __FUNCTION__, __LINE__, i, ps32ClassRoiNum[i]);
        for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++)
        {
            f32Score = (HI_FLOAT)ps32Score[u32ScoreBias + j] / TEST_NNIE_QUANT_BASE;
            if (f32Score < f32PrintResultThresh)
            {
                break;
            }
            s32XMin = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM];
            s32YMin = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 1];
            s32XMax = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 2];
            s32YMax = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 3];
            printf("%d %d %d %d %f\n", s32XMin, s32YMin, s32XMax, s32YMax, f32Score);
        }
        u32RoiNumBias += ps32ClassRoiNum[i];
    }
    return HI_SUCCESS;
}

static HI_S32 TEST_NNIE_Detection_ssd_PrintResult(SVP_BLOB_S *pstDstScore,
                                                  SVP_BLOB_S *pstDstRoi, SVP_BLOB_S *pstClassRoiNum, HI_FLOAT f32PrintResultThresh, const char *image_file_org)
{
    HI_U32 i = 0, j = 0;
    HI_U32 u32RoiNumBias = 0;
    HI_U32 u32ScoreBias = 0;
    HI_U32 u32BboxBias = 0;
    HI_FLOAT f32Score = 0.0f;
    HI_S32 *ps32Score = (HI_S32 *)pstDstScore->u64VirAddr;
    HI_S32 *ps32Roi = (HI_S32 *)pstDstRoi->u64VirAddr;
    HI_S32 *ps32ClassRoiNum = (HI_S32 *)pstClassRoiNum->u64VirAddr;
    HI_U32 u32ClassNum = pstClassRoiNum->unShape.stWhc.u32Width;
    HI_S32 s32XMin = 0, s32YMin = 0, s32XMax = 0, s32YMax = 0;
    u32RoiNumBias += ps32ClassRoiNum[0];
    cv::Mat img = cv::imread(image_file_org);
    if (img.empty())
    {
        std::cerr << "failed to read image file "
                  << image_file_org
                  << "\n";
        return -1;
    }
    int raw_h = img.size().height;
    int raw_w = img.size().width;
    printf("raw_h:%d raw_w:%d \n", raw_h, raw_w);
    cv::resize(img, img, cv::Size(300, 300), 0, 0, cv::INTER_LINEAR);

    for (i = 1; i < u32ClassNum; i++)
    {
        u32ScoreBias = u32RoiNumBias;
        u32BboxBias = u32RoiNumBias * TEST_NNIE_COORDI_NUM;
        /*if the confidence score greater than result threshold, the result will be printed*/
        if ((HI_FLOAT)ps32Score[u32ScoreBias] / TEST_NNIE_QUANT_BASE >=
                f32PrintResultThresh &&
            ps32ClassRoiNum[i] != 0)
        {
            printf("==== The %dth class box info====\n", i);
        }
        // printf("[%s][%d] ps32ClassRoiNum[%d]:%d \n", __FUNCTION__, __LINE__, i, ps32ClassRoiNum[i]);
        for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++)
        {
            f32Score = (HI_FLOAT)ps32Score[u32ScoreBias + j] / TEST_NNIE_QUANT_BASE;
            if (f32Score < f32PrintResultThresh)
            {
                break;
            }
            s32XMin = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM];
            s32YMin = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 1];
            s32XMax = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 2];
            s32YMax = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 3];
            printf("%d %d %d %d %f\n", s32XMin, s32YMin, s32XMax, s32YMax, f32Score);
            cv::rectangle(img, Point(s32XMin, s32YMin), Point(s32XMax, s32YMax), Scalar(139, 0, 139, 255), 2);
        }
        u32RoiNumBias += ps32ClassRoiNum[i];
    }
    cv::resize(img, img, cv::Size(raw_w, raw_h), 0, 0, cv::INTER_LINEAR);

    cv::imwrite("ssd_out.jpg", img);
    printf("write ssd_out.jpg successful!\n");
    return HI_SUCCESS;
}

static HI_S32 TEST_NNIE_Detection_Yolov2_PrintResult(SVP_BLOB_S *pstDstScore,
                                                     SVP_BLOB_S *pstDstRoi, SVP_BLOB_S *pstClassRoiNum, HI_FLOAT f32PrintResultThresh, const char *image_file_org)
{
    HI_U32 i = 0, j = 0;
    HI_U32 u32RoiNumBias = 0;
    HI_U32 u32ScoreBias = 0;
    HI_U32 u32BboxBias = 0;
    HI_FLOAT f32Score = 0.0f;
    HI_S32 *ps32Score = (HI_S32 *)pstDstScore->u64VirAddr;
    HI_S32 *ps32Roi = (HI_S32 *)pstDstRoi->u64VirAddr;
    HI_S32 *ps32ClassRoiNum = (HI_S32 *)pstClassRoiNum->u64VirAddr;
    HI_U32 u32ClassNum = pstClassRoiNum->unShape.stWhc.u32Width;
    HI_S32 s32XMin = 0, s32YMin = 0, s32XMax = 0, s32YMax = 0;
    // printf("[%s][%d] u32ClassNum:%d \n", __FUNCTION__, __LINE__,u32ClassNum);
    u32RoiNumBias += ps32ClassRoiNum[0];
    cv::Mat img = cv::imread(image_file_org);
    if (img.empty())
    {
        std::cerr << "failed to read image file "
                  << image_file_org
                  << "\n";
        return -1;
    }
    int raw_h = img.size().height;
    int raw_w = img.size().width;
    printf("raw_h:%d raw_w:%d \n", raw_h, raw_w);
    cv::resize(img, img, cv::Size(416, 416), 0, 0, cv::INTER_LINEAR);
    for (i = 1; i < u32ClassNum; i++)
    {
        u32ScoreBias = u32RoiNumBias;
        u32BboxBias = u32RoiNumBias * TEST_NNIE_COORDI_NUM;
        /*if the confidence score greater than result threshold, the result will be printed*/
        if ((HI_FLOAT)ps32Score[u32ScoreBias] / TEST_NNIE_QUANT_BASE >=
                f32PrintResultThresh &&
            ps32ClassRoiNum[i] != 0)
        {
            printf("==== The %dth class box info====\n", i);
        }
        // printf("[%s][%d] ps32ClassRoiNum[%d]:%d \n", __FUNCTION__, __LINE__, i, ps32ClassRoiNum[i]);
        for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++)
        {
            f32Score = (HI_FLOAT)ps32Score[u32ScoreBias + j] / TEST_NNIE_QUANT_BASE;
            if (f32Score < f32PrintResultThresh)
            {
                break;
            }
            s32XMin = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM];
            s32YMin = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 1];
            s32XMax = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 2];
            s32YMax = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 3];
            printf("%d %d %d %d %f\n", s32XMin, s32YMin, s32XMax, s32YMax, f32Score);
            cv::rectangle(img, Point(s32XMin, s32YMin), Point(s32XMax, s32YMax), Scalar(139, 0, 139, 255), 2);
        }
        u32RoiNumBias += ps32ClassRoiNum[i];
    }
    cv::resize(img, img, cv::Size(raw_w, raw_h), 0, 0, cv::INTER_LINEAR);

    cv::imwrite("Yolov2_out.jpg", img);
    printf("write Yolov2_out.jpg successful!\n");
    return HI_SUCCESS;
}

static HI_S32 TEST_NNIE_SoftMax(HI_FLOAT *pf32Src, HI_U32 u32Num)
{
    HI_FLOAT f32Max = 0;
    HI_FLOAT f32Sum = 0;
    HI_U32 i = 0;

    for (i = 0; i < u32Num; ++i)
    {
        if (f32Max < pf32Src[i])
        {
            f32Max = pf32Src[i];
        }
    }

    for (i = 0; i < u32Num; ++i)
    {
        pf32Src[i] = (HI_FLOAT)TEST_NNIE_QuickExp((HI_S32)((pf32Src[i] - f32Max) * TEST_NNIE_QUANT_BASE));
        f32Sum += pf32Src[i];
    }

    for (i = 0; i < u32Num; ++i)
    {
        pf32Src[i] /= f32Sum;
    }
    return HI_SUCCESS;
}

static HI_S32 TEST_NNIE_FilterLowScoreBbox(HI_S32 *ps32Proposals, HI_U32 u32AnchorsNum,
                                           HI_U32 u32FilterThresh, HI_U32 *u32NumAfterFilter)
{
    HI_U32 u32ProposalCnt = u32AnchorsNum;
    HI_U32 i = 0;

    if (u32FilterThresh > 0)
    {
        for (i = 0; i < u32AnchorsNum; i++)
        {
            if (ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 4] < (HI_S32)u32FilterThresh)
            {
                ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 5] = 1;
            }
        }

        u32ProposalCnt = 0;
        for (i = 0; i < u32AnchorsNum; i++)
        {
            if (0 == ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 5])
            {
                ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * u32ProposalCnt] = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i];
                ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * u32ProposalCnt + 1] = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 1];
                ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * u32ProposalCnt + 2] = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 2];
                ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * u32ProposalCnt + 3] = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 3];
                ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * u32ProposalCnt + 4] = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 4];
                ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * u32ProposalCnt + 5] = ps32Proposals[TEST_NNIE_PROPOSAL_WIDTH * i + 5];
                u32ProposalCnt++;
            }
        }
    }
    *u32NumAfterFilter = u32ProposalCnt;
    return HI_SUCCESS;
}

static HI_S32 SVP_NNIE_Rpn(HI_S32 **pps32Src, HI_U32 u32NumRatioAnchors,
                           HI_U32 u32NumScaleAnchors, HI_U32 *au32Scales, HI_U32 *au32Ratios, HI_U32 u32OriImHeight,
                           HI_U32 u32OriImWidth, HI_U32 *pu32ConvHeight, HI_U32 *pu32ConvWidth, HI_U32 *pu32ConvChannel,
                           HI_U32 u32ConvStride, HI_U32 u32MaxRois, HI_U32 u32MinSize, HI_U32 u32SpatialScale,
                           HI_U32 u32NmsThresh, HI_U32 u32FilterThresh, HI_U32 u32NumBeforeNms, HI_U32 *pu32MemPool,
                           HI_S32 *ps32ProposalResult, HI_U32 *pu32NumRois)
{
    /******************** define parameters ****************/
    HI_U32 u32Size = 0;
    HI_S32 *ps32Anchors = NULL;
    HI_S32 *ps32BboxDelta = NULL;
    HI_S32 *ps32Proposals = NULL;
    HI_U32 *pu32Ptr = NULL;
    HI_S32 *ps32Ptr = NULL;
    HI_U32 u32NumAfterFilter = 0;
    HI_U32 u32NumAnchors = 0;
    HI_FLOAT f32BaseW = 0;
    HI_FLOAT f32BaseH = 0;
    HI_FLOAT f32BaseXCtr = 0;
    HI_FLOAT f32BaseYCtr = 0;
    HI_FLOAT f32SizeRatios = 0;
    HI_FLOAT *pf32RatioAnchors = NULL;
    HI_FLOAT *pf32Ptr = NULL;
    HI_FLOAT *pf32Ptr2 = NULL;
    HI_FLOAT *pf32ScaleAnchors = NULL;
    HI_FLOAT *pf32Scores = NULL;
    HI_FLOAT f32Ratios = 0;
    HI_FLOAT f32Size = 0;
    HI_U32 u32PixelInterval = 0;
    HI_U32 u32SrcBboxIndex = 0;
    HI_U32 u32SrcFgProbIndex = 0;
    HI_U32 u32SrcBgProbIndex = 0;
    HI_U32 u32SrcBboxBias = 0;
    HI_U32 u32SrcProbBias = 0;
    HI_U32 u32DesBox = 0;
    HI_U32 u32BgBlobSize = 0;
    HI_U32 u32AnchorsPerPixel = 0;
    HI_U32 u32MapSize = 0;
    HI_U32 u32LineSize = 0;
    HI_S32 *ps32Ptr2 = NULL;
    HI_S32 *ps32Ptr3 = NULL;
    HI_S32 s32ProposalWidth = 0;
    HI_S32 s32ProposalHeight = 0;
    HI_S32 s32ProposalCenterX = 0;
    HI_S32 s32ProposalCenterY = 0;
    HI_S32 s32PredW = 0;
    HI_S32 s32PredH = 0;
    HI_S32 s32PredCenterX = 0;
    HI_S32 s32PredCenterY = 0;
    HI_U32 u32DesBboxDeltaIndex = 0;
    HI_U32 u32DesScoreIndex = 0;
    HI_U32 u32RoiCount = 0;
    TEST_NNIE_STACK_S *pstStack = NULL;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 c = 0;
    HI_U32 h = 0;
    HI_U32 w = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 p = 0;
    HI_U32 q = 0;
    HI_U32 z = 0;
    HI_U32 au32BaseAnchor[4] = {0, 0, (u32MinSize - 1), (u32MinSize - 1)};
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    /*********************************** Faster RCNN *********************************************/
    /********* calculate the start pointer of each part in MemPool *********/
    pu32Ptr = (HI_U32 *)pu32MemPool;
    ps32Anchors = (HI_S32 *)pu32Ptr;
    u32NumAnchors = u32NumRatioAnchors * u32NumScaleAnchors * (pu32ConvHeight[0] * pu32ConvWidth[0]);
    u32Size = TEST_NNIE_COORDI_NUM * u32NumAnchors;
    pu32Ptr += u32Size;

    ps32BboxDelta = (HI_S32 *)pu32Ptr;
    pu32Ptr += u32Size;

    ps32Proposals = (HI_S32 *)pu32Ptr;
    u32Size = TEST_NNIE_PROPOSAL_WIDTH * u32NumAnchors;
    pu32Ptr += u32Size;

    pf32RatioAnchors = (HI_FLOAT *)pu32Ptr;
    pf32Ptr = (HI_FLOAT *)pu32Ptr;
    u32Size = u32NumRatioAnchors * TEST_NNIE_COORDI_NUM;
    pf32Ptr = pf32Ptr + u32Size;

    pf32ScaleAnchors = pf32Ptr;
    u32Size = u32NumScaleAnchors * u32NumRatioAnchors * TEST_NNIE_COORDI_NUM;
    pf32Ptr = pf32Ptr + u32Size;

    pf32Scores = pf32Ptr;
    u32Size = u32NumAnchors * TEST_NNIE_SCORE_NUM;
    pf32Ptr = pf32Ptr + u32Size;

    pstStack = (TEST_NNIE_STACK_S *)pf32Ptr;

    gettimeofday(&t1, NULL);
    float mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << " calculate the start pointer of each part in MemPool used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    gettimeofday(&t0, NULL);
    /********************* Generate the base anchor ***********************/
    f32BaseW = (HI_FLOAT)(au32BaseAnchor[2] - au32BaseAnchor[0] + 1);
    f32BaseH = (HI_FLOAT)(au32BaseAnchor[3] - au32BaseAnchor[1] + 1);
    f32BaseXCtr = (HI_FLOAT)(au32BaseAnchor[0] + ((f32BaseW - 1) * 0.5));
    f32BaseYCtr = (HI_FLOAT)(au32BaseAnchor[1] + ((f32BaseH - 1) * 0.5));

    /*************** Generate Ratio Anchors for the base anchor ***********/
    pf32Ptr = pf32RatioAnchors;
    f32Size = f32BaseW * f32BaseH;
    for (i = 0; i < u32NumRatioAnchors; i++)
    {
        f32Ratios = (HI_FLOAT)au32Ratios[i] / TEST_NNIE_QUANT_BASE;
        f32SizeRatios = f32Size / f32Ratios;
        f32BaseW = sqrt(f32SizeRatios);
        f32BaseW = (HI_FLOAT)(1.0 * ((f32BaseW) >= 0 ? (HI_S32)(f32BaseW + TEST_NNIE_HALF) : (HI_S32)(f32BaseW - TEST_NNIE_HALF)));
        f32BaseH = f32BaseW * f32Ratios;
        f32BaseH = (HI_FLOAT)(1.0 * ((f32BaseH) >= 0 ? (HI_S32)(f32BaseH + TEST_NNIE_HALF) : (HI_S32)(f32BaseH - TEST_NNIE_HALF)));

        *pf32Ptr++ = (HI_FLOAT)(f32BaseXCtr - ((f32BaseW - 1) * TEST_NNIE_HALF));
        *(pf32Ptr++) = (HI_FLOAT)(f32BaseYCtr - ((f32BaseH - 1) * TEST_NNIE_HALF));
        *(pf32Ptr++) = (HI_FLOAT)(f32BaseXCtr + ((f32BaseW - 1) * TEST_NNIE_HALF));
        *(pf32Ptr++) = (HI_FLOAT)(f32BaseYCtr + ((f32BaseH - 1) * TEST_NNIE_HALF));
    }
    gettimeofday(&t1, NULL);
    mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << " Generate Ratio Anchors for the base anchor used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    gettimeofday(&t0, NULL);
    /********* Generate Scale Anchors for each Ratio Anchor **********/
    pf32Ptr = pf32RatioAnchors;
    pf32Ptr2 = pf32ScaleAnchors;
    /* Generate Scale Anchors for one pixel */
    for (i = 0; i < u32NumRatioAnchors; i++)
    {
        for (j = 0; j < u32NumScaleAnchors; j++)
        {
            f32BaseW = *(pf32Ptr + 2) - *(pf32Ptr) + 1;
            f32BaseH = *(pf32Ptr + 3) - *(pf32Ptr + 1) + 1;
            f32BaseXCtr = (HI_FLOAT)(*(pf32Ptr) + ((f32BaseW - 1) * TEST_NNIE_HALF));
            f32BaseYCtr = (HI_FLOAT)(*(pf32Ptr + 1) + ((f32BaseH - 1) * TEST_NNIE_HALF));

            *(pf32Ptr2++) = (HI_FLOAT)(f32BaseXCtr - ((f32BaseW * ((HI_FLOAT)au32Scales[j] / TEST_NNIE_QUANT_BASE) - 1) * TEST_NNIE_HALF));
            *(pf32Ptr2++) = (HI_FLOAT)(f32BaseYCtr - ((f32BaseH * ((HI_FLOAT)au32Scales[j] / TEST_NNIE_QUANT_BASE) - 1) * TEST_NNIE_HALF));
            *(pf32Ptr2++) = (HI_FLOAT)(f32BaseXCtr + ((f32BaseW * ((HI_FLOAT)au32Scales[j] / TEST_NNIE_QUANT_BASE) - 1) * TEST_NNIE_HALF));
            *(pf32Ptr2++) = (HI_FLOAT)(f32BaseYCtr + ((f32BaseH * ((HI_FLOAT)au32Scales[j] / TEST_NNIE_QUANT_BASE) - 1) * TEST_NNIE_HALF));
        }
        pf32Ptr += TEST_NNIE_COORDI_NUM;
    }
    gettimeofday(&t1, NULL);
    mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << " Generate Scale Anchors for each Ratio Anchor used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    gettimeofday(&t0, NULL);
    /******************* Copy the anchors to every pixel in the feature map ******************/
    ps32Ptr = ps32Anchors;
    u32PixelInterval = TEST_NNIE_QUANT_BASE / u32SpatialScale;

    for (p = 0; p < pu32ConvHeight[0]; p++)
    {
        for (q = 0; q < pu32ConvWidth[0]; q++)
        {
            pf32Ptr2 = pf32ScaleAnchors;
            for (z = 0; z < u32NumScaleAnchors * u32NumRatioAnchors; z++)
            {
                *(ps32Ptr++) = (HI_S32)(q * u32PixelInterval + *(pf32Ptr2++));
                *(ps32Ptr++) = (HI_S32)(p * u32PixelInterval + *(pf32Ptr2++));
                *(ps32Ptr++) = (HI_S32)(q * u32PixelInterval + *(pf32Ptr2++));
                *(ps32Ptr++) = (HI_S32)(p * u32PixelInterval + *(pf32Ptr2++));
            }
        }
    }
    gettimeofday(&t1, NULL);
    mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << " Copy the anchors to every pixel in the feature map used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    gettimeofday(&t0, NULL);
    /********** do transpose, convert the blob from (M,C,H,W) to (M,H,W,C) **********/
    u32MapSize = pu32ConvHeight[1] * u32ConvStride / sizeof(HI_U32);
    u32AnchorsPerPixel = u32NumRatioAnchors * u32NumScaleAnchors;
    u32BgBlobSize = u32AnchorsPerPixel * u32MapSize;
    u32LineSize = u32ConvStride / sizeof(HI_U32);
    u32SrcProbBias = 0;
    u32SrcBboxBias = 0;

    for (c = 0; c < pu32ConvChannel[1]; c++)
    {
        for (h = 0; h < pu32ConvHeight[1]; h++)
        {
            for (w = 0; w < pu32ConvWidth[1]; w++)
            {
                u32SrcBboxIndex = u32SrcBboxBias + c * u32MapSize + h * u32LineSize + w;
                u32SrcBgProbIndex = u32SrcProbBias + (c / TEST_NNIE_COORDI_NUM) * u32MapSize + h * u32LineSize + w;
                u32SrcFgProbIndex = u32BgBlobSize + u32SrcBgProbIndex;

                u32DesBox = (u32AnchorsPerPixel) * (h * pu32ConvWidth[1] + w) + c / TEST_NNIE_COORDI_NUM;

                u32DesBboxDeltaIndex = TEST_NNIE_COORDI_NUM * u32DesBox + c % TEST_NNIE_COORDI_NUM;
                ps32BboxDelta[u32DesBboxDeltaIndex] = (HI_S32)pps32Src[1][u32SrcBboxIndex];

                u32DesScoreIndex = (TEST_NNIE_SCORE_NUM)*u32DesBox;
                pf32Scores[u32DesScoreIndex] = (HI_FLOAT)((HI_S32)pps32Src[0][u32SrcBgProbIndex]) / TEST_NNIE_QUANT_BASE;
                pf32Scores[u32DesScoreIndex + 1] = (HI_FLOAT)((HI_S32)pps32Src[0][u32SrcFgProbIndex]) / TEST_NNIE_QUANT_BASE;
            }
        }
    }
    gettimeofday(&t1, NULL);
    mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << "  do transpose, convert the blob from (M,C,H,W) to (M,H,W,C) used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    gettimeofday(&t0, NULL);
    /************************* do softmax ****************************/
    pf32Ptr = pf32Scores;
    for (i = 0; i < u32NumAnchors; i++)
    {
        s32Ret = TEST_NNIE_SoftMax(pf32Ptr, TEST_NNIE_SCORE_NUM);
        pf32Ptr += TEST_NNIE_SCORE_NUM;
    }
    gettimeofday(&t1, NULL);
    mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << " do softmax used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    gettimeofday(&t0, NULL);
    /************************* BBox Transform *****************************/
    /* use parameters from Conv3 to adjust the coordinates of anchors */
    ps32Ptr = ps32Anchors;
    ps32Ptr2 = ps32Proposals;
    ps32Ptr3 = ps32BboxDelta;
    for (i = 0; i < u32NumAnchors; i++)
    {
        ps32Ptr = ps32Anchors;
        ps32Ptr = ps32Ptr + TEST_NNIE_COORDI_NUM * i;
        ps32Ptr2 = ps32Proposals;
        ps32Ptr2 = ps32Ptr2 + TEST_NNIE_PROPOSAL_WIDTH * i;
        ps32Ptr3 = ps32BboxDelta;
        ps32Ptr3 = ps32Ptr3 + TEST_NNIE_COORDI_NUM * i;
        pf32Ptr = pf32Scores;
        pf32Ptr = pf32Ptr + i * (TEST_NNIE_SCORE_NUM);

        s32ProposalWidth = *(ps32Ptr + 2) - *(ps32Ptr) + 1;
        s32ProposalHeight = *(ps32Ptr + 3) - *(ps32Ptr + 1) + 1;
        s32ProposalCenterX = *(ps32Ptr) + (HI_S32)(s32ProposalWidth * TEST_NNIE_HALF);
        s32ProposalCenterY = *(ps32Ptr + 1) + (HI_S32)(s32ProposalHeight * TEST_NNIE_HALF);
        s32PredCenterX = (HI_S32)(((HI_FLOAT)(*(ps32Ptr3)) / TEST_NNIE_QUANT_BASE) * s32ProposalWidth + s32ProposalCenterX);
        s32PredCenterY = (HI_S32)(((HI_FLOAT)(*(ps32Ptr3 + 1)) / TEST_NNIE_QUANT_BASE) * s32ProposalHeight + s32ProposalCenterY);

        s32PredW = (HI_S32)(s32ProposalWidth * TEST_NNIE_QuickExp((HI_S32)(*(ps32Ptr3 + 2))));
        s32PredH = (HI_S32)(s32ProposalHeight * TEST_NNIE_QuickExp((HI_S32)(*(ps32Ptr3 + 3))));
        *(ps32Ptr2) = (HI_S32)(s32PredCenterX - TEST_NNIE_HALF * s32PredW);
        *(ps32Ptr2 + 1) = (HI_S32)(s32PredCenterY - TEST_NNIE_HALF * s32PredH);
        *(ps32Ptr2 + 2) = (HI_S32)(s32PredCenterX + TEST_NNIE_HALF * s32PredW);
        *(ps32Ptr2 + 3) = (HI_S32)(s32PredCenterY + TEST_NNIE_HALF * s32PredH);
        *(ps32Ptr2 + 4) = (HI_S32)(*(pf32Ptr + 1) * TEST_NNIE_QUANT_BASE);
        *(ps32Ptr2 + 5) = 0;
    }
    gettimeofday(&t1, NULL);
    mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << " BBox Transform used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    gettimeofday(&t0, NULL);
    /************************ clip bbox *****************************/
    for (i = 0; i < u32NumAnchors; i++)
    {
        ps32Ptr = ps32Proposals;
        ps32Ptr = ps32Ptr + TEST_NNIE_PROPOSAL_WIDTH * i;
        *ps32Ptr = TEST_NNIE_MAX(TEST_NNIE_MIN(*ps32Ptr, (HI_S32)u32OriImWidth - 1), 0);
        *(ps32Ptr + 1) = TEST_NNIE_MAX(TEST_NNIE_MIN(*(ps32Ptr + 1), (HI_S32)u32OriImHeight - 1), 0);
        *(ps32Ptr + 2) = TEST_NNIE_MAX(TEST_NNIE_MIN(*(ps32Ptr + 2), (HI_S32)u32OriImWidth - 1), 0);
        *(ps32Ptr + 3) = TEST_NNIE_MAX(TEST_NNIE_MIN(*(ps32Ptr + 3), (HI_S32)u32OriImHeight - 1), 0);
    }
    gettimeofday(&t1, NULL);
    mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << " clip bbox used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    gettimeofday(&t0, NULL);
    /************ remove the bboxes which are too small *************/
    for (i = 0; i < u32NumAnchors; i++)
    {
        ps32Ptr = ps32Proposals;
        ps32Ptr = ps32Ptr + TEST_NNIE_PROPOSAL_WIDTH * i;
        s32ProposalWidth = *(ps32Ptr + 2) - *(ps32Ptr) + 1;
        s32ProposalHeight = *(ps32Ptr + 3) - *(ps32Ptr + 1) + 1;
        if (s32ProposalWidth < (HI_S32)u32MinSize || s32ProposalHeight < (HI_S32)u32MinSize)
        {
            *(ps32Ptr + 5) = 1;
        }
    }
    gettimeofday(&t1, NULL);
    mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << " remove the bboxes which are too small used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    gettimeofday(&t0, NULL);
    /********** remove low score bboxes ************/
    (void)TEST_NNIE_FilterLowScoreBbox(ps32Proposals, u32NumAnchors, u32FilterThresh, &u32NumAfterFilter);
    gettimeofday(&t1, NULL);
    mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << " remove low score bboxes used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    gettimeofday(&t0, NULL);
    /********** sort ***********/
    (void)TEST_NNIE_NonRecursiveArgQuickSort(ps32Proposals, 0, u32NumAfterFilter - 1, pstStack, u32NumBeforeNms);
    u32NumAfterFilter = (u32NumAfterFilter < u32NumBeforeNms) ? u32NumAfterFilter : u32NumBeforeNms;
    gettimeofday(&t1, NULL);
    mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << " NonRecursiveArgQuickSort used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    gettimeofday(&t0, NULL);
    /* do nms to remove highly overlapped bbox */
    (void)TEST_NNIE_NonMaxSuppression(ps32Proposals, u32NumAfterFilter, u32NmsThresh, u32MaxRois); /* function NMS */
    gettimeofday(&t1, NULL);
    mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << " do nms to remove highly overlapped bbox used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    gettimeofday(&t0, NULL);
    /************** write the final result to output ***************/
    u32RoiCount = 0;
    for (i = 0; i < u32NumAfterFilter; i++)
    {
        ps32Ptr = ps32Proposals;
        ps32Ptr = ps32Ptr + TEST_NNIE_PROPOSAL_WIDTH * i;
        if (*(ps32Ptr + 5) == 0)
        {
            /*In this sample,the output Roi coordinates will be input in hardware,
            so the type coordinates are convert to HI_S20Q12*/
            ps32ProposalResult[TEST_NNIE_COORDI_NUM * u32RoiCount] = *ps32Ptr * TEST_NNIE_QUANT_BASE;
            ps32ProposalResult[TEST_NNIE_COORDI_NUM * u32RoiCount + 1] = *(ps32Ptr + 1) * TEST_NNIE_QUANT_BASE;
            ps32ProposalResult[TEST_NNIE_COORDI_NUM * u32RoiCount + 2] = *(ps32Ptr + 2) * TEST_NNIE_QUANT_BASE;
            ps32ProposalResult[TEST_NNIE_COORDI_NUM * u32RoiCount + 3] = *(ps32Ptr + 3) * TEST_NNIE_QUANT_BASE;
            u32RoiCount++;
        }
        if (u32RoiCount >= u32MaxRois)
        {
            break;
        }
    }
    gettimeofday(&t1, NULL);
    mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << " write the final result to output used Time: " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";
    *pu32NumRois = u32RoiCount;

    return s32Ret;
}

static TEST_NNIE_FASTERRCNN_SOFTWARE_PARAM_S stFasterRcnnSoftWareParam;
HI_S32 TEST_FasterRcnn_Rpn(struct custom_op *op, tensor_t inputs[], int input_num,
                           tensor_t outputs[], int output_num)
{
    // printf("[%s][%d] \n",__FUNCTION__,__LINE__);
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    TEST_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstSoftwareParam = (TEST_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *)op->param;
    HI_S32 s32Ret = HI_SUCCESS;
    gettimeofday(&t0, NULL);
    s32Ret = SVP_NNIE_Rpn(
        pstSoftwareParam->aps32Conv, pstSoftwareParam->u32NumRatioAnchors, pstSoftwareParam->u32NumScaleAnchors,
        pstSoftwareParam->au32Scales, pstSoftwareParam->au32Ratios, pstSoftwareParam->u32OriImHeight,
        pstSoftwareParam->u32OriImWidth, pstSoftwareParam->au32ConvHeight, pstSoftwareParam->au32ConvWidth,
        pstSoftwareParam->au32ConvChannel, pstSoftwareParam->u32ConvStride, pstSoftwareParam->u32MaxRoiNum,
        pstSoftwareParam->u32MinSize, pstSoftwareParam->u32SpatialScale, pstSoftwareParam->u32NmsThresh,
        pstSoftwareParam->u32FilterThresh, pstSoftwareParam->u32NumBeforeNms,
        (HI_U32 *)(HI_UL)pstSoftwareParam->stRpnTmpBuf.u64VirAddr,
        (HI_S32 *)(HI_UL)pstSoftwareParam->stRpnBbox.u64VirAddr,
        &pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height);
    TEST_COMM_FlushCache(
        pstSoftwareParam->stRpnBbox.u64PhyAddr, (HI_VOID *)(HI_UL)pstSoftwareParam->stRpnBbox.u64VirAddr,
        pstSoftwareParam->stRpnBbox.u32Num * pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Chn *
            pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height * pstSoftwareParam->stRpnBbox.u32Stride);
    memcpy(&stFasterRcnnSoftWareParam.stRpnBbox, &pstSoftwareParam->stRpnBbox, sizeof(SVP_DST_BLOB_S));
    gettimeofday(&t1, NULL);
    float mycpu_time = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << "\n TEST_FasterRcnn_Rpn: " << mycpu_time << " ms\n";
    std::cout << "--------------------------------------\n";

    return s32Ret;
}

static struct custom_op test_fasterRcnn_rpn_op = {
    "NnieOpCpuProrosal",
    nullptr,
    0,
    TEST_FasterRcnn_Rpn};

void TEST_NNIE_FasterRcnn()
{
    const char *image_file = "./data/nnie_image/rgb_planar/single_person_1240x375.bgr";
    const char *model_file = "./data/nnie_model/detection/inst_alexnet_frcnn_cycle.wk";
    const char *cpuConfigFile = "./inst_alexnet_frcnn_cycle_cpu.cfg";
    struct timeval t0, t1;
    /* prepare input data */
    struct stat statbuf;
    stat(image_file, &statbuf);
    int input_length = statbuf.st_size;

    void *input_data = malloc(input_length);
    if (!get_input_data(image_file, input_data, input_length))
        return;

    register_custom_op(&test_fasterRcnn_rpn_op);
    context_t nnie_context = nullptr;
    graph_t graph = create_graph(nnie_context, "nnie", model_file, cpuConfigFile);

    if (graph == nullptr)
    {
        std::cout << "Create graph failed errno: " << get_tengine_errno() << std::endl;
        return;
    }
    // dump_graph(graph);
    prerun_graph(graph);

    gettimeofday(&t0, NULL);
    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if (input_tensor == nullptr)
    {
        std::cout << "Cannot find input tensor, node_idx: " << node_idx << ",tensor_idx: " << tensor_idx << "\n";
        return;
    }
    /* setup input buffer */
    if (set_tensor_buffer(input_tensor, input_data, input_length) < 0)
    {
        std::cout << "Set data for input tensor failed\n";
        return;
    }

    gettimeofday(&t1, NULL);
    float mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << "\n mytime_0 " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";

    /* run the graph */
    float avg_time = 0.f;
    for (int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        if (run_graph(graph, 1) < 0)
        {
            std::cerr << "Run graph failed\n";
            return;
        }
        gettimeofday(&t1, NULL);

        float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
    }
    std::cout << "Model file : " << model_file << "\n"
              << "image file : " << image_file << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";
    std::cout << "--------------------------------------\n";

    gettimeofday(&t0, NULL);
    TEST_NNIE_FasterRcnn_SoftwareInit(graph, &stFasterRcnnSoftWareParam);
    TEST_NNIE_FasterRcnn_GetResult(graph, &stFasterRcnnSoftWareParam);
    gettimeofday(&t1, NULL);
    float mytime_2 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << "\n mytime_2 " << mytime_2 << " ms\n";
    std::cout << "--------------------------------------\n";

    printf("print result, Alexnet_FasterRcnn has 2 classes:\n");
    printf(" class 0:background     class 1:pedestrian \n");
    printf("FasterRcnn result:\n");

    HI_FLOAT f32PrintResultThresh = 0.8f;
    TEST_NNIE_Detection_PrintResult(&stFasterRcnnSoftWareParam.stDstScore,
                                    &stFasterRcnnSoftWareParam.stDstRoi, &stFasterRcnnSoftWareParam.stClassRoiNum, f32PrintResultThresh);
    HI_MPI_SYS_MmzFree(stFasterRcnnSoftWareParam.stRpnTmpBuf.u64PhyAddr,
                       (void *)stFasterRcnnSoftWareParam.stRpnTmpBuf.u64VirAddr);
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);
}

static HI_S32 TEST_NNIE_Cnn_SoftwareParaInit(TEST_NNIE_CNN_SOFTWARE_PARAM_S *pstCnnSoftWarePara)
{
    HI_U32 u32GetTopNMemSize = 0;
    HI_U32 u32GetTopNAssistBufSize = 0;
    HI_U32 u32GetTopNPerFrameSize = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32ClassNum = 1;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32MaxInputNum = 1;
    /*get mem size*/
    u32GetTopNPerFrameSize = pstCnnSoftWarePara->u32TopN * sizeof(TEST_NNIE_CNN_GETTOPN_UNIT_S);
    u32GetTopNMemSize = TEST_NNIE_ALIGN16(u32GetTopNPerFrameSize) * u32MaxInputNum;
    u32GetTopNAssistBufSize = u32ClassNum * sizeof(TEST_NNIE_CNN_GETTOPN_UNIT_S);
    u32TotalSize = u32GetTopNMemSize + u32GetTopNAssistBufSize;

    /*malloc mem*/
    s32Ret = SAMPLE_COMM_SVP_MallocMem("SAMPLE_CNN_INIT", NULL, (HI_U64 *)&u64PhyAddr, (void **)&pu8VirAddr, u32TotalSize);
    if (HI_SUCCESS != s32Ret)
    {
        printf("Error,Malloc memory failed!\n");
        memset(pu8VirAddr, 0, u32TotalSize);
    }

    /*init GetTopn */
    pstCnnSoftWarePara->stGetTopN.u32Num = u32MaxInputNum;
    pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Chn = 1;
    pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Height = 1;
    pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Width = u32GetTopNPerFrameSize / sizeof(HI_U32);
    pstCnnSoftWarePara->stGetTopN.u32Stride = TEST_NNIE_ALIGN16(u32GetTopNPerFrameSize);
    pstCnnSoftWarePara->stGetTopN.u64PhyAddr = u64PhyAddr;
    pstCnnSoftWarePara->stGetTopN.u64VirAddr = (HI_U64)pu8VirAddr;

    /*init AssistBuf */
    pstCnnSoftWarePara->stAssistBuf.u32Size = u32GetTopNAssistBufSize;
    pstCnnSoftWarePara->stAssistBuf.u64PhyAddr = u64PhyAddr + u32GetTopNMemSize;
    pstCnnSoftWarePara->stAssistBuf.u64VirAddr = (HI_U64)pu8VirAddr + u32GetTopNMemSize;

    return s32Ret;
}

static HI_S32 TEST_NNIE_Cnn_GetTopN(HI_S32 *ps32Fc, HI_U32 u32FcStride,
                                    HI_U32 u32ClassNum, HI_U32 u32BatchNum, HI_U32 u32TopN, HI_S32 *ps32TmpBuf,
                                    HI_U32 u32TopNStride, HI_S32 *ps32GetTopN)
{
    HI_U32 i = 0, j = 0, n = 0;
    HI_U32 u32Id = 0;
    HI_S32 *ps32Score = NULL;
    TEST_NNIE_CNN_GETTOPN_UNIT_S stTmp = {0};
    TEST_NNIE_CNN_GETTOPN_UNIT_S *pstTopN = NULL;
    TEST_NNIE_CNN_GETTOPN_UNIT_S *pstTmpBuf = (TEST_NNIE_CNN_GETTOPN_UNIT_S *)ps32TmpBuf;
    for (n = 0; n < u32BatchNum; n++)
    {
        ps32Score = (HI_S32 *)((HI_U8 *)ps32Fc + n * u32FcStride);
        pstTopN = (TEST_NNIE_CNN_GETTOPN_UNIT_S *)((HI_U8 *)ps32GetTopN + n * u32TopNStride);
        for (i = 0; i < u32ClassNum; i++)
        {
            pstTmpBuf[i].u32ClassId = i;
            pstTmpBuf[i].u32Confidence = (HI_U32)ps32Score[i];
        }

        for (i = 0; i < u32TopN; i++)
        {
            u32Id = i;
            pstTopN[i].u32ClassId = pstTmpBuf[i].u32ClassId;
            pstTopN[i].u32Confidence = pstTmpBuf[i].u32Confidence;
            for (j = i + 1; j < u32ClassNum; j++)
            {
                if (pstTmpBuf[u32Id].u32Confidence < pstTmpBuf[j].u32Confidence)
                {
                    u32Id = j;
                }
            }

            stTmp.u32ClassId = pstTmpBuf[u32Id].u32ClassId;
            stTmp.u32Confidence = pstTmpBuf[u32Id].u32Confidence;

            if (i != u32Id)
            {
                pstTmpBuf[u32Id].u32ClassId = pstTmpBuf[i].u32ClassId;
                pstTmpBuf[u32Id].u32Confidence = pstTmpBuf[i].u32Confidence;
                pstTmpBuf[i].u32ClassId = stTmp.u32ClassId;
                pstTmpBuf[i].u32Confidence = stTmp.u32Confidence;

                pstTopN[i].u32ClassId = stTmp.u32ClassId;
                pstTopN[i].u32Confidence = stTmp.u32Confidence;
            }
        }
    }

    return HI_SUCCESS;
}

HI_S32 TEST_NNIE_Cnn_PrintResult(SVP_BLOB_S *pstGetTopN, HI_U32 u32TopN)
{
    HI_U32 i = 0, j = 0;
    HI_U32 *pu32Tmp = NULL;
    HI_U32 u32Stride = pstGetTopN->u32Stride;
    if (NULL == pstGetTopN)
        printf("Error,pstGetTopN can't be NULL!\n");

    for (j = 0; j < pstGetTopN->u32Num; j++)
    {
        printf("==== The %dth image info====\n", j);
        pu32Tmp = (HI_U32 *)((HI_UL)pstGetTopN->u64VirAddr + j * u32Stride);
        for (i = 0; i < u32TopN * 2; i += 2)
        {
            printf("%d:%d\n", pu32Tmp[i], pu32Tmp[i + 1]);
        }
    }
    return HI_SUCCESS;
}

void TEST_NNIE_Cnn()
{
    const char *image_file = "./data/nnie_image/y/0_28x28.y";
    const char *model_file = "./data/nnie_model/classification/inst_mnist_cycle.wk";
    /* prepare input data */
    struct stat statbuf;
    stat(image_file, &statbuf);
    int input_length = statbuf.st_size;

    void *input_data = malloc(input_length);
    if (!get_input_data(image_file, input_data, input_length))
        return;

    context_t nnie_context = nullptr;
    graph_t graph = create_graph(nnie_context, "nnie", model_file, "noconfig");

    if (graph == nullptr)
    {
        std::cout << "Create graph failed errno: " << get_tengine_errno() << std::endl;
        return;
    }
    // dump_graph(graph);

    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if (input_tensor == nullptr)
    {
        std::cout << "Cannot find input tensor, node_idx: " << node_idx << ",tensor_idx: " << tensor_idx << "\n";
        return;
    }
    /* setup input buffer */
    if (set_tensor_buffer(input_tensor, input_data, input_length) < 0)
    {
        std::cout << "Set data for input tensor failed\n";
        return;
    }

    prerun_graph(graph);
    /* run the graph */
    struct timeval t0, t1;
    float avg_time = 0.f;
    for (int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        if (run_graph(graph, 1) < 0)
        {
            std::cerr << "Run graph failed\n";
            return;
        }
        gettimeofday(&t1, NULL);

        float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
    }
    std::cout << "Model file : " << model_file << "\n"
              << "image file : " << image_file << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";
    std::cout << "--------------------------------------\n";

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    void *output_data = get_tensor_buffer(output_tensor);
    int output_length = get_tensor_buffer_size(output_tensor);
    printf("output_data:%p output_length:%d\n", output_data, output_length);
    int dims[4];
    int dimssize = 4;
    get_tensor_shape(output_tensor, dims, dimssize);
    printf("%d:%d:%d:%d\n", dims[0], dims[1], dims[2], dims[3]);

    TEST_NNIE_CNN_SOFTWARE_PARAM_S stSoftwareParam;
    stSoftwareParam.u32TopN = 2;
    TEST_NNIE_Cnn_SoftwareParaInit(&stSoftwareParam);
    int s32Ret = TEST_NNIE_Cnn_GetTopN((HI_S32 *)output_data, dims[3], 1, 5, 1,
                                       (HI_S32 *)stSoftwareParam.stAssistBuf.u64VirAddr,
                                       stSoftwareParam.stGetTopN.u32Stride, (HI_S32 *)stSoftwareParam.stGetTopN.u64VirAddr);
    if (s32Ret != 0)
        printf("TEST_NNIE_Cnn_GetTopN failed:%d\n", s32Ret);

    TEST_NNIE_Cnn_PrintResult(&(stSoftwareParam.stGetTopN), stSoftwareParam.u32TopN);
    HI_MPI_SYS_MmzFree(stSoftwareParam.stGetTopN.u64PhyAddr, (void *)stSoftwareParam.stGetTopN.u64VirAddr);
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);
}

HI_U32 TEST_NNIE_Ssd_GetResultTmpBuf(TEST_NNIE_SSD_SOFTWARE_PARAM_S *pstSoftwareParam)
{
    HI_U32 u32PriorBoxSize = 0;
    HI_U32 u32SoftMaxSize = 0;
    HI_U32 u32DetectionSize = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32PriorNum = 0;
    HI_U32 i = 0;

    /*priorbox size*/
    for (i = 0; i < TEST_NNIE_SSD_SOFTMAX_NUM; i++)
    {
        u32PriorBoxSize += pstSoftwareParam->au32PriorBoxHeight[i] * pstSoftwareParam->au32PriorBoxWidth[i] *
                           TEST_NNIE_COORDI_NUM * 2 * (pstSoftwareParam->u32MaxSizeNum + pstSoftwareParam->u32MinSizeNum + pstSoftwareParam->au32InputAspectRatioNum[i] * 2 * pstSoftwareParam->u32MinSizeNum) * sizeof(HI_U32);
    }
    pstSoftwareParam->stPriorBoxTmpBuf.u32Size = u32PriorBoxSize;
    u32TotalSize += u32PriorBoxSize;

    /*softmax size*/
    for (i = 0; i < pstSoftwareParam->u32ConcatNum; i++)
    {
        u32SoftMaxSize += pstSoftwareParam->au32SoftMaxInChn[i] * sizeof(HI_U32);
    }
    pstSoftwareParam->stSoftMaxTmpBuf.u32Size = u32SoftMaxSize;
    u32TotalSize += u32SoftMaxSize;

    /*detection size*/
    for (i = 0; i < pstSoftwareParam->u32ConcatNum; i++)
    {
        u32PriorNum += pstSoftwareParam->au32DetectInputChn[i] / TEST_NNIE_COORDI_NUM;
    }
    u32DetectionSize += u32PriorNum * TEST_NNIE_COORDI_NUM * sizeof(HI_U32);
    u32DetectionSize += u32PriorNum * TEST_NNIE_PROPOSAL_WIDTH * sizeof(HI_U32) * 2;
    u32DetectionSize += u32PriorNum * 2 * sizeof(HI_U32);
    pstSoftwareParam->stGetResultTmpBuf.u32Size = u32DetectionSize;
    u32TotalSize += u32DetectionSize;

    return u32TotalSize;
}

static HI_S32 TEST_NNIE_Ssd_SoftwareInit(graph_t graph, TEST_NNIE_SSD_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_U32 i = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32TmpBufTotalSize = 0;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;

    /*Set Conv Parameters*/
    /*the SSD sample report resule is after permute operation,
     conv result is (C, H, W), after permute, the report node's
     (C1, H1, W1) is (H, W, C), the stride of report result is aligned according to C dim*/
    for (i = 0; i < TEST_NNIE_SSD_REPORT_NODE_NUM; i++)
    {
        tensor_t output_tensor = get_graph_output_tensor(graph, 0, i);
        int dims[4];
        int dimssize = 4;
        get_tensor_shape(output_tensor, dims, dimssize);
        //printf("tensor name:%s\n", get_tensor_name(output_tensor));

        pstSoftWareParam->au32ConvHeight[i] = dims[1];  //pstNnieParam->pstModel->astSeg[0].astDstNode[i].unShape.stWhc.u32Chn;
        pstSoftWareParam->au32ConvWidth[i] = dims[2];   //pstNnieParam->pstModel->astSeg[0].astDstNode[i].unShape.stWhc.u32Height;
        pstSoftWareParam->au32ConvChannel[i] = dims[3]; //pstNnieParam->pstModel->astSeg[0].astDstNode[i].unShape.stWhc.u32Width;
        if (i % 2 == 1)
        {
            pstSoftWareParam->au32ConvStride[i / 2] = 0; //TEST_NNIE_ALIGN16(pstSoftWareParam->au32ConvChannel[i] * sizeof(HI_U32)) / sizeof(HI_U32);
        }
    }

    /*Set PriorBox Parameters*/
    pstSoftWareParam->au32PriorBoxWidth[0] = 38;
    pstSoftWareParam->au32PriorBoxWidth[1] = 19;
    pstSoftWareParam->au32PriorBoxWidth[2] = 10;
    pstSoftWareParam->au32PriorBoxWidth[3] = 5;
    pstSoftWareParam->au32PriorBoxWidth[4] = 3;
    pstSoftWareParam->au32PriorBoxWidth[5] = 1;

    pstSoftWareParam->au32PriorBoxHeight[0] = 38;
    pstSoftWareParam->au32PriorBoxHeight[1] = 19;
    pstSoftWareParam->au32PriorBoxHeight[2] = 10;
    pstSoftWareParam->au32PriorBoxHeight[3] = 5;
    pstSoftWareParam->au32PriorBoxHeight[4] = 3;
    pstSoftWareParam->au32PriorBoxHeight[5] = 1;

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    int dims[4];
    int dimssize = 4;
    get_tensor_shape(input_tensor, dims, dimssize);
    printf("input tensor dims[%d:%d:%d:%d]\n", dims[0], dims[1], dims[2], dims[3]);

    pstSoftWareParam->u32OriImHeight = dims[2]; //pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = dims[3];  //pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;

    pstSoftWareParam->af32PriorBoxMinSize[0][0] = 30.0f;
    pstSoftWareParam->af32PriorBoxMinSize[1][0] = 60.0f;
    pstSoftWareParam->af32PriorBoxMinSize[2][0] = 111.0f;
    pstSoftWareParam->af32PriorBoxMinSize[3][0] = 162.0f;
    pstSoftWareParam->af32PriorBoxMinSize[4][0] = 213.0f;
    pstSoftWareParam->af32PriorBoxMinSize[5][0] = 264.0f;

    pstSoftWareParam->af32PriorBoxMaxSize[0][0] = 60.0f;
    pstSoftWareParam->af32PriorBoxMaxSize[1][0] = 111.0f;
    pstSoftWareParam->af32PriorBoxMaxSize[2][0] = 162.0f;
    pstSoftWareParam->af32PriorBoxMaxSize[3][0] = 213.0f;
    pstSoftWareParam->af32PriorBoxMaxSize[4][0] = 264.0f;
    pstSoftWareParam->af32PriorBoxMaxSize[5][0] = 315.0f;

    pstSoftWareParam->u32MinSizeNum = 1;
    pstSoftWareParam->u32MaxSizeNum = 1;
    pstSoftWareParam->bFlip = HI_TRUE;
    pstSoftWareParam->bClip = HI_FALSE;

    pstSoftWareParam->au32InputAspectRatioNum[0] = 1;
    pstSoftWareParam->au32InputAspectRatioNum[1] = 2;
    pstSoftWareParam->au32InputAspectRatioNum[2] = 2;
    pstSoftWareParam->au32InputAspectRatioNum[3] = 2;
    pstSoftWareParam->au32InputAspectRatioNum[4] = 1;
    pstSoftWareParam->au32InputAspectRatioNum[5] = 1;

    pstSoftWareParam->af32PriorBoxAspectRatio[0][0] = 2;
    pstSoftWareParam->af32PriorBoxAspectRatio[0][1] = 0;
    pstSoftWareParam->af32PriorBoxAspectRatio[1][0] = 2;
    pstSoftWareParam->af32PriorBoxAspectRatio[1][1] = 3;
    pstSoftWareParam->af32PriorBoxAspectRatio[2][0] = 2;
    pstSoftWareParam->af32PriorBoxAspectRatio[2][1] = 3;
    pstSoftWareParam->af32PriorBoxAspectRatio[3][0] = 2;
    pstSoftWareParam->af32PriorBoxAspectRatio[3][1] = 3;
    pstSoftWareParam->af32PriorBoxAspectRatio[4][0] = 2;
    pstSoftWareParam->af32PriorBoxAspectRatio[4][1] = 0;
    pstSoftWareParam->af32PriorBoxAspectRatio[5][0] = 2;
    pstSoftWareParam->af32PriorBoxAspectRatio[5][1] = 0;

    pstSoftWareParam->af32PriorBoxStepWidth[0] = 8;
    pstSoftWareParam->af32PriorBoxStepWidth[1] = 16;
    pstSoftWareParam->af32PriorBoxStepWidth[2] = 32;
    pstSoftWareParam->af32PriorBoxStepWidth[3] = 64;
    pstSoftWareParam->af32PriorBoxStepWidth[4] = 100;
    pstSoftWareParam->af32PriorBoxStepWidth[5] = 300;

    pstSoftWareParam->af32PriorBoxStepHeight[0] = 8;
    pstSoftWareParam->af32PriorBoxStepHeight[1] = 16;
    pstSoftWareParam->af32PriorBoxStepHeight[2] = 32;
    pstSoftWareParam->af32PriorBoxStepHeight[3] = 64;
    pstSoftWareParam->af32PriorBoxStepHeight[4] = 100;
    pstSoftWareParam->af32PriorBoxStepHeight[5] = 300;

    pstSoftWareParam->f32Offset = 0.5f;

    pstSoftWareParam->as32PriorBoxVar[0] = (HI_S32)(0.1f * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->as32PriorBoxVar[1] = (HI_S32)(0.1f * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->as32PriorBoxVar[2] = (HI_S32)(0.2f * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->as32PriorBoxVar[3] = (HI_S32)(0.2f * TEST_NNIE_QUANT_BASE);

    /*Set Softmax Parameters*/
    pstSoftWareParam->u32SoftMaxInHeight = 21;
    pstSoftWareParam->au32SoftMaxInChn[0] = 121296;
    pstSoftWareParam->au32SoftMaxInChn[1] = 45486;
    pstSoftWareParam->au32SoftMaxInChn[2] = 12600;
    pstSoftWareParam->au32SoftMaxInChn[3] = 3150;
    pstSoftWareParam->au32SoftMaxInChn[4] = 756;
    pstSoftWareParam->au32SoftMaxInChn[5] = 84;

    pstSoftWareParam->u32ConcatNum = 6;
    pstSoftWareParam->u32SoftMaxOutWidth = 1;
    pstSoftWareParam->u32SoftMaxOutHeight = 21;
    pstSoftWareParam->u32SoftMaxOutChn = 8732;

    /*Set DetectionOut Parameters*/
    pstSoftWareParam->u32ClassNum = 21;
    pstSoftWareParam->u32TopK = 400;
    pstSoftWareParam->u32KeepTopK = 200;
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.3f * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->u32ConfThresh = (HI_U32)(0.000245f * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->au32DetectInputChn[0] = 23104;
    pstSoftWareParam->au32DetectInputChn[1] = 8664;
    pstSoftWareParam->au32DetectInputChn[2] = 2400;
    pstSoftWareParam->au32DetectInputChn[3] = 600;
    pstSoftWareParam->au32DetectInputChn[4] = 144;
    pstSoftWareParam->au32DetectInputChn[5] = 16;

    /*Malloc assist buffer memory*/
    u32ClassNum = pstSoftWareParam->u32ClassNum;
    u32TotalSize = TEST_NNIE_Ssd_GetResultTmpBuf(pstSoftWareParam);
    u32DstRoiSize = TEST_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32TopK * sizeof(HI_U32) * TEST_NNIE_COORDI_NUM);
    u32DstScoreSize = TEST_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32TopK * sizeof(HI_U32));
    u32ClassRoiNumSize = TEST_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32TotalSize + u32DstRoiSize + u32DstScoreSize + u32ClassRoiNumSize;
    s32Ret = TEST_COMM_MallocCached("SAMPLE_SSD_INIT", NULL, (HI_U64 *)&u64PhyAddr,
                                    (void **)&pu8VirAddr, u32TotalSize);
    if (HI_SUCCESS != s32Ret)
    {
        printf("Error,Malloc memory failed!\n");
    }
    memset(pu8VirAddr, 0, u32TotalSize);
    TEST_COMM_FlushCache(u64PhyAddr, (void *)pu8VirAddr, u32TotalSize);

    /*set each tmp buffer addr*/
    pstSoftWareParam->stPriorBoxTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stPriorBoxTmpBuf.u64VirAddr = (HI_U64)(pu8VirAddr);

    pstSoftWareParam->stSoftMaxTmpBuf.u64PhyAddr = u64PhyAddr +
                                                   pstSoftWareParam->stPriorBoxTmpBuf.u32Size;
    pstSoftWareParam->stSoftMaxTmpBuf.u64VirAddr = (HI_U64)(pu8VirAddr +
                                                            pstSoftWareParam->stPriorBoxTmpBuf.u32Size);

    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr +
                                                     pstSoftWareParam->stPriorBoxTmpBuf.u32Size + pstSoftWareParam->stSoftMaxTmpBuf.u32Size;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)(pu8VirAddr +
                                                              pstSoftWareParam->stPriorBoxTmpBuf.u32Size + pstSoftWareParam->stSoftMaxTmpBuf.u32Size);

    u32TmpBufTotalSize = pstSoftWareParam->stPriorBoxTmpBuf.u32Size +
                         pstSoftWareParam->stSoftMaxTmpBuf.u32Size + pstSoftWareParam->stGetResultTmpBuf.u32Size;

    /*set result blob*/
    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize;
    pstSoftWareParam->stDstRoi.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize);
    pstSoftWareParam->stDstRoi.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum *
                                                             pstSoftWareParam->u32TopK * sizeof(HI_U32) * TEST_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width = u32ClassNum *
                                                        pstSoftWareParam->u32TopK * TEST_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize + u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum *
                                                               pstSoftWareParam->u32TopK * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum *
                                                          pstSoftWareParam->u32TopK;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize +
                                                 u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize +
                                                          u32DstRoiSize + u32DstScoreSize);
    pstSoftWareParam->stClassRoiNum.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}

static HI_S32 TEST_NNIE_Ssd_PriorBoxForward(HI_U32 u32PriorBoxWidth,
                                            HI_U32 u32PriorBoxHeight, HI_U32 u32OriImWidth, HI_U32 u32OriImHeight,
                                            HI_FLOAT *pf32PriorBoxMinSize, HI_U32 u32MinSizeNum, HI_FLOAT *pf32PriorBoxMaxSize,
                                            HI_U32 u32MaxSizeNum, HI_BOOL bFlip, HI_BOOL bClip, HI_U32 u32InputAspectRatioNum,
                                            HI_FLOAT af32PriorBoxAspectRatio[], HI_FLOAT f32PriorBoxStepWidth,
                                            HI_FLOAT f32PriorBoxStepHeight, HI_FLOAT f32Offset, HI_S32 as32PriorBoxVar[],
                                            HI_S32 *ps32PriorboxOutputData)
{
    HI_U32 u32AspectRatioNum = 0;
    HI_U32 u32Index = 0;
    HI_FLOAT af32AspectRatio[TEST_NNIE_SSD_ASPECT_RATIO_NUM] = {0};
    HI_U32 u32NumPrior = 0;
    HI_FLOAT f32CenterX = 0;
    HI_FLOAT f32CenterY = 0;
    HI_FLOAT f32BoxHeight = 0;
    HI_FLOAT f32BoxWidth = 0;
    HI_FLOAT f32MaxBoxWidth = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 n = 0;
    HI_U32 h = 0;
    HI_U32 w = 0;

    // generate aspect_ratios
    u32AspectRatioNum = 0;
    af32AspectRatio[0] = 1;
    u32AspectRatioNum++;
    for (i = 0; i < u32InputAspectRatioNum; i++)
    {
        af32AspectRatio[u32AspectRatioNum++] = af32PriorBoxAspectRatio[i];
        if (bFlip)
        {
            af32AspectRatio[u32AspectRatioNum++] = 1.0f / af32PriorBoxAspectRatio[i];
        }
    }
    u32NumPrior = u32MinSizeNum * u32AspectRatioNum + u32MaxSizeNum;

    u32Index = 0;
    for (h = 0; h < u32PriorBoxHeight; h++)
    {
        for (w = 0; w < u32PriorBoxWidth; w++)
        {
            f32CenterX = (w + f32Offset) * f32PriorBoxStepWidth;
            f32CenterY = (h + f32Offset) * f32PriorBoxStepHeight;
            for (n = 0; n < u32MinSizeNum; n++)
            {
                /*** first prior ***/
                f32BoxHeight = pf32PriorBoxMinSize[n];
                f32BoxWidth = pf32PriorBoxMinSize[n];
                ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX - f32BoxWidth * TEST_NNIE_HALF);
                ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY - f32BoxHeight * TEST_NNIE_HALF);
                ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX + f32BoxWidth * TEST_NNIE_HALF);
                ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY + f32BoxHeight * TEST_NNIE_HALF);
                /*** second prior ***/
                if (u32MaxSizeNum > 0)
                {
                    f32MaxBoxWidth = sqrt(pf32PriorBoxMinSize[n] * pf32PriorBoxMaxSize[n]);
                    f32BoxHeight = f32MaxBoxWidth;
                    f32BoxWidth = f32MaxBoxWidth;
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX - f32BoxWidth * TEST_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY - f32BoxHeight * TEST_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX + f32BoxWidth * TEST_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY + f32BoxHeight * TEST_NNIE_HALF);
                }
                /**** rest of priors, skip AspectRatio == 1 ****/
                for (i = 1; i < u32AspectRatioNum; i++)
                {
                    f32BoxWidth = (HI_FLOAT)(pf32PriorBoxMinSize[n] * sqrt(af32AspectRatio[i]));
                    f32BoxHeight = (HI_FLOAT)(pf32PriorBoxMinSize[n] / sqrt(af32AspectRatio[i]));
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX - f32BoxWidth * TEST_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY - f32BoxHeight * TEST_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX + f32BoxWidth * TEST_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY + f32BoxHeight * TEST_NNIE_HALF);
                }
            }
        }
    }
    /************ clip the priors' coordidates, within [0, u32ImgWidth] & [0, u32ImgHeight] *************/
    if (bClip)
    {
        for (i = 0; i < (HI_U32)(u32PriorBoxWidth * u32PriorBoxHeight * TEST_NNIE_COORDI_NUM * u32NumPrior / 2); i++)
        {
            ps32PriorboxOutputData[2 * i] = TEST_NNIE_MIN((HI_U32)TEST_NNIE_MAX(ps32PriorboxOutputData[2 * i], 0), u32OriImWidth);
            ps32PriorboxOutputData[2 * i + 1] = TEST_NNIE_MIN((HI_U32)TEST_NNIE_MAX(ps32PriorboxOutputData[2 * i + 1], 0), u32OriImHeight);
        }
    }
    /*********************** get var **********************/
    for (h = 0; h < u32PriorBoxHeight; h++)
    {
        for (w = 0; w < u32PriorBoxWidth; w++)
        {
            for (i = 0; i < u32NumPrior; i++)
            {
                for (j = 0; j < TEST_NNIE_COORDI_NUM; j++)
                {
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)as32PriorBoxVar[j];
                }
            }
        }
    }
    return HI_SUCCESS;
}
static HI_S32 TEST_NNIE_SSD_SoftMax(HI_S32 *ps32Src, HI_S32 s32ArraySize, HI_S32 *ps32Dst)
{
    /***** define parameters ****/
    HI_S32 s32Max = 0;
    HI_S32 s32Sum = 0;
    HI_S32 i = 0;
    for (i = 0; i < s32ArraySize; ++i)
    {
        if (s32Max < ps32Src[i])
        {
            s32Max = ps32Src[i];
        }
    }
    for (i = 0; i < s32ArraySize; ++i)
    {
        ps32Dst[i] = (HI_S32)(TEST_NNIE_QUANT_BASE * exp((HI_FLOAT)(ps32Src[i] - s32Max) / TEST_NNIE_QUANT_BASE));
        s32Sum += ps32Dst[i];
    }
    for (i = 0; i < s32ArraySize; ++i)
    {
        ps32Dst[i] = (HI_S32)(((HI_FLOAT)ps32Dst[i] / (HI_FLOAT)s32Sum) * TEST_NNIE_QUANT_BASE);
    }
    return HI_SUCCESS;
}
static HI_S32 TEST_NNIE_SSD_SoftmaxForward(HI_U32 u32SoftMaxInHeight,
                                           HI_U32 au32SoftMaxInChn[], HI_U32 u32ConcatNum, HI_U32 au32ConvStride[],
                                           HI_U32 au32SoftMaxWidth[], HI_S32 *aps32SoftMaxInputData[], HI_S32 *ps32SoftMaxOutputData)
{
    HI_S32 *ps32InputData = NULL;
    HI_S32 *ps32OutputTmp = NULL;
    HI_U32 u32OuterNum = 0;
    HI_U32 u32InnerNum = 0;
    HI_U32 u32InputChannel = 0;
    HI_U32 i = 0;
    HI_U32 u32ConcatCnt = 0;
    HI_S32 s32Ret = 0;
    // HI_U32 u32Stride = 0;
    HI_U32 u32Skip = 0;
    HI_U32 u32Left = 0;
    ps32OutputTmp = ps32SoftMaxOutputData;
    for (u32ConcatCnt = 0; u32ConcatCnt < u32ConcatNum; u32ConcatCnt++)
    {
        ps32InputData = aps32SoftMaxInputData[u32ConcatCnt];
        // u32Stride = au32ConvStride[u32ConcatCnt];
        u32InputChannel = au32SoftMaxInChn[u32ConcatCnt];
        u32OuterNum = u32InputChannel / u32SoftMaxInHeight;
        u32InnerNum = u32SoftMaxInHeight;
        u32Skip = au32SoftMaxWidth[u32ConcatCnt] / u32InnerNum;
        u32Left = 0; //u32Stride - au32SoftMaxWidth[u32ConcatCnt];
        for (i = 0; i < u32OuterNum; i++)
        {
            s32Ret = TEST_NNIE_SSD_SoftMax(ps32InputData, (HI_S32)u32InnerNum, ps32OutputTmp);
            if ((i + 1) % u32Skip == 0)
            {
                ps32InputData += u32Left;
            }
            ps32InputData += u32InnerNum;
            ps32OutputTmp += u32InnerNum;
        }
    }
    return s32Ret;
}

static HI_S32 TEST_NNIE_Ssd_DetectionOutForward(HI_U32 u32ConcatNum,
                                                HI_U32 u32ConfThresh, HI_U32 u32ClassNum, HI_U32 u32TopK, HI_U32 u32KeepTopK, HI_U32 u32NmsThresh,
                                                HI_U32 au32DetectInputChn[], HI_S32 *aps32AllLocPreds[], HI_S32 *aps32AllPriorBoxes[],
                                                HI_S32 *ps32ConfScores, HI_S32 *ps32AssistMemPool, HI_S32 *ps32DstScoreSrc,
                                                HI_S32 *ps32DstBboxSrc, HI_S32 *ps32RoiOutCntSrc)
{
    /************* check input parameters ****************/
    /******** define variables **********/
    HI_S32 *ps32LocPreds = NULL;
    HI_S32 *ps32PriorBoxes = NULL;
    HI_S32 *ps32PriorVar = NULL;
    HI_S32 *ps32AllDecodeBoxes = NULL;
    HI_S32 *ps32DstScore = NULL;
    HI_S32 *ps32DstBbox = NULL;
    HI_S32 *ps32ClassRoiNum = NULL;
    HI_U32 u32RoiOutCnt = 0;
    HI_S32 *ps32SingleProposal = NULL;
    HI_S32 *ps32AfterTopK = NULL;
    TEST_NNIE_STACK_S *pstStack = NULL;
    HI_U32 u32PriorNum = 0;
    HI_U32 u32NumPredsPerClass = 0;
    HI_FLOAT f32PriorWidth = 0;
    HI_FLOAT f32PriorHeight = 0;
    HI_FLOAT f32PriorCenterX = 0;
    HI_FLOAT f32PriorCenterY = 0;
    HI_FLOAT f32DecodeBoxCenterX = 0;
    HI_FLOAT f32DecodeBoxCenterY = 0;
    HI_FLOAT f32DecodeBoxWidth = 0;
    HI_FLOAT f32DecodeBoxHeight = 0;
    HI_U32 u32SrcIdx = 0;
    HI_U32 u32AfterFilter = 0;
    HI_U32 u32AfterTopK = 0;
    HI_U32 u32KeepCnt = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 u32Offset = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    u32PriorNum = 0;
    for (i = 0; i < u32ConcatNum; i++)
    {
        u32PriorNum += au32DetectInputChn[i] / TEST_NNIE_COORDI_NUM;
    }
    //prepare for Assist MemPool
    ps32AllDecodeBoxes = ps32AssistMemPool;
    ps32SingleProposal = ps32AllDecodeBoxes + u32PriorNum * TEST_NNIE_COORDI_NUM;
    ps32AfterTopK = ps32SingleProposal + TEST_NNIE_PROPOSAL_WIDTH * u32PriorNum;
    pstStack = (TEST_NNIE_STACK_S *)(ps32AfterTopK + u32PriorNum * TEST_NNIE_PROPOSAL_WIDTH);
    u32SrcIdx = 0;
    for (i = 0; i < u32ConcatNum; i++)
    {
        /********** get loc predictions ************/
        ps32LocPreds = aps32AllLocPreds[i];
        u32NumPredsPerClass = au32DetectInputChn[i] / TEST_NNIE_COORDI_NUM;
        /********** get Prior Bboxes ************/
        ps32PriorBoxes = aps32AllPriorBoxes[i];
        ps32PriorVar = ps32PriorBoxes + u32NumPredsPerClass * TEST_NNIE_COORDI_NUM;
        for (j = 0; j < u32NumPredsPerClass; j++)
        {
            //printf("ps32PriorBoxes start***************\n");
            f32PriorWidth = (HI_FLOAT)(ps32PriorBoxes[j * TEST_NNIE_COORDI_NUM + 2] - ps32PriorBoxes[j * TEST_NNIE_COORDI_NUM]);
            f32PriorHeight = (HI_FLOAT)(ps32PriorBoxes[j * TEST_NNIE_COORDI_NUM + 3] - ps32PriorBoxes[j * TEST_NNIE_COORDI_NUM + 1]);
            f32PriorCenterX = (ps32PriorBoxes[j * TEST_NNIE_COORDI_NUM + 2] + ps32PriorBoxes[j * TEST_NNIE_COORDI_NUM]) * TEST_NNIE_HALF;
            f32PriorCenterY = (ps32PriorBoxes[j * TEST_NNIE_COORDI_NUM + 3] + ps32PriorBoxes[j * TEST_NNIE_COORDI_NUM + 1]) * TEST_NNIE_HALF;

            f32DecodeBoxCenterX = ((HI_FLOAT)ps32PriorVar[j * TEST_NNIE_COORDI_NUM] / TEST_NNIE_QUANT_BASE) *
                                      ((HI_FLOAT)ps32LocPreds[j * TEST_NNIE_COORDI_NUM] / TEST_NNIE_QUANT_BASE) * f32PriorWidth +
                                  f32PriorCenterX;

            f32DecodeBoxCenterY = ((HI_FLOAT)ps32PriorVar[j * TEST_NNIE_COORDI_NUM + 1] / TEST_NNIE_QUANT_BASE) *
                                      ((HI_FLOAT)ps32LocPreds[j * TEST_NNIE_COORDI_NUM + 1] / TEST_NNIE_QUANT_BASE) * f32PriorHeight +
                                  f32PriorCenterY;

            f32DecodeBoxWidth = exp(((HI_FLOAT)ps32PriorVar[j * TEST_NNIE_COORDI_NUM + 2] / TEST_NNIE_QUANT_BASE) *
                                    ((HI_FLOAT)ps32LocPreds[j * TEST_NNIE_COORDI_NUM + 2] / TEST_NNIE_QUANT_BASE)) *
                                f32PriorWidth;

            f32DecodeBoxHeight = exp(((HI_FLOAT)ps32PriorVar[j * TEST_NNIE_COORDI_NUM + 3] / TEST_NNIE_QUANT_BASE) *
                                     ((HI_FLOAT)ps32LocPreds[j * TEST_NNIE_COORDI_NUM + 3] / TEST_NNIE_QUANT_BASE)) *
                                 f32PriorHeight;

            //printf("ps32PriorBoxes end***************\n");

            ps32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterX - f32DecodeBoxWidth * TEST_NNIE_HALF);
            ps32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterY - f32DecodeBoxHeight * TEST_NNIE_HALF);
            ps32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterX + f32DecodeBoxWidth * TEST_NNIE_HALF);
            ps32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterY + f32DecodeBoxHeight * TEST_NNIE_HALF);
        }
    }
    /********** do NMS for each class *************/
    u32AfterTopK = 0;
    for (i = 0; i < u32ClassNum; i++)
    {
        for (j = 0; j < u32PriorNum; j++)
        {
            ps32SingleProposal[j * TEST_NNIE_PROPOSAL_WIDTH] = ps32AllDecodeBoxes[j * TEST_NNIE_COORDI_NUM];
            ps32SingleProposal[j * TEST_NNIE_PROPOSAL_WIDTH + 1] = ps32AllDecodeBoxes[j * TEST_NNIE_COORDI_NUM + 1];
            ps32SingleProposal[j * TEST_NNIE_PROPOSAL_WIDTH + 2] = ps32AllDecodeBoxes[j * TEST_NNIE_COORDI_NUM + 2];
            ps32SingleProposal[j * TEST_NNIE_PROPOSAL_WIDTH + 3] = ps32AllDecodeBoxes[j * TEST_NNIE_COORDI_NUM + 3];
            ps32SingleProposal[j * TEST_NNIE_PROPOSAL_WIDTH + 4] = ps32ConfScores[j * u32ClassNum + i];
            ps32SingleProposal[j * TEST_NNIE_PROPOSAL_WIDTH + 5] = 0;
        }
        s32Ret = TEST_NNIE_NonRecursiveArgQuickSort(ps32SingleProposal, 0, u32PriorNum - 1, pstStack, u32TopK);
        u32AfterFilter = (u32PriorNum < u32TopK) ? u32PriorNum : u32TopK;
        s32Ret = TEST_NNIE_NonMaxSuppression(ps32SingleProposal, u32AfterFilter, u32NmsThresh, u32AfterFilter);
        u32RoiOutCnt = 0;
        ps32DstScore = (HI_S32 *)ps32DstScoreSrc;
        ps32DstBbox = (HI_S32 *)ps32DstBboxSrc;
        ps32ClassRoiNum = (HI_S32 *)ps32RoiOutCntSrc;
        ps32DstScore += (HI_S32)u32AfterTopK;
        ps32DstBbox += (HI_S32)(u32AfterTopK * TEST_NNIE_COORDI_NUM);
        for (j = 0; j < u32TopK; j++)
        {
            if (ps32SingleProposal[j * TEST_NNIE_PROPOSAL_WIDTH + 5] == 0 &&
                ps32SingleProposal[j * TEST_NNIE_PROPOSAL_WIDTH + 4] > (HI_S32)u32ConfThresh)
            {
                ps32DstScore[u32RoiOutCnt] = ps32SingleProposal[j * 6 + 4];
                ps32DstBbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM] = ps32SingleProposal[j * TEST_NNIE_PROPOSAL_WIDTH];
                ps32DstBbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM + 1] = ps32SingleProposal[j * TEST_NNIE_PROPOSAL_WIDTH + 1];
                ps32DstBbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM + 2] = ps32SingleProposal[j * TEST_NNIE_PROPOSAL_WIDTH + 2];
                ps32DstBbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM + 3] = ps32SingleProposal[j * TEST_NNIE_PROPOSAL_WIDTH + 3];
                u32RoiOutCnt++;
            }
        }
        ps32ClassRoiNum[i] = (HI_S32)u32RoiOutCnt;
        u32AfterTopK += u32RoiOutCnt;
    }

    u32KeepCnt = 0;
    u32Offset = 0;
    if (u32AfterTopK > u32KeepTopK)
    {
        u32Offset = ps32ClassRoiNum[0];
        for (i = 1; i < u32ClassNum; i++)
        {
            ps32DstScore = (HI_S32 *)ps32DstScoreSrc;
            ps32DstBbox = (HI_S32 *)ps32DstBboxSrc;
            ps32ClassRoiNum = (HI_S32 *)ps32RoiOutCntSrc;
            ps32DstScore += (HI_S32)(u32Offset);
            ps32DstBbox += (HI_S32)(u32Offset * TEST_NNIE_COORDI_NUM);
            for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++)
            {
                ps32AfterTopK[u32KeepCnt * TEST_NNIE_PROPOSAL_WIDTH] = ps32DstBbox[j * TEST_NNIE_COORDI_NUM];
                ps32AfterTopK[u32KeepCnt * TEST_NNIE_PROPOSAL_WIDTH + 1] = ps32DstBbox[j * TEST_NNIE_COORDI_NUM + 1];
                ps32AfterTopK[u32KeepCnt * TEST_NNIE_PROPOSAL_WIDTH + 2] = ps32DstBbox[j * TEST_NNIE_COORDI_NUM + 2];
                ps32AfterTopK[u32KeepCnt * TEST_NNIE_PROPOSAL_WIDTH + 3] = ps32DstBbox[j * TEST_NNIE_COORDI_NUM + 3];
                ps32AfterTopK[u32KeepCnt * TEST_NNIE_PROPOSAL_WIDTH + 4] = ps32DstScore[j];
                ps32AfterTopK[u32KeepCnt * TEST_NNIE_PROPOSAL_WIDTH + 5] = i;
                u32KeepCnt++;
            }
            u32Offset = u32Offset + ps32ClassRoiNum[i];
        }
        s32Ret = TEST_NNIE_NonRecursiveArgQuickSort(ps32AfterTopK, 0, u32KeepCnt - 1, pstStack, u32KeepCnt);

        u32Offset = 0;
        u32Offset = ps32ClassRoiNum[0];
        for (i = 1; i < u32ClassNum; i++)
        {
            u32RoiOutCnt = 0;
            ps32DstScore = (HI_S32 *)ps32DstScoreSrc;
            ps32DstBbox = (HI_S32 *)ps32DstBboxSrc;
            ps32ClassRoiNum = (HI_S32 *)ps32RoiOutCntSrc;
            ps32DstScore += (HI_S32)(u32Offset);
            ps32DstBbox += (HI_S32)(u32Offset * TEST_NNIE_COORDI_NUM);
            for (j = 0; j < u32KeepTopK; j++)
            {
                if (ps32AfterTopK[j * TEST_NNIE_PROPOSAL_WIDTH + 5] == (HI_S32)i)
                {
                    ps32DstScore[u32RoiOutCnt] = ps32AfterTopK[j * TEST_NNIE_PROPOSAL_WIDTH + 4];
                    ps32DstBbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM] = ps32AfterTopK[j * TEST_NNIE_PROPOSAL_WIDTH];
                    ps32DstBbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM + 1] = ps32AfterTopK[j * TEST_NNIE_PROPOSAL_WIDTH + 1];
                    ps32DstBbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM + 2] = ps32AfterTopK[j * TEST_NNIE_PROPOSAL_WIDTH + 2];
                    ps32DstBbox[u32RoiOutCnt * TEST_NNIE_COORDI_NUM + 3] = ps32AfterTopK[j * TEST_NNIE_PROPOSAL_WIDTH + 3];
                    u32RoiOutCnt++;
                }
            }
            ps32ClassRoiNum[i] = (HI_S32)u32RoiOutCnt;
            u32Offset += u32RoiOutCnt;
        }
    }
    return s32Ret;
}

HI_S32 TEST_NNIE_Ssd_GetResult(graph_t graph, TEST_NNIE_SSD_SOFTWARE_PARAM_S *pstSoftwareParam)
{
    HI_S32 *aps32PermuteResult[TEST_NNIE_SSD_REPORT_NODE_NUM];
    HI_S32 *aps32PriorboxOutputData[TEST_NNIE_SSD_PRIORBOX_NUM];
    HI_S32 *aps32SoftMaxInputData[TEST_NNIE_SSD_SOFTMAX_NUM];
    HI_S32 *aps32DetectionLocData[TEST_NNIE_SSD_SOFTMAX_NUM];
    HI_S32 *ps32SoftMaxOutputData = NULL;
    HI_S32 *ps32DetectionOutTmpBuf = NULL;
    HI_U32 au32SoftMaxWidth[TEST_NNIE_SSD_SOFTMAX_NUM];
    HI_U32 u32Size = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0;

    /*get permut result*/
    for (i = 0; i < TEST_NNIE_SSD_REPORT_NODE_NUM; i++)
    {
        tensor_t output_tensor = get_graph_output_tensor(graph, 0, i);
        void *output_data = get_tensor_buffer(output_tensor);
        aps32PermuteResult[i] = (HI_S32 *)output_data;
    }

    /*priorbox*/
    aps32PriorboxOutputData[0] = (HI_S32 *)pstSoftwareParam->stPriorBoxTmpBuf.u64VirAddr;
    for (i = 1; i < TEST_NNIE_SSD_PRIORBOX_NUM; i++)
    {
        u32Size = pstSoftwareParam->au32PriorBoxHeight[i - 1] * pstSoftwareParam->au32PriorBoxWidth[i - 1] *
                  TEST_NNIE_COORDI_NUM * 2 * (pstSoftwareParam->u32MaxSizeNum + pstSoftwareParam->u32MinSizeNum + pstSoftwareParam->au32InputAspectRatioNum[i - 1] * 2 * pstSoftwareParam->u32MinSizeNum);
        aps32PriorboxOutputData[i] = aps32PriorboxOutputData[i - 1] + u32Size;
    }

    for (i = 0; i < TEST_NNIE_SSD_PRIORBOX_NUM; i++)
    {
        s32Ret = TEST_NNIE_Ssd_PriorBoxForward(pstSoftwareParam->au32PriorBoxWidth[i],
                                               pstSoftwareParam->au32PriorBoxHeight[i], pstSoftwareParam->u32OriImWidth,
                                               pstSoftwareParam->u32OriImHeight, pstSoftwareParam->af32PriorBoxMinSize[i],
                                               pstSoftwareParam->u32MinSizeNum, pstSoftwareParam->af32PriorBoxMaxSize[i],
                                               pstSoftwareParam->u32MaxSizeNum, pstSoftwareParam->bFlip, pstSoftwareParam->bClip,
                                               pstSoftwareParam->au32InputAspectRatioNum[i], pstSoftwareParam->af32PriorBoxAspectRatio[i],
                                               pstSoftwareParam->af32PriorBoxStepWidth[i], pstSoftwareParam->af32PriorBoxStepHeight[i],
                                               pstSoftwareParam->f32Offset, pstSoftwareParam->as32PriorBoxVar,
                                               aps32PriorboxOutputData[i]);
    }

    /*softmax*/
    ps32SoftMaxOutputData = (HI_S32 *)pstSoftwareParam->stSoftMaxTmpBuf.u64VirAddr;

    for (i = 0; i < TEST_NNIE_SSD_SOFTMAX_NUM; i++)
    {
        aps32SoftMaxInputData[i] = aps32PermuteResult[i * 2 + 1];
        au32SoftMaxWidth[i] = pstSoftwareParam->au32ConvChannel[i * 2 + 1];
    }

    (void)TEST_NNIE_SSD_SoftmaxForward(pstSoftwareParam->u32SoftMaxInHeight,
                                       pstSoftwareParam->au32SoftMaxInChn, pstSoftwareParam->u32ConcatNum,
                                       pstSoftwareParam->au32ConvStride, au32SoftMaxWidth,
                                       aps32SoftMaxInputData, ps32SoftMaxOutputData);

    /*detection*/
    ps32DetectionOutTmpBuf = (HI_S32 *)pstSoftwareParam->stGetResultTmpBuf.u64VirAddr;
    for (i = 0; i < TEST_NNIE_SSD_PRIORBOX_NUM; i++)
    {
        aps32DetectionLocData[i] = aps32PermuteResult[i * 2];
    }

    (void)TEST_NNIE_Ssd_DetectionOutForward(pstSoftwareParam->u32ConcatNum,
                                            pstSoftwareParam->u32ConfThresh, pstSoftwareParam->u32ClassNum, pstSoftwareParam->u32TopK,
                                            pstSoftwareParam->u32KeepTopK, pstSoftwareParam->u32NmsThresh, pstSoftwareParam->au32DetectInputChn,
                                            aps32DetectionLocData, aps32PriorboxOutputData, ps32SoftMaxOutputData,
                                            ps32DetectionOutTmpBuf, (HI_S32 *)pstSoftwareParam->stDstScore.u64VirAddr,
                                            (HI_S32 *)pstSoftwareParam->stDstRoi.u64VirAddr,
                                            (HI_S32 *)pstSoftwareParam->stClassRoiNum.u64VirAddr);

    return s32Ret;
}

void TEST_NNIE_Ssd()
{
    // const char *image_file = "./data/nnie_image/rgb_planar/dog_bike_car_300x300.bgr";
    const char *image_file_org = "./data/nnie_image/rgb_planar/dog_bike_car.jpg";
    const char *model_file = "./data/nnie_model/detection/inst_ssd_cycle.wk";
    /* prepare input data */
    // struct stat statbuf;
    // stat(image_file, &statbuf);
    //int input_length = statbuf.st_size;
    int input_length = 3 * 300 * 300;

    void *input_data = malloc(input_length);
    std::string imgfile = image_file_org;
    // get_nnie_input_data(imgfile, (float *)input_data, 300, 300);
    int img_h = 300, img_w = 300;
    unsigned char *input_data_f = (unsigned char *)input_data;

    cv::Mat img = cv::imread(imgfile);
    if (img.empty())
    {
        std::cerr << "failed to read image file " << imgfile << "\n";
        return;
    }
    cv::resize(img, img, cv::Size(img_h, img_w));
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // img.convertTo(img, CV_32FC3);
    // img_float = (img_float - 127.5) / 128;
    unsigned char *img_data = (unsigned char *)img.data;
    int hw = img_h * img_w;
    // float mean[3]={104.f,117.f,123.f};
    //float mean[3] = {127.5, 127.5, 127.5};
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                // input_data_f[c * hw + h * img_w + w] = (*img_data - mean[c]);
                input_data_f[c * hw + h * img_w + w] = (*img_data);
                img_data++;
            }
        }
    }
    write_output_data("test_ssd.bgr", input_data, input_length);
    // const float mean[3] = {0,0,0};
    // if (!get_input_data_from_org(image_file_org, (float *)input_data, 300, 300, (float *)mean, 1.f))
    //     return;
    //if (!get_input_data(image_file, input_data, input_length)) return;

    context_t nnie_context = nullptr;
    graph_t graph = create_graph(nnie_context, "nnie", model_file, "noconfig");

    if (graph == nullptr)
    {
        std::cout << "Create graph failed errno: " << get_tengine_errno() << std::endl;
        return;
    }
    //dump_graph(graph);

    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if (input_tensor == nullptr)
    {
        std::cout << "Cannot find input tensor, node_idx: " << node_idx << ",tensor_idx: " << tensor_idx << "\n";
        return;
    }
    /* setup input buffer */
    if (set_tensor_buffer(input_tensor, input_data, input_length) < 0)
    {
        std::cout << "Set data for input tensor failed\n";
        return;
    }

    prerun_graph(graph);
    /* run the graph */
    struct timeval t0, t1;
    float avg_time = 0.f;
    for (int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        if (run_graph(graph, 1) < 0)
        {
            std::cerr << "Run graph failed\n";
            return;
        }
        gettimeofday(&t1, NULL);

        float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
    }
    std::cout << "Model file : " << model_file << "\n"
              << "org image file : " << image_file_org << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";
    std::cout << "--------------------------------------\n";

    TEST_NNIE_SSD_SOFTWARE_PARAM_S stSoftWareParam;
    TEST_NNIE_Ssd_SoftwareInit(graph, &stSoftWareParam);
    TEST_NNIE_Ssd_GetResult(graph, &stSoftWareParam);
    printf("print result, this sample has 21 classes:\n");
    printf(" class 0:background     class 1:plane           class 2:bicycle\n");
    printf(" class 3:bird           class 4:boat            class 5:bottle\n");
    printf(" class 6:bus            class 7:car             class 8:cat\n");
    printf(" class 9:chair          class10:cow             class11:diningtable\n");
    printf(" class 12:dog           class13:horse           class14:motorbike\n");
    printf(" class 15:person        class16:pottedplant     class17:sheep\n");
    printf(" class 18:sofa          class19:train           class20:tvmonitor\n");
    printf("Ssd result:\n");
    HI_FLOAT f32PrintResultThresh = 0.8f;
    TEST_NNIE_Detection_ssd_PrintResult(&stSoftWareParam.stDstScore,
                                        &stSoftWareParam.stDstRoi, &stSoftWareParam.stClassRoiNum, f32PrintResultThresh, image_file_org);
    HI_MPI_SYS_MmzFree(stSoftWareParam.stPriorBoxTmpBuf.u64PhyAddr, (void *)stSoftWareParam.stPriorBoxTmpBuf.u64VirAddr);
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);
}

HI_U32 TEST_NNIE_Yolov1_GetResultTmpBuf(TEST_NNIE_YOLOV1_SOFTWARE_PARAM_S *pstSoftwareParam)
{
    HI_U32 u32TotalGridNum = pstSoftwareParam->u32GridNumHeight * pstSoftwareParam->u32GridNumWidth;
    HI_U32 u32ClassNum = pstSoftwareParam->u32ClassNum;
    HI_U32 u32EachGridBboxNum = pstSoftwareParam->u32BboxNumEachGrid;
    HI_U32 u32TotalBboxNum = u32TotalGridNum * u32EachGridBboxNum;
    HI_U32 u32TransSize = (u32ClassNum + u32EachGridBboxNum * (TEST_NNIE_COORDI_NUM + 1)) *
                          u32TotalGridNum * sizeof(HI_U32);
    HI_U32 u32Probsize = u32ClassNum * u32TotalBboxNum * sizeof(HI_U32);
    HI_U32 u32ScoreSize = u32TotalBboxNum * sizeof(TEST_NNIE_YOLOV1_SCORE_S);
    HI_U32 u32StackSize = u32TotalBboxNum * sizeof(TEST_NNIE_STACK_S);
    HI_U32 u32TotalSize = u32TransSize + u32Probsize + u32ScoreSize + u32StackSize;
    return u32TotalSize;
}

static HI_S32 TEST_NNIE_Yolov1_SoftwareInit(graph_t graph, TEST_NNIE_YOLOV1_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32BboxNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32TmpBufTotalSize = 0;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    int dims[4]; //NCHW
    int dimssize = 4;
    get_tensor_shape(input_tensor, dims, dimssize); //NCHW
    printf("input tensor dims[%d:%d:%d:%d]\n", dims[0], dims[1], dims[2], dims[3]);

    pstSoftWareParam->u32OriImHeight = dims[2]; //pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = dims[3];  //pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32BboxNumEachGrid = 2;
    pstSoftWareParam->u32ClassNum = 20;
    pstSoftWareParam->u32GridNumHeight = 7;
    pstSoftWareParam->u32GridNumWidth = 7;
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.5f * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->u32ConfThresh = (HI_U32)(0.2f * TEST_NNIE_QUANT_BASE);

    /*Malloc assist buffer memory*/
    u32ClassNum = pstSoftWareParam->u32ClassNum + 1;
    u32BboxNum = pstSoftWareParam->u32BboxNumEachGrid * pstSoftWareParam->u32GridNumHeight *
                 pstSoftWareParam->u32GridNumWidth;
    u32TmpBufTotalSize = TEST_NNIE_Yolov1_GetResultTmpBuf(pstSoftWareParam);
    u32DstRoiSize = TEST_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32) * TEST_NNIE_COORDI_NUM);
    u32DstScoreSize = TEST_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32));
    u32ClassRoiNumSize = TEST_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32TotalSize + u32DstRoiSize + u32DstScoreSize + u32ClassRoiNumSize + u32TmpBufTotalSize;
    s32Ret = TEST_COMM_MallocCached("SAMPLE_YOLOV1_INIT", NULL, (HI_U64 *)&u64PhyAddr,
                                    (void **)&pu8VirAddr, u32TotalSize);
    if (HI_SUCCESS != s32Ret)
    {
        printf("Error,Malloc memory failed!\n");
    }
    memset(pu8VirAddr, 0, u32TotalSize);
    TEST_COMM_FlushCache(u64PhyAddr, (void *)pu8VirAddr, u32TotalSize);

    /*set each tmp buffer addr*/
    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)(pu8VirAddr);

    /*set result blob*/
    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize;
    pstSoftWareParam->stDstRoi.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize);
    pstSoftWareParam->stDstRoi.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum *
                                                             u32BboxNum * sizeof(HI_U32) * TEST_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width = u32ClassNum *
                                                        u32BboxNum * TEST_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize + u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum *
                                                               u32BboxNum * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum * u32BboxNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize +
                                                 u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize +
                                                          u32DstRoiSize + u32DstScoreSize);
    pstSoftWareParam->stClassRoiNum.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}

static void TEST_NNIE_Yolov1_Argswap(HI_S32 *ps32Src1, HI_S32 *ps32Src2,
                                     HI_U32 u32ArraySize)
{
    HI_U32 i = 0;
    HI_S32 s32Tmp = 0;
    for (i = 0; i < u32ArraySize; i++)
    {
        s32Tmp = ps32Src1[i];
        ps32Src1[i] = ps32Src2[i];
        ps32Src2[i] = s32Tmp;
    }
}

static HI_S32 TEST_NNIE_Yolo_NonRecursiveArgQuickSort(HI_S32 *ps32Array,
                                                      HI_S32 s32Low, HI_S32 s32High, HI_U32 u32ArraySize, HI_U32 u32ScoreIdx,
                                                      TEST_NNIE_STACK_S *pstStack)
{
    HI_S32 i = s32Low;
    HI_S32 j = s32High;
    HI_S32 s32Top = 0;
    HI_S32 s32KeyConfidence = ps32Array[u32ArraySize * s32Low + u32ScoreIdx];
    pstStack[s32Top].s32Min = s32Low;
    pstStack[s32Top].s32Max = s32High;

    while (s32Top > -1)
    {
        s32Low = pstStack[s32Top].s32Min;
        s32High = pstStack[s32Top].s32Max;
        i = s32Low;
        j = s32High;
        s32Top--;

        s32KeyConfidence = ps32Array[u32ArraySize * s32Low + u32ScoreIdx];

        while (i < j)
        {
            while ((i < j) && (s32KeyConfidence > ps32Array[j * u32ArraySize + u32ScoreIdx]))
            {
                j--;
            }
            if (i < j)
            {
                TEST_NNIE_Yolov1_Argswap(&ps32Array[i * u32ArraySize], &ps32Array[j * u32ArraySize], u32ArraySize);
                i++;
            }

            while ((i < j) && (s32KeyConfidence < ps32Array[i * u32ArraySize + u32ScoreIdx]))
            {
                i++;
            }
            if (i < j)
            {
                TEST_NNIE_Yolov1_Argswap(&ps32Array[i * u32ArraySize], &ps32Array[j * u32ArraySize], u32ArraySize);
                j--;
            }
        }

        if (s32Low < i - 1)
        {
            s32Top++;
            pstStack[s32Top].s32Min = s32Low;
            pstStack[s32Top].s32Max = i - 1;
        }

        if (s32High > i + 1)
        {
            s32Top++;
            pstStack[s32Top].s32Min = i + 1;
            pstStack[s32Top].s32Max = s32High;
        }
    }
    return HI_SUCCESS;
}

static HI_S32 TEST_NNIE_Yolov1_Iou(HI_FLOAT *pf32Bbox, HI_U32 u32Idx1, HI_U32 u32Idx2)
{
    HI_FLOAT f32WidthDis = 0.0f, f32HeightDis = 0.0f;
    HI_FLOAT f32Intersection = 0.0f;
    HI_FLOAT f32Iou = 0.0f;
    f32WidthDis = TEST_NNIE_MIN(pf32Bbox[u32Idx1 * TEST_NNIE_COORDI_NUM] +
                                    0.5f * pf32Bbox[u32Idx1 * TEST_NNIE_COORDI_NUM + 2],
                                pf32Bbox[u32Idx2 * TEST_NNIE_COORDI_NUM] +
                                    0.5f * pf32Bbox[u32Idx2 * TEST_NNIE_COORDI_NUM + 2]) -
                  TEST_NNIE_MAX(pf32Bbox[u32Idx1 * TEST_NNIE_COORDI_NUM] -
                                    0.5f * pf32Bbox[u32Idx1 * TEST_NNIE_COORDI_NUM + 2],
                                pf32Bbox[u32Idx2 * TEST_NNIE_COORDI_NUM] -
                                    0.5f * pf32Bbox[u32Idx2 * TEST_NNIE_COORDI_NUM + 2]);

    f32HeightDis = TEST_NNIE_MIN(pf32Bbox[u32Idx1 * TEST_NNIE_COORDI_NUM + 1] +
                                     0.5f * pf32Bbox[u32Idx1 * TEST_NNIE_COORDI_NUM + 3],
                                 pf32Bbox[u32Idx2 * TEST_NNIE_COORDI_NUM + 1] +
                                     0.5f * pf32Bbox[u32Idx2 * TEST_NNIE_COORDI_NUM + 3]) -
                   TEST_NNIE_MAX(pf32Bbox[u32Idx1 * TEST_NNIE_COORDI_NUM + 1] -
                                     0.5f * pf32Bbox[u32Idx1 * TEST_NNIE_COORDI_NUM + 3],
                                 pf32Bbox[u32Idx2 * TEST_NNIE_COORDI_NUM + 1] -
                                     0.5f * pf32Bbox[u32Idx2 * TEST_NNIE_COORDI_NUM + 3]);

    if (f32WidthDis < 0 || f32HeightDis < 0)
    {
        f32Intersection = 0;
    }
    else
    {
        f32Intersection = f32WidthDis * f32HeightDis;
    }
    f32Iou = f32Intersection / (pf32Bbox[u32Idx1 * TEST_NNIE_COORDI_NUM + 2] *
                                    pf32Bbox[u32Idx1 * TEST_NNIE_COORDI_NUM + 3] +
                                pf32Bbox[u32Idx2 * TEST_NNIE_COORDI_NUM + 2] *
                                    pf32Bbox[u32Idx2 * TEST_NNIE_COORDI_NUM + 3] -
                                f32Intersection);

    return (HI_S32)(f32Iou * TEST_NNIE_QUANT_BASE);
}

static HI_S32 TEST_NNIE_Yolov1_Nms(HI_S32 *ps32Score, HI_FLOAT *pf32Bbox,
                                   HI_U32 u32BboxNum, HI_U32 u32ConfThresh, HI_U32 u32NmsThresh, HI_U32 *pu32TmpBuf)
{
    HI_U32 i = 0, j = 0;
    HI_U32 u32Idx1 = 0, u32Idx2 = 0;
    TEST_NNIE_YOLOV1_SCORE_S *pstScore = (TEST_NNIE_YOLOV1_SCORE_S *)pu32TmpBuf;
    TEST_NNIE_STACK_S *pstAssitBuf = (TEST_NNIE_STACK_S *)((HI_U8 *)pu32TmpBuf +
                                                           u32BboxNum * sizeof(TEST_NNIE_YOLOV1_SCORE_S));
    for (i = 0; i < u32BboxNum; i++)
    {
        if (ps32Score[i] < (HI_S32)u32ConfThresh)
        {
            ps32Score[i] = 0;
        }
    }

    for (i = 0; i < u32BboxNum; ++i)
    {
        pstScore[i].u32Idx = i;
        pstScore[i].s32Score = (ps32Score[i]);
    }

    /*quick sort*/
    (void)TEST_NNIE_Yolo_NonRecursiveArgQuickSort((HI_S32 *)pstScore, 0, u32BboxNum - 1,
                                                  sizeof(TEST_NNIE_YOLOV1_SCORE_S) / sizeof(HI_U32), 1, pstAssitBuf);

    /*NMS*/
    for (i = 0; i < u32BboxNum; i++)
    {
        u32Idx1 = pstScore[i].u32Idx;
        if (0 == pstScore[i].s32Score)
        {
            continue;
        }
        for (j = i + 1; j < u32BboxNum; j++)
        {
            u32Idx2 = pstScore[j].u32Idx;
            if (0 == pstScore[j].s32Score)
            {
                continue;
            }
            if (TEST_NNIE_Yolov1_Iou(pf32Bbox, u32Idx1, u32Idx2) > (HI_S32)u32NmsThresh)
            {
                pstScore[j].s32Score = 0;
                ps32Score[pstScore[j].u32Idx] = 0;
            }
        }
    }

    return HI_SUCCESS;
}

static void TEST_NNIE_Yolov1_ConvertPosition(HI_FLOAT *pf32Bbox,
                                             HI_U32 u32OriImgWidth, HI_U32 u32OriImagHeight, HI_FLOAT af32Roi[])
{
    HI_FLOAT f32Xmin, f32Ymin, f32Xmax, f32Ymax;
    f32Xmin = *pf32Bbox - *(pf32Bbox + 2) * TEST_NNIE_HALF;
    f32Xmin = f32Xmin > 0 ? f32Xmin : 0;
    f32Ymin = *(pf32Bbox + 1) - *(pf32Bbox + 3) * TEST_NNIE_HALF;
    f32Ymin = f32Ymin > 0 ? f32Ymin : 0;
    f32Xmax = *pf32Bbox + *(pf32Bbox + 2) * TEST_NNIE_HALF;
    f32Xmax = f32Xmax > u32OriImgWidth ? u32OriImgWidth : f32Xmax;
    f32Ymax = *(pf32Bbox + 1) + *(pf32Bbox + 3) * TEST_NNIE_HALF;
    f32Ymax = f32Ymax > u32OriImagHeight ? u32OriImagHeight : f32Ymax;

    af32Roi[0] = f32Xmin;
    af32Roi[1] = f32Ymin;
    af32Roi[2] = f32Xmax;
    af32Roi[3] = f32Ymax;
}

static HI_S32 TEST_NNIE_Yolov1_Detection(HI_S32 *ps32Score, HI_FLOAT *pf32Bbox,
                                         HI_U32 u32ClassNum, HI_U32 u32GridNum, HI_U32 u32BboxNum, HI_U32 u32ConfThresh,
                                         HI_U32 u32NmsThresh, HI_U32 u32OriImgWidth, HI_U32 u32OriImgHeight,
                                         HI_U32 *pu32MemPool, HI_S32 *ps32DstScores, HI_S32 *ps32DstRoi,
                                         HI_S32 *ps32ClassRoiNum)
{
    HI_U32 i = 0, j = 0;
    HI_U32 u32Idx = 0;
    HI_U32 u32RoiNum = 0;
    HI_S32 *ps32EachClassScore = NULL;
    HI_FLOAT af32Roi[TEST_NNIE_COORDI_NUM] = {0.0f};
    TEST_NNIE_YOLOV1_SCORE_S *pstScore = NULL;
    *(ps32ClassRoiNum++) = 0;
    for (i = 0; i < u32ClassNum; i++)
    {
        ps32EachClassScore = ps32Score + u32BboxNum * i;
        (void)TEST_NNIE_Yolov1_Nms(ps32EachClassScore, pf32Bbox, u32BboxNum, u32ConfThresh,
                                   u32NmsThresh, pu32MemPool);

        pstScore = (TEST_NNIE_YOLOV1_SCORE_S *)pu32MemPool;
        u32RoiNum = 0;
        for (j = 0; j < u32BboxNum; j++)
        {
            if (pstScore[j].s32Score != 0)
            {
                u32RoiNum++;
                *(ps32DstScores++) = pstScore[j].s32Score;
                u32Idx = pstScore[j].u32Idx;
                (void)TEST_NNIE_Yolov1_ConvertPosition((pf32Bbox + u32Idx * TEST_NNIE_COORDI_NUM),
                                                       u32OriImgWidth, u32OriImgHeight, af32Roi);
                *(ps32DstRoi++) = (HI_S32)af32Roi[0];
                *(ps32DstRoi++) = (HI_S32)af32Roi[1];
                *(ps32DstRoi++) = (HI_S32)af32Roi[2];
                *(ps32DstRoi++) = (HI_S32)af32Roi[3];
            }
            else
            {
                continue;
            }
        }
        *(ps32ClassRoiNum++) = u32RoiNum;
    }
    return HI_SUCCESS;
}

HI_S32 TEST_NNIE_Yolov1_GetResult(graph_t graph, TEST_NNIE_YOLOV1_SOFTWARE_PARAM_S *pstSoftwareParam)
{
    HI_FLOAT *pf32ClassProb = NULL;
    HI_FLOAT *pf32Confidence = NULL;
    HI_FLOAT *pf32Bbox = NULL;
    HI_S32 *ps32Score = NULL;
    HI_U32 *pu32AssistBuf = NULL;
    HI_U32 u32Offset = 0;
    HI_U32 u32Index = 0;
    HI_U32 u32GridNum = pstSoftwareParam->u32GridNumHeight * pstSoftwareParam->u32GridNumWidth;
    HI_U32 i = 0, j = 0, k = 0;
    HI_U8 *pu8Tmp = (HI_U8 *)pstSoftwareParam->stGetResultTmpBuf.u64VirAddr;
    HI_FLOAT f32Score = 0.0f;
    u32Offset = u32GridNum * (pstSoftwareParam->u32BboxNumEachGrid * 5 + pstSoftwareParam->u32ClassNum);
    pf32ClassProb = (HI_FLOAT *)pu8Tmp;
    pf32Confidence = pf32ClassProb + u32GridNum * pstSoftwareParam->u32ClassNum;
    pf32Bbox = pf32Confidence + u32GridNum * pstSoftwareParam->u32BboxNumEachGrid;

    ps32Score = (HI_S32 *)(pf32ClassProb + u32Offset);
    pu32AssistBuf = (HI_U32 *)(ps32Score + u32GridNum * pstSoftwareParam->u32BboxNumEachGrid *
                                               pstSoftwareParam->u32ClassNum);

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    void *output_data = get_tensor_buffer(output_tensor);
    for (i = 0; i < u32Offset; i++)
    {
        ((HI_FLOAT *)pu8Tmp)[i] = ((HI_S32 *)output_data)[i] / ((HI_FLOAT)TEST_NNIE_QUANT_BASE);
    }
    for (i = 0; i < u32GridNum; i++)
    {
        for (j = 0; j < pstSoftwareParam->u32BboxNumEachGrid; j++)
        {
            for (k = 0; k < pstSoftwareParam->u32ClassNum; k++)
            {
                u32Offset = k * u32GridNum * pstSoftwareParam->u32BboxNumEachGrid;
                f32Score = *(pf32ClassProb + i * pstSoftwareParam->u32ClassNum + k) * *(pf32Confidence + i * pstSoftwareParam->u32BboxNumEachGrid + j);
                *(ps32Score + u32Offset + u32Index) = (HI_S32)(f32Score * TEST_NNIE_QUANT_BASE);
            }
            u32Index++;
        }
    }

    for (i = 0; i < u32GridNum; i++)
    {
        for (j = 0; j < pstSoftwareParam->u32BboxNumEachGrid; j++)
        {
            pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * TEST_NNIE_COORDI_NUM + 0] =
                (pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * TEST_NNIE_COORDI_NUM + 0] +
                 i % pstSoftwareParam->u32GridNumWidth) /
                pstSoftwareParam->u32GridNumWidth * pstSoftwareParam->u32OriImWidth;
            pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * TEST_NNIE_COORDI_NUM + 1] =
                (pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * TEST_NNIE_COORDI_NUM + 1] +
                 i / pstSoftwareParam->u32GridNumWidth) /
                pstSoftwareParam->u32GridNumHeight * pstSoftwareParam->u32OriImHeight;
            pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * TEST_NNIE_COORDI_NUM + 2] =
                pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * TEST_NNIE_COORDI_NUM + 2] *
                pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * TEST_NNIE_COORDI_NUM + 2] * pstSoftwareParam->u32OriImWidth;
            pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * TEST_NNIE_COORDI_NUM + 3] =
                pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * TEST_NNIE_COORDI_NUM + 3] *
                pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * TEST_NNIE_COORDI_NUM + 3] * pstSoftwareParam->u32OriImHeight;
        }
    }

    (void)TEST_NNIE_Yolov1_Detection(ps32Score, pf32Bbox,
                                     pstSoftwareParam->u32ClassNum, u32GridNum, u32GridNum * pstSoftwareParam->u32BboxNumEachGrid,
                                     pstSoftwareParam->u32ConfThresh, pstSoftwareParam->u32NmsThresh,
                                     pstSoftwareParam->u32OriImWidth, pstSoftwareParam->u32OriImHeight, pu32AssistBuf,
                                     (HI_S32 *)pstSoftwareParam->stDstScore.u64VirAddr,
                                     (HI_S32 *)pstSoftwareParam->stDstRoi.u64VirAddr,
                                     (HI_S32 *)pstSoftwareParam->stClassRoiNum.u64VirAddr);
    return HI_SUCCESS;
}

void TEST_NNIE_Yolov1()
{
    const char *image_file = "./data/nnie_image/rgb_planar/dog_bike_car_448x448.bgr";
    const char *model_file = "./data/nnie_model/detection/inst_yolov1_cycle.wk";
    /* prepare input data */
    struct stat statbuf;
    stat(image_file, &statbuf);
    int input_length = statbuf.st_size;

    void *input_data = malloc(input_length);
    if (!get_input_data(image_file, input_data, input_length))
        return;

    context_t nnie_context = nullptr;
    graph_t graph = create_graph(nnie_context, "nnie", model_file, "noconfig");

    if (graph == nullptr)
    {
        std::cout << "Create graph failed errno: " << get_tengine_errno() << std::endl;
        return;
    }
    // dump_graph(graph);

    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if (input_tensor == nullptr)
    {
        std::cout << "Cannot find input tensor, node_idx: " << node_idx << ",tensor_idx: " << tensor_idx << "\n";
        return;
    }
    /* setup input buffer */
    if (set_tensor_buffer(input_tensor, input_data, input_length) < 0)
    {
        std::cout << "Set data for input tensor failed\n";
        return;
    }

    prerun_graph(graph);
    /* run the graph */
    struct timeval t0, t1;
    float avg_time = 0.f;
    for (int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        if (run_graph(graph, 1) < 0)
        {
            std::cerr << "Run graph failed\n";
            return;
        }
        gettimeofday(&t1, NULL);

        float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
    }
    std::cout << "Model file : " << model_file << "\n"
              << "image file : " << image_file << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";
    std::cout << "--------------------------------------\n";

    TEST_NNIE_YOLOV1_SOFTWARE_PARAM_S stSoftWareParam;
    TEST_NNIE_Yolov1_SoftwareInit(graph, &stSoftWareParam);
    TEST_NNIE_Yolov1_GetResult(graph, &stSoftWareParam);
    printf("print result, this sample has 21 classes:\n");
    printf(" class 0:background     class 1:plane           class 2:bicycle\n");
    printf(" class 3:bird           class 4:boat            class 5:bottle\n");
    printf(" class 6:bus            class 7:car             class 8:cat\n");
    printf(" class 9:chair          class10:cow             class11:diningtable\n");
    printf(" class 12:dog           class13:horse           class14:motorbike\n");
    printf(" class 15:person        class16:pottedplant     class17:sheep\n");
    printf(" class 18:sofa          class19:train           class20:tvmonitor\n");
    printf("Yolov1 result:\n");
    HI_FLOAT f32PrintResultThresh = 0.3f;
    TEST_NNIE_Detection_PrintResult(&stSoftWareParam.stDstScore, &stSoftWareParam.stDstRoi, &stSoftWareParam.stClassRoiNum, f32PrintResultThresh);
    HI_MPI_SYS_MmzFree(stSoftWareParam.stGetResultTmpBuf.u64PhyAddr, (void *)stSoftWareParam.stGetResultTmpBuf.u64VirAddr);
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);
}

HI_U32 TEST_NNIE_Yolov2_GetResultTmpBuf(TEST_NNIE_YOLOV2_SOFTWARE_PARAM_S *pstSoftwareParam)
{
    HI_U32 u32TotalGridNum = pstSoftwareParam->u32GridNumHeight * pstSoftwareParam->u32GridNumWidth;
    HI_U32 u32ParaLength = pstSoftwareParam->u32BboxNumEachGrid * (TEST_NNIE_COORDI_NUM + 1 + pstSoftwareParam->u32ClassNum);
    HI_U32 u32TotalBboxNum = u32TotalGridNum * pstSoftwareParam->u32BboxNumEachGrid;
    HI_U32 u32TransSize = u32TotalGridNum * u32ParaLength * sizeof(HI_U32);
    HI_U32 u32BboxAssistBufSize = u32TotalBboxNum * sizeof(TEST_NNIE_STACK_S);
    HI_U32 u32BboxBufSize = u32TotalBboxNum * sizeof(TEST_NNIE_YOLOV2_BBOX_S);
    HI_U32 u32BboxTmpBufSize = u32TotalGridNum * u32ParaLength * sizeof(HI_FLOAT);
    HI_U32 u32TotalSize = u32TransSize + u32BboxAssistBufSize + u32BboxBufSize + u32BboxTmpBufSize;
    return u32TotalSize;
}

static HI_S32 TEST_NNIE_Yolov2_SoftwareInit(graph_t graph, TEST_NNIE_YOLOV2_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32BboxNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32TmpBufTotalSize = 0;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    int dims[4]; //NCHW
    int dimssize = 4;
    get_tensor_shape(input_tensor, dims, dimssize); //NCHW
    printf("input tensor dims[%d:%d:%d:%d]\n", dims[0], dims[1], dims[2], dims[3]);

    pstSoftWareParam->u32OriImHeight = dims[2]; //pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = dims[3];  //pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32BboxNumEachGrid = 5;
    pstSoftWareParam->u32ClassNum = 5;
    pstSoftWareParam->u32GridNumHeight = 13;
    pstSoftWareParam->u32GridNumWidth = 13;
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.3f * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->u32ConfThresh = (HI_U32)(0.25f * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->u32MaxRoiNum = 10;
    pstSoftWareParam->af32Bias[0] = 1.08;
    pstSoftWareParam->af32Bias[1] = 1.19;
    pstSoftWareParam->af32Bias[2] = 3.42;
    pstSoftWareParam->af32Bias[3] = 4.41;
    pstSoftWareParam->af32Bias[4] = 6.63;
    pstSoftWareParam->af32Bias[5] = 11.38;
    pstSoftWareParam->af32Bias[6] = 9.42;
    pstSoftWareParam->af32Bias[7] = 5.11;
    pstSoftWareParam->af32Bias[8] = 16.62;
    pstSoftWareParam->af32Bias[9] = 10.52;

    /*Malloc assist buffer memory*/
    u32ClassNum = pstSoftWareParam->u32ClassNum + 1;
    u32BboxNum = pstSoftWareParam->u32BboxNumEachGrid * pstSoftWareParam->u32GridNumHeight *
                 pstSoftWareParam->u32GridNumWidth;
    u32TmpBufTotalSize = TEST_NNIE_Yolov2_GetResultTmpBuf(pstSoftWareParam);
    u32DstRoiSize = TEST_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32) * TEST_NNIE_COORDI_NUM);
    u32DstScoreSize = TEST_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32));
    u32ClassRoiNumSize = TEST_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32TotalSize + u32DstRoiSize + u32DstScoreSize + u32ClassRoiNumSize + u32TmpBufTotalSize;
    s32Ret = TEST_COMM_MallocCached("SAMPLE_YOLOV2_INIT", NULL, (HI_U64 *)&u64PhyAddr,
                                    (void **)&pu8VirAddr, u32TotalSize);
    if (HI_SUCCESS != s32Ret)
    {
        printf("Error,Malloc memory failed!\n");
    }
    memset(pu8VirAddr, 0, u32TotalSize);
    TEST_COMM_FlushCache(u64PhyAddr, (void *)pu8VirAddr, u32TotalSize);

    /*set each tmp buffer addr*/
    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)(pu8VirAddr);

    /*set result blob*/
    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize;
    pstSoftWareParam->stDstRoi.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize);
    pstSoftWareParam->stDstRoi.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum *
                                                             u32BboxNum * sizeof(HI_U32) * TEST_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width = u32ClassNum *
                                                        u32BboxNum * TEST_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize + u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum *
                                                               u32BboxNum * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum * u32BboxNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize +
                                                 u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize +
                                                          u32DstRoiSize + u32DstScoreSize);
    pstSoftWareParam->stClassRoiNum.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}

static HI_FLOAT TEST_NNIE_Yolov2_GetMaxVal(HI_FLOAT *pf32Val, HI_U32 u32Num,
                                           HI_U32 *pu32MaxValueIndex)
{
    HI_U32 i = 0;
    HI_FLOAT f32MaxTmp = 0;

    f32MaxTmp = pf32Val[0];
    *pu32MaxValueIndex = 0;
    for (i = 1; i < u32Num; i++)
    {
        if (pf32Val[i] > f32MaxTmp)
        {
            f32MaxTmp = pf32Val[i];
            *pu32MaxValueIndex = i;
        }
    }

    return f32MaxTmp;
}

static HI_DOUBLE TEST_NNIE_Yolov2_Iou(TEST_NNIE_YOLOV2_BBOX_S *pstBbox1,
                                      TEST_NNIE_YOLOV2_BBOX_S *pstBbox2)
{
    HI_FLOAT f32InterWidth = 0.0;
    HI_FLOAT f32InterHeight = 0.0;
    HI_DOUBLE f64InterArea = 0.0;
    HI_DOUBLE f64Box1Area = 0.0;
    HI_DOUBLE f64Box2Area = 0.0;
    HI_DOUBLE f64UnionArea = 0.0;

    f32InterWidth = TEST_NNIE_MIN(pstBbox1->f32Xmax, pstBbox2->f32Xmax) - TEST_NNIE_MAX(pstBbox1->f32Xmin, pstBbox2->f32Xmin);
    f32InterHeight = TEST_NNIE_MIN(pstBbox1->f32Ymax, pstBbox2->f32Ymax) - TEST_NNIE_MAX(pstBbox1->f32Ymin, pstBbox2->f32Ymin);

    if (f32InterWidth <= 0 || f32InterHeight <= 0)
        return 0;

    f64InterArea = f32InterWidth * f32InterHeight;
    f64Box1Area = (pstBbox1->f32Xmax - pstBbox1->f32Xmin) * (pstBbox1->f32Ymax - pstBbox1->f32Ymin);
    f64Box2Area = (pstBbox2->f32Xmax - pstBbox2->f32Xmin) * (pstBbox2->f32Ymax - pstBbox2->f32Ymin);
    f64UnionArea = f64Box1Area + f64Box2Area - f64InterArea;

    return f64InterArea / f64UnionArea;
}

static HI_S32 TEST_NNIE_Yolov2_NonMaxSuppression(TEST_NNIE_YOLOV2_BBOX_S *pstBbox,
                                                 HI_U32 u32BboxNum, HI_U32 u32NmsThresh, HI_U32 u32MaxRoiNum)
{
    HI_U32 i, j;
    HI_U32 u32Num = 0;
    HI_DOUBLE f64Iou = 0.0;

    for (i = 0; i < u32BboxNum && u32Num < u32MaxRoiNum; i++)
    {
        if (pstBbox[i].u32Mask == 0)
        {
            u32Num++;
            for (j = i + 1; j < u32BboxNum; j++)
            {
                if (pstBbox[j].u32Mask == 0)
                {
                    f64Iou = TEST_NNIE_Yolov2_Iou(&pstBbox[i], &pstBbox[j]);
                    if (f64Iou >= (HI_DOUBLE)u32NmsThresh / TEST_NNIE_QUANT_BASE)
                    {
                        pstBbox[j].u32Mask = 1;
                    }
                }
            }
        }
    }

    return HI_SUCCESS;
}

static HI_S32 TEST_NNIE_Yolov2_GetResult(HI_S32 *ps32InputData, HI_U32 u32GridNumWidth,
                                         HI_U32 u32GridNumHeight, HI_U32 u32EachGridBbox, HI_U32 u32ClassNum, HI_U32 u32SrcWidth,
                                         HI_U32 u32SrcHeight, HI_U32 u32MaxRoiNum, HI_U32 u32NmsThresh, HI_U32 u32ConfThresh,
                                         HI_FLOAT af32Bias[], HI_U32 *pu32TmpBuf, HI_S32 *ps32DstScores, HI_S32 *ps32DstRoi,
                                         HI_S32 *ps32ClassRoiNum)
{
    HI_U32 u32GridNum = u32GridNumWidth * u32GridNumHeight;
    HI_U32 u32ParaNum = (TEST_NNIE_COORDI_NUM + 1 + u32ClassNum);
    HI_U32 u32TotalBboxNum = u32GridNum * u32EachGridBbox;
    HI_U32 u32CStep = u32GridNum;
    HI_U32 u32HStep = u32GridNumWidth;
    HI_U32 u32BoxsNum = 0;
    HI_FLOAT *pf32BoxTmp = NULL;
    HI_FLOAT *f32InputData = NULL;
    HI_FLOAT f32ObjScore = 0.0;
    HI_FLOAT f32MaxScore = 0.0;
    HI_S32 s32Score = 0;
    HI_U32 u32MaxValueIndex = 0;
    HI_U32 h = 0, w = 0, n = 0;
    HI_U32 c = 0, k = 0, i = 0;
    HI_U32 u32Index = 0;
    HI_FLOAT x, y, f32Width, f32Height;
    HI_U32 u32AssistBuffSize = u32TotalBboxNum * sizeof(TEST_NNIE_STACK_S);
    HI_U32 u32BoxBuffSize = u32TotalBboxNum * sizeof(TEST_NNIE_YOLOV2_BBOX_S);
    HI_U32 u32BoxResultNum = 0;
    TEST_NNIE_STACK_S *pstAssistStack = NULL;
    TEST_NNIE_YOLOV2_BBOX_S *pstBox = NULL;

    /*store float type data*/
    f32InputData = (HI_FLOAT *)pu32TmpBuf;
    /*assist buffer for sort*/
    pstAssistStack = (TEST_NNIE_STACK_S *)(f32InputData + u32TotalBboxNum * u32ParaNum);
    /*assit box buffer*/
    pstBox = (TEST_NNIE_YOLOV2_BBOX_S *)((HI_U8 *)pstAssistStack + u32AssistBuffSize);
    /*box tmp buffer*/
    pf32BoxTmp = (HI_FLOAT *)((HI_U8 *)pstBox + u32BoxBuffSize);

    for (i = 0; i < u32TotalBboxNum * u32ParaNum; i++)
    {
        f32InputData[i] = (HI_FLOAT)(ps32InputData[i]) / TEST_NNIE_QUANT_BASE;
    }

    //permute
    for (h = 0; h < u32GridNumHeight; h++)
    {
        for (w = 0; w < u32GridNumWidth; w++)
        {
            for (c = 0; c < u32EachGridBbox * u32ParaNum; c++)
            {
                pf32BoxTmp[n++] = f32InputData[c * u32CStep + h * u32HStep + w];
            }
        }
    }

    for (n = 0; n < u32GridNum; n++)
    {
        //Grid
        w = n % u32GridNumWidth;
        h = n / u32GridNumWidth;
        for (k = 0; k < u32EachGridBbox; k++)
        {
            u32Index = (n * u32EachGridBbox + k) * u32ParaNum;
            x = (HI_FLOAT)((w + TEST_NNIE_SIGMOID(pf32BoxTmp[u32Index + 0])) / u32GridNumWidth);              // x
            y = (HI_FLOAT)((h + TEST_NNIE_SIGMOID(pf32BoxTmp[u32Index + 1])) / u32GridNumHeight);             // y
            f32Width = (HI_FLOAT)((exp(pf32BoxTmp[u32Index + 2]) * af32Bias[2 * k]) / u32GridNumWidth);       // w
            f32Height = (HI_FLOAT)((exp(pf32BoxTmp[u32Index + 3]) * af32Bias[2 * k + 1]) / u32GridNumHeight); // h

            f32ObjScore = TEST_NNIE_SIGMOID(pf32BoxTmp[u32Index + 4]);
            TEST_NNIE_SoftMax(&pf32BoxTmp[u32Index + 5], u32ClassNum);

            f32MaxScore = TEST_NNIE_Yolov2_GetMaxVal(&pf32BoxTmp[u32Index + 5], u32ClassNum, &u32MaxValueIndex);

            s32Score = (HI_S32)(f32MaxScore * f32ObjScore * TEST_NNIE_QUANT_BASE);
            if (s32Score > (HI_S32)u32ConfThresh)
            {
                pstBox[u32BoxsNum].f32Xmin = (HI_FLOAT)(x - f32Width * TEST_NNIE_HALF);
                pstBox[u32BoxsNum].f32Xmax = (HI_FLOAT)(x + f32Width * TEST_NNIE_HALF);
                pstBox[u32BoxsNum].f32Ymin = (HI_FLOAT)(y - f32Height * TEST_NNIE_HALF);
                pstBox[u32BoxsNum].f32Ymax = (HI_FLOAT)(y + f32Height * TEST_NNIE_HALF);
                pstBox[u32BoxsNum].s32ClsScore = s32Score;
                pstBox[u32BoxsNum].u32ClassIdx = u32MaxValueIndex + 1;
                pstBox[u32BoxsNum].u32Mask = 0;
                u32BoxsNum++;
            }
        }
    }
    //quick_sort
    if (u32BoxsNum > 1)
    {
        TEST_NNIE_Yolo_NonRecursiveArgQuickSort((HI_S32 *)pstBox, 0, u32BoxsNum - 1, sizeof(TEST_NNIE_YOLOV2_BBOX_S) / sizeof(HI_S32),
                                                4, pstAssistStack);
    }
    //Nms
    TEST_NNIE_Yolov2_NonMaxSuppression(pstBox, u32BoxsNum, u32NmsThresh, u32MaxRoiNum);
    //Get the result
    memset((void *)ps32ClassRoiNum, 0, (u32ClassNum + 1) * sizeof(HI_U32));
    for (i = 1; i < u32ClassNum + 1; i++)
    {
        for (n = 0; n < u32BoxsNum && u32BoxResultNum < u32MaxRoiNum; n++)
        {
            if (0 == pstBox[n].u32Mask && i == pstBox[n].u32ClassIdx)
            {
                *(ps32DstRoi++) = (HI_S32)TEST_NNIE_MAX(pstBox[n].f32Xmin * u32SrcWidth, 0);
                *(ps32DstRoi++) = (HI_S32)TEST_NNIE_MAX(pstBox[n].f32Ymin * u32SrcHeight, 0);
                *(ps32DstRoi++) = (HI_S32)TEST_NNIE_MIN(pstBox[n].f32Xmax * u32SrcWidth, u32SrcWidth);
                *(ps32DstRoi++) = (HI_S32)TEST_NNIE_MIN(pstBox[n].f32Ymax * u32SrcHeight, u32SrcHeight);
                *(ps32DstScores++) = pstBox[n].s32ClsScore;
                *(ps32ClassRoiNum + pstBox[n].u32ClassIdx) = *(ps32ClassRoiNum + pstBox[n].u32ClassIdx) + 1;
                u32BoxResultNum++;
            }
        }
    }
    return HI_SUCCESS;
}

void TEST_NNIE_Yolov2()
{
    const char *image_file = "./data/nnie_image/rgb_planar/street_cars_416x416.bgr";
    const char *image_file_org = "./data/nnie_image/rgb_planar/street_cars.png";
    const char *model_file = "./data/nnie_model/detection/inst_yolov2_cycle.wk";
    struct timeval t0, t1;
    /* prepare input data */
    struct stat statbuf;
    stat(image_file, &statbuf);
    int input_length = statbuf.st_size;

    void *input_data = malloc(input_length);
    if (!get_input_data(image_file, input_data, input_length))
        return;
    context_t nnie_context = nullptr;
    graph_t graph = create_graph(nnie_context, "nnie", model_file, "noconfig");

    if (graph == nullptr)
    {
        std::cout << "Create graph failed errno: " << get_tengine_errno() << std::endl;
        return;
    }
    // dump_graph(graph);
    prerun_graph(graph);

    gettimeofday(&t0, NULL);
    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if (input_tensor == nullptr)
    {
        std::cout << "Cannot find input tensor, node_idx: " << node_idx << ",tensor_idx: " << tensor_idx << "\n";
        return;
    }
    /* setup input buffer */
    if (set_tensor_buffer(input_tensor, input_data, input_length) < 0)
    {
        std::cout << "Set data for input tensor failed\n";
        return;
    }
    gettimeofday(&t1, NULL);
    float mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << "\n mytime_0 " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";

    /* run the graph */
    float avg_time = 0.f;
    for (int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        if (run_graph(graph, 1) < 0)
        {
            std::cerr << "Run graph failed\n";
            return;
        }
        gettimeofday(&t1, NULL);

        float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
    }
    std::cout << "Model file : " << model_file << "\n"
              << "image file : " << image_file << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";
    std::cout << "--------------------------------------\n";

    gettimeofday(&t0, NULL);
    TEST_NNIE_YOLOV2_SOFTWARE_PARAM_S stSoftWareParam;
    TEST_NNIE_Yolov2_SoftwareInit(graph, &stSoftWareParam);
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    void *output_data = get_tensor_buffer(output_tensor);
    TEST_NNIE_Yolov2_GetResult((HI_S32 *)output_data,
                               stSoftWareParam.u32GridNumWidth,
                               stSoftWareParam.u32GridNumHeight,
                               stSoftWareParam.u32BboxNumEachGrid, stSoftWareParam.u32ClassNum,
                               stSoftWareParam.u32OriImWidth,
                               stSoftWareParam.u32OriImHeight,
                               stSoftWareParam.u32MaxRoiNum, stSoftWareParam.u32NmsThresh,
                               stSoftWareParam.u32ConfThresh, stSoftWareParam.af32Bias,
                               (HI_U32 *)stSoftWareParam.stGetResultTmpBuf.u64VirAddr,
                               (HI_S32 *)stSoftWareParam.stDstScore.u64VirAddr,
                               (HI_S32 *)stSoftWareParam.stDstRoi.u64VirAddr,
                               (HI_S32 *)stSoftWareParam.stClassRoiNum.u64VirAddr);
    gettimeofday(&t1, NULL);
    float mytime_2 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << "\n mytime_2 " << mytime_2 << " ms\n";
    std::cout << "--------------------------------------\n";

    printf("print result, this sample has 6 classes:\n");
    printf("class 0:background     class 1:Carclass           class 2:Vanclass\n");
    printf("class 3:Truckclass     class 4:Pedestrianclass    class 5:Cyclist\n");
    printf("Yolov2 result:\n");
    HI_FLOAT f32PrintResultThresh = 0.25f;
    TEST_NNIE_Detection_Yolov2_PrintResult(&stSoftWareParam.stDstScore,
                                           &stSoftWareParam.stDstRoi, &stSoftWareParam.stClassRoiNum, f32PrintResultThresh, image_file_org);
    HI_MPI_SYS_MmzFree(stSoftWareParam.stGetResultTmpBuf.u64PhyAddr, (void *)stSoftWareParam.stGetResultTmpBuf.u64VirAddr);
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);
}

HI_U32 TEST_NNIE_Yolov3_GetResultTmpBuf(graph_t graph,
                                        TEST_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftwareParam)
{
    HI_U32 u32TotalSize = 0;
    HI_U32 u32AssistStackSize = 0;
    HI_U32 u32TotalBboxNum = 0;
    HI_U32 u32TotalBboxSize = 0;
    HI_U32 u32DstBlobSize = 0;
    HI_U32 u32MaxBlobSize = 0;
    HI_U32 i = 0;
    node_t outputNode = get_graph_output_node(graph, 0);
    HI_U32 outputNum = get_node_output_number(outputNode);

    for (i = 0; i < outputNum; i++)
    {
        tensor_t output_tensor = get_graph_output_tensor(graph, 0, i);
        u32DstBlobSize = get_tensor_buffer_size(output_tensor);
        if (u32MaxBlobSize < u32DstBlobSize)
        {
            u32MaxBlobSize = u32DstBlobSize;
        }
        u32TotalBboxNum += pstSoftwareParam->au32GridNumWidth[i] * pstSoftwareParam->au32GridNumHeight[i] *
                           pstSoftwareParam->u32BboxNumEachGrid;
    }
    u32AssistStackSize = u32TotalBboxNum * sizeof(TEST_NNIE_STACK_S);
    u32TotalBboxSize = u32TotalBboxNum * sizeof(TEST_NNIE_YOLOV3_BBOX_S);
    u32TotalSize += (u32MaxBlobSize + u32AssistStackSize + u32TotalBboxSize);

    return u32TotalSize;
}

static HI_S32 TEST_NNIE_Yolov3_SoftwareInit(graph_t graph, TEST_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32TmpBufTotalSize = 0;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    int dims[4]; //NCHW
    int dimssize = 4;
    get_tensor_shape(input_tensor, dims, dimssize); //NCHW
    printf("input tensor dims[%d:%d:%d:%d]\n", dims[0], dims[1], dims[2], dims[3]);

    pstSoftWareParam->u32OriImHeight = dims[2]; //pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = dims[3];  //pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32BboxNumEachGrid = 3;
    pstSoftWareParam->u32ClassNum = 80;
    pstSoftWareParam->au32GridNumHeight[0] = 13;
    pstSoftWareParam->au32GridNumHeight[1] = 26;
    pstSoftWareParam->au32GridNumHeight[2] = 52;
    pstSoftWareParam->au32GridNumWidth[0] = 13;
    pstSoftWareParam->au32GridNumWidth[1] = 26;
    pstSoftWareParam->au32GridNumWidth[2] = 52;
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.3f * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->u32ConfThresh = (HI_U32)(0.5f * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->u32MaxRoiNum = 10;
    pstSoftWareParam->af32Bias[0][0] = 116;
    pstSoftWareParam->af32Bias[0][1] = 90;
    pstSoftWareParam->af32Bias[0][2] = 156;
    pstSoftWareParam->af32Bias[0][3] = 198;
    pstSoftWareParam->af32Bias[0][4] = 373;
    pstSoftWareParam->af32Bias[0][5] = 326;
    pstSoftWareParam->af32Bias[1][0] = 30;
    pstSoftWareParam->af32Bias[1][1] = 61;
    pstSoftWareParam->af32Bias[1][2] = 62;
    pstSoftWareParam->af32Bias[1][3] = 45;
    pstSoftWareParam->af32Bias[1][4] = 59;
    pstSoftWareParam->af32Bias[1][5] = 119;
    pstSoftWareParam->af32Bias[2][0] = 10;
    pstSoftWareParam->af32Bias[2][1] = 13;
    pstSoftWareParam->af32Bias[2][2] = 16;
    pstSoftWareParam->af32Bias[2][3] = 30;
    pstSoftWareParam->af32Bias[2][4] = 33;
    pstSoftWareParam->af32Bias[2][5] = 23;

    /*Malloc assist buffer memory*/
    u32ClassNum = pstSoftWareParam->u32ClassNum + 1;

    u32TmpBufTotalSize = TEST_NNIE_Yolov3_GetResultTmpBuf(graph, pstSoftWareParam);
    u32DstRoiSize = TEST_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32) * TEST_NNIE_COORDI_NUM);
    u32DstScoreSize = TEST_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32));
    u32ClassRoiNumSize = TEST_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32TotalSize + u32DstRoiSize + u32DstScoreSize + u32ClassRoiNumSize + u32TmpBufTotalSize;
    s32Ret = TEST_COMM_MallocCached("SAMPLE_YOLOV3_INIT", NULL, (HI_U64 *)&u64PhyAddr,
                                    (void **)&pu8VirAddr, u32TotalSize);
    if (HI_SUCCESS != s32Ret)
        printf("Error,Malloc memory failed!\n");

    memset(pu8VirAddr, 0, u32TotalSize);
    TEST_COMM_FlushCache(u64PhyAddr, (void *)pu8VirAddr, u32TotalSize);

    /*set each tmp buffer addr*/
    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)(pu8VirAddr);

    /*set result blob*/
    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize;
    pstSoftWareParam->stDstRoi.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize);
    pstSoftWareParam->stDstRoi.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum *
                                                             pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32) * TEST_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width = u32ClassNum *
                                                        pstSoftWareParam->u32MaxRoiNum * TEST_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize + u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum *
                                                               pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum * pstSoftWareParam->u32MaxRoiNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize +
                                                 u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize +
                                                          u32DstRoiSize + u32DstScoreSize);
    pstSoftWareParam->stClassRoiNum.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}

static HI_S32 TEST_NNIE_Yolov3_GetResult(HI_S32 **pps32InputData, HI_U32 au32GridNumWidth[],
                                         HI_U32 au32GridNumHeight[], HI_U32 au32Stride[], HI_U32 u32EachGridBbox, HI_U32 u32ClassNum, HI_U32 u32SrcWidth,
                                         HI_U32 u32SrcHeight, HI_U32 u32MaxRoiNum, HI_U32 u32NmsThresh, HI_U32 u32ConfThresh,
                                         HI_FLOAT af32Bias[TEST_NNIE_YOLOV3_REPORT_BLOB_NUM][TEST_NNIE_YOLOV3_EACH_GRID_BIAS_NUM],
                                         HI_S32 *ps32TmpBuf, HI_S32 *ps32DstScore, HI_S32 *ps32DstRoi, HI_S32 *ps32ClassRoiNum)
{
    HI_S32 *ps32InputBlob = NULL;
    HI_FLOAT *pf32Permute = NULL;
    TEST_NNIE_YOLOV3_BBOX_S *pstBbox = NULL;
    HI_S32 *ps32AssistBuf = NULL;
    HI_U32 u32TotalBboxNum = 0;
    HI_U32 u32ChnOffset = 0;
    HI_U32 u32HeightOffset = 0;
    HI_U32 u32BboxNum = 0;
    HI_U32 u32GridXIdx;
    HI_U32 u32GridYIdx;
    HI_U32 u32Offset;
    HI_FLOAT f32StartX;
    HI_FLOAT f32StartY;
    HI_FLOAT f32Width;
    HI_FLOAT f32Height;
    HI_FLOAT f32ObjScore;
    HI_U32 u32MaxValueIndex = 0;
    HI_FLOAT f32MaxScore;
    HI_S32 s32ClassScore;
    HI_U32 u32ClassRoiNum;
    HI_U32 i = 0, j = 0, k = 0, c = 0, h = 0, w = 0;
    HI_U32 u32BlobSize = 0;
    HI_U32 u32MaxBlobSize = 0;

    for (i = 0; i < TEST_NNIE_YOLOV3_REPORT_BLOB_NUM; i++)
    {
        u32BlobSize = au32GridNumWidth[i] * au32GridNumHeight[i] * sizeof(HI_U32) *
                      TEST_NNIE_YOLOV3_EACH_BBOX_INFER_RESULT_NUM * u32EachGridBbox;
        if (u32MaxBlobSize < u32BlobSize)
        {
            u32MaxBlobSize = u32BlobSize;
        }
    }

    for (i = 0; i < TEST_NNIE_YOLOV3_REPORT_BLOB_NUM; i++)
    {
        u32TotalBboxNum += au32GridNumWidth[i] * au32GridNumHeight[i] * u32EachGridBbox;
    }

    //get each tmpbuf addr
    pf32Permute = (HI_FLOAT *)ps32TmpBuf;
    pstBbox = (TEST_NNIE_YOLOV3_BBOX_S *)(pf32Permute + u32MaxBlobSize / sizeof(HI_S32));
    ps32AssistBuf = (HI_S32 *)(pstBbox + u32TotalBboxNum);

    for (i = 0; i < TEST_NNIE_YOLOV3_REPORT_BLOB_NUM; i++)
    {
        //permute
        u32Offset = 0;
        ps32InputBlob = pps32InputData[i];
        u32ChnOffset = au32GridNumHeight[i] * au32GridNumWidth[i];
        u32HeightOffset = au32GridNumWidth[i];

        for (h = 0; h < au32GridNumHeight[i]; h++)
        {
            for (w = 0; w < au32GridNumWidth[i]; w++)
            {
                for (c = 0; c < TEST_NNIE_YOLOV3_EACH_BBOX_INFER_RESULT_NUM * u32EachGridBbox; c++)
                {
                    pf32Permute[u32Offset++] = (HI_FLOAT)(ps32InputBlob[c * u32ChnOffset + h * u32HeightOffset + w]) / TEST_NNIE_QUANT_BASE;
                }
            }
        }

        //decode bbox and calculate score
        for (j = 0; j < au32GridNumWidth[i] * au32GridNumHeight[i]; j++)
        {
            u32GridXIdx = j % au32GridNumWidth[i];
            u32GridYIdx = j / au32GridNumWidth[i];
            for (k = 0; k < u32EachGridBbox; k++)
            {
                u32MaxValueIndex = 0;
                u32Offset = (j * u32EachGridBbox + k) * TEST_NNIE_YOLOV3_EACH_BBOX_INFER_RESULT_NUM;
                //decode bbox
                float a[4] = {-pf32Permute[u32Offset + 0], -pf32Permute[u32Offset + 1], pf32Permute[u32Offset + 2], pf32Permute[u32Offset + 3]};
                float x[4] = {0.f, 0.f, 0.f, 0.f};
                fast_exp_4f(a, x);
                f32StartX = ((HI_FLOAT)u32GridXIdx + TEST_NNIE_SIGMOID_NOEXP(x[0])) / au32GridNumWidth[i];
                f32StartY = ((HI_FLOAT)u32GridYIdx + TEST_NNIE_SIGMOID_NOEXP(x[1])) / au32GridNumHeight[i];
                f32Width = (HI_FLOAT)((x[2]) * af32Bias[i][2 * k]) / u32SrcWidth;
                f32Height = (HI_FLOAT)((x[3]) * af32Bias[i][2 * k + 1]) / u32SrcHeight;

                //calculate score
                f32ObjScore = TEST_NNIE_SIGMOID(pf32Permute[u32Offset + 4]);
                (void)TEST_NNIE_SoftMax(&pf32Permute[u32Offset + 5], u32ClassNum);
                f32MaxScore = TEST_NNIE_Yolov2_GetMaxVal(&pf32Permute[u32Offset + 5], u32ClassNum, &u32MaxValueIndex);
                s32ClassScore = (HI_S32)(f32MaxScore * f32ObjScore * TEST_NNIE_QUANT_BASE);
                //filter low score roi
                if (s32ClassScore > (HI_S32)u32ConfThresh)
                {
                    pstBbox[u32BboxNum].f32Xmin = (HI_FLOAT)(f32StartX - f32Width * 0.5f);
                    pstBbox[u32BboxNum].f32Ymin = (HI_FLOAT)(f32StartY - f32Height * 0.5f);
                    pstBbox[u32BboxNum].f32Xmax = (HI_FLOAT)(f32StartX + f32Width * 0.5f);
                    pstBbox[u32BboxNum].f32Ymax = (HI_FLOAT)(f32StartY + f32Height * 0.5f);
                    pstBbox[u32BboxNum].s32ClsScore = s32ClassScore;
                    pstBbox[u32BboxNum].u32Mask = 0;
                    pstBbox[u32BboxNum].u32ClassIdx = (HI_S32)(u32MaxValueIndex + 1);
                    u32BboxNum++;
                }
            }
        }
    }

    // float mytime_exp = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    //quick sort
    (void)TEST_NNIE_Yolo_NonRecursiveArgQuickSort((HI_S32 *)pstBbox, 0, u32BboxNum - 1,
                                                  sizeof(TEST_NNIE_YOLOV3_BBOX_S) / sizeof(HI_U32), 4, (TEST_NNIE_STACK_S *)ps32AssistBuf);
    (void)TEST_NNIE_Yolov2_NonMaxSuppression(pstBbox, u32BboxNum, u32NmsThresh, sizeof(TEST_NNIE_YOLOV3_BBOX_S) / sizeof(HI_U32));

    //Get result
    for (i = 1; i < u32ClassNum; i++)
    {
        u32ClassRoiNum = 0;
        for (j = 0; j < u32BboxNum; j++)
        {
            if ((0 == pstBbox[j].u32Mask) && (i == pstBbox[j].u32ClassIdx) && (u32ClassRoiNum < u32MaxRoiNum))
            {
                *(ps32DstRoi++) = TEST_NNIE_MAX((HI_S32)(pstBbox[j].f32Xmin * u32SrcWidth), 0);
                *(ps32DstRoi++) = TEST_NNIE_MAX((HI_S32)(pstBbox[j].f32Ymin * u32SrcHeight), 0);
                *(ps32DstRoi++) = TEST_NNIE_MIN((pstBbox[j].f32Xmax * u32SrcWidth), u32SrcWidth);
                *(ps32DstRoi++) = TEST_NNIE_MIN((pstBbox[j].f32Ymax * u32SrcHeight), u32SrcHeight);
                *(ps32DstScore++) = pstBbox[j].s32ClsScore;
                u32ClassRoiNum++;
            }
        }
        *(ps32ClassRoiNum + i) = u32ClassRoiNum;
    }
    return HI_SUCCESS;
}

HI_S32 TEST_NNIE_Yolov3_GetResult(graph_t graph,
                                  TEST_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftwareParam)
{
    HI_U32 i = 0;
    HI_S32 *aps32InputBlob[TEST_NNIE_YOLOV3_REPORT_BLOB_NUM] = {0};
    HI_U32 au32Stride[TEST_NNIE_YOLOV3_REPORT_BLOB_NUM] = {0};

    for (i = 0; i < TEST_NNIE_YOLOV3_REPORT_BLOB_NUM; i++)
    {
        tensor_t output_tensor = get_graph_output_tensor(graph, 0, i);
        void *output_data = get_tensor_buffer(output_tensor);
        aps32InputBlob[i] = (HI_S32 *)output_data; //pstNnieParam->astSegData[0].astDst[i].u64VirAddr;
        au32Stride[i] = 0;
    }
    return TEST_NNIE_Yolov3_GetResult(aps32InputBlob, pstSoftwareParam->au32GridNumWidth,
                                      pstSoftwareParam->au32GridNumHeight, au32Stride, pstSoftwareParam->u32BboxNumEachGrid,
                                      pstSoftwareParam->u32ClassNum, pstSoftwareParam->u32OriImWidth,
                                      pstSoftwareParam->u32OriImWidth, pstSoftwareParam->u32MaxRoiNum, pstSoftwareParam->u32NmsThresh,
                                      pstSoftwareParam->u32ConfThresh, pstSoftwareParam->af32Bias,
                                      (HI_S32 *)pstSoftwareParam->stGetResultTmpBuf.u64VirAddr,
                                      (HI_S32 *)pstSoftwareParam->stDstScore.u64VirAddr,
                                      (HI_S32 *)pstSoftwareParam->stDstRoi.u64VirAddr,
                                      (HI_S32 *)pstSoftwareParam->stClassRoiNum.u64VirAddr);
}

static HI_S32 TEST_NNIE_Detection_Yolov3_PrintResult(SVP_BLOB_S *pstDstScore,
                                                     SVP_BLOB_S *pstDstRoi, SVP_BLOB_S *pstClassRoiNum, HI_FLOAT f32PrintResultThresh, const char *image_file)
{
    HI_U32 i = 0, j = 0;
    HI_U32 u32RoiNumBias = 0;
    HI_U32 u32ScoreBias = 0;
    HI_U32 u32BboxBias = 0;
    HI_FLOAT f32Score = 0.0f;
    HI_S32 *ps32Score = (HI_S32 *)pstDstScore->u64VirAddr;
    HI_S32 *ps32Roi = (HI_S32 *)pstDstRoi->u64VirAddr;
    HI_S32 *ps32ClassRoiNum = (HI_S32 *)pstClassRoiNum->u64VirAddr;
    HI_U32 u32ClassNum = pstClassRoiNum->unShape.stWhc.u32Width;
    HI_S32 s32XMin = 0, s32YMin = 0, s32XMax = 0, s32YMax = 0;

    u32RoiNumBias += ps32ClassRoiNum[0];
    cv::Mat img = cv::imread(image_file);
    if (img.empty())
    {
        std::cerr << "failed to read image file "
                  << image_file
                  << "\n";
        return -1;
    }
    int raw_h = img.size().height;
    int raw_w = img.size().width;
    printf("raw_h:%d raw_w:%d \n", raw_h, raw_w);
    cv::resize(img, img, cv::Size(416, 416), 0, 0, cv::INTER_LINEAR);

    for (i = 1; i < u32ClassNum; i++)
    {
        u32ScoreBias = u32RoiNumBias;
        u32BboxBias = u32RoiNumBias * TEST_NNIE_COORDI_NUM;
        /*if the confidence score greater than result threshold, the result will be printed*/
        if ((HI_FLOAT)ps32Score[u32ScoreBias] / TEST_NNIE_QUANT_BASE >=
                f32PrintResultThresh &&
            ps32ClassRoiNum[i] != 0)
        {
            printf("==== The %dth class box info====\n", i);
        }
        for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++)
        {
            f32Score = (HI_FLOAT)ps32Score[u32ScoreBias + j] / TEST_NNIE_QUANT_BASE;
            if (f32Score < f32PrintResultThresh)
            {
                break;
            }
            s32XMin = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM];
            s32YMin = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 1];
            s32XMax = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 2];
            s32YMax = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 3];
            printf("%d %d %d %d %f\n", s32XMin, s32YMin, s32XMax, s32YMax, f32Score);
            cv::rectangle(img, Point(s32XMin, s32YMin), Point(s32XMax, s32YMax), Scalar(255 / i, 0, 139, 255), 2);
        }
        u32RoiNumBias += ps32ClassRoiNum[i];
    }
    cv::resize(img, img, cv::Size(raw_w, raw_h), 0, 0, cv::INTER_LINEAR);
    cv::imwrite("Yolov3_out.jpg", img);
    printf("write Yolov3_out.jpg successful!\n");
    return HI_SUCCESS;
}

void TEST_NNIE_Yolov3()
{
 //  const char *image_file = "./data/nnie_image/rgb_planar/dog_bike_car_416x416.bgr";
    const char *image_file_org = "./data/nnie_image/rgb_planar/dog_bike_car.jpg";
 //   const char *model_file = "./data/nnie_model/detection/inst_yolov3_cycle.wk";
	const char *image_file = "./dog_bike_car_416x416.bgr";
    const char *model_file = "./inst_yolov3_cycle.wk";
    struct timeval t0, t1;
    /* prepare input data */
    struct stat statbuf;
    stat(image_file, &statbuf);
    int input_length = statbuf.st_size;

    void *input_data = malloc(input_length);
    if (!get_input_data(image_file, input_data, input_length))
        return;
    context_t nnie_context = nullptr;
    graph_t graph = create_graph(nnie_context, "nnie", model_file, "noconfig");

    if (graph == nullptr)
    {
        std::cout << "Create graph failed errno: " << get_tengine_errno() << std::endl;
        return;
    }
    // dump_graph(graph);
    prerun_graph(graph);

    gettimeofday(&t0, NULL);
    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if (input_tensor == nullptr)
    {
        std::cout << "Cannot find input tensor, node_idx: " << node_idx << ",tensor_idx: " << tensor_idx << "\n";
        return;
    }
    /* setup input buffer */
    if (set_tensor_buffer(input_tensor, input_data, input_length) < 0)
    {
        std::cout << "Set data for input tensor failed\n";
        return;
    }

    gettimeofday(&t1, NULL);
    float mytime_0 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << "\n mytime_0 " << mytime_0 << " ms\n";
    std::cout << "--------------------------------------\n";

    /* run the graph */
    float avg_time = 0.f;
    for (int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        if (run_graph(graph, 1) < 0)
        {
            std::cerr << "Run graph failed\n";
            return;
        }
        gettimeofday(&t1, NULL);

        float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
    }
    std::cout << "Model file : " << model_file << "\n"
              << "image file : " << image_file << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";
    std::cout << "--------------------------------------\n";

    gettimeofday(&t0, NULL);
    TEST_NNIE_YOLOV3_SOFTWARE_PARAM_S stSoftWareParam;
    TEST_NNIE_Yolov3_SoftwareInit(graph, &stSoftWareParam);
    TEST_NNIE_Yolov3_GetResult(graph, &stSoftWareParam);
    gettimeofday(&t1, NULL);
    float mytime_2 = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout << "\n mytime_2 " << mytime_2 << " ms\n";
    std::cout << "--------------------------------------\n";

    printf("print result, this sample has 81 classes:\n");
    printf("class 0:background      class 1:person       class 2:bicycle         class 3:car            class 4:motorbike      class 5:aeroplane\n");
    printf("class 6:bus             class 7:train        class 8:truck           class 9:boat           class 10:traffic light\n");
    printf("class 11:fire hydrant   class 12:stop sign   class 13:parking meter  class 14:bench         class 15:bird\n");
    printf("class 16:cat            class 17:dog         class 18:horse          class 19:sheep         class 20:cow\n");
    printf("class 21:elephant       class 22:bear        class 23:zebra          class 24:giraffe       class 25:backpack\n");
    printf("class 26:umbrella       class 27:handbag     class 28:tie            class 29:suitcase      class 30:frisbee\n");
    printf("class 31:skis           class 32:snowboard   class 33:sports ball    class 34:kite          class 35:baseball bat\n");
    printf("class 36:baseball glove class 37:skateboard  class 38:surfboard      class 39:tennis racket class 40bottle\n");
    printf("class 41:wine glass     class 42:cup         class 43:fork           class 44:knife         class 45:spoon\n");
    printf("class 46:bowl           class 47:banana      class 48:apple          class 49:sandwich      class 50orange\n");
    printf("class 51:broccoli       class 52:carrot      class 53:hot dog        class 54:pizza         class 55:donut\n");
    printf("class 56:cake           class 57:chair       class 58:sofa           class 59:pottedplant   class 60bed\n");
    printf("class 61:diningtable    class 62:toilet      class 63:vmonitor       class 64:laptop        class 65:mouse\n");
    printf("class 66:remote         class 67:keyboard    class 68:cell phone     class 69:microwave     class 70:oven\n");
    printf("class 71:toaster        class 72:sink        class 73:refrigerator   class 74:book          class 75:clock\n");
    printf("class 76:vase           class 77:scissors    class 78:teddy bear     class 79:hair drier    class 80:toothbrush\n");
    printf("Yolov3 result:\n");
    HI_FLOAT f32PrintResultThresh = 0.8f;
    TEST_NNIE_Detection_Yolov3_PrintResult(&stSoftWareParam.stDstScore,
                                           &stSoftWareParam.stDstRoi, &stSoftWareParam.stClassRoiNum, f32PrintResultThresh, image_file_org);
    HI_MPI_SYS_MmzFree(stSoftWareParam.stGetResultTmpBuf.u64PhyAddr, (void *)stSoftWareParam.stGetResultTmpBuf.u64VirAddr);
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);
}

void TEST_NNIE_Lstm()
{
    const char *image_file[3] = {"./data/nnie_image/vector/Seq.SEQ_S32",
                                 "./data/nnie_image/vector/Vec1.VEC_S32",
                                 "./data/nnie_image/vector/Vec2.VEC_S32"};
    const char *model_file = "./data/nnie_model/recurrent/lstm_3_3.wk";
    /* prepare input data */
    struct stat statbuf;
    int input_length[3] = {0};

    stat(image_file[0], &statbuf);
    input_length[0] = statbuf.st_size;
    stat(image_file[1], &statbuf);
    input_length[1] = statbuf.st_size;
    stat(image_file[2], &statbuf);
    input_length[2] = statbuf.st_size;

    void *input_data[3];
    input_data[0] = malloc(input_length[0]);
    input_data[1] = malloc(input_length[1]);
    input_data[2] = malloc(input_length[2]);
    if (!get_input_data(image_file[0], input_data[0], input_length[0]))
        return;
    if (!get_input_data(image_file[1], input_data[1], input_length[1]))
        return;
    if (!get_input_data(image_file[2], input_data[2], input_length[2]))
        return;

    context_t nnie_context = nullptr;
    graph_t graph = create_graph(nnie_context, "nnie", model_file, "noconfig");

    if (graph == nullptr)
    {
        std::cout << "Create graph failed errno: " << get_tengine_errno() << std::endl;
        return;
    }
    // dump_graph(graph);

    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if (input_tensor == nullptr)
    {
        std::cout << "Cannot find input tensor, node_idx: " << node_idx << ",tensor_idx: " << tensor_idx << "\n";
        return;
    }

    /* setup input buffer */
    if (set_tensor_buffer(input_tensor, input_data[0], input_length[0]) < 0)
    {
        std::cout << "Set data for input tensor failed\n";
        return;
    }

    node_idx = 1;
    tensor_idx = 0;
    input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if (input_tensor == nullptr)
    {
        std::cout << "Cannot find input tensor, node_idx: " << node_idx << ",tensor_idx: " << tensor_idx << "\n";
        return;
    }

    /* setup input buffer */
    if (set_tensor_buffer(input_tensor, input_data[1], input_length[1]) < 0)
    {
        std::cout << "Set data for input tensor failed\n";
        return;
    }

    node_idx = 2;
    tensor_idx = 0;
    input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if (input_tensor == nullptr)
    {
        std::cout << "Cannot find input tensor, node_idx: " << node_idx << ",tensor_idx: " << tensor_idx << "\n";
        return;
    }

    /* setup input buffer */
    if (set_tensor_buffer(input_tensor, input_data[2], input_length[2]) < 0)
    {
        std::cout << "Set data for input tensor failed\n";
        return;
    }

    prerun_graph(graph);
    /* run the graph */
    struct timeval t0, t1;
    float avg_time = 0.f;
    for (int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        if (run_graph(graph, 1) < 0)
        {
            std::cerr << "Run graph failed\n";
            return;
        }
        gettimeofday(&t1, NULL);

        float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
    }
    std::cout << "Model file : " << model_file << "\n"
              << "image file : " << image_file << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";
    std::cout << "--------------------------------------\n";

    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data[0]);
    free(input_data[1]);
    free(input_data[2]);
}

void TEST_NNIE_Pvanet()
{
    const char *image_file = "./data/nnie_image/rgb_planar/horse_dog_car_person_224x224.bgr";
    const char *model_file = "./data/nnie_model/detection/inst_fasterrcnn_pvanet_inst.wk";
    /* prepare input data */
    struct stat statbuf;
    stat(image_file, &statbuf);
    int input_length = statbuf.st_size;

    void *input_data = malloc(input_length);
    if (!get_input_data(image_file, input_data, input_length))
        return;

    context_t nnie_context = nullptr;
    graph_t graph = create_graph(nnie_context, "nnie", model_file, "noconfig");

    if (graph == nullptr)
    {
        std::cout << "Create graph failed errno: " << get_tengine_errno() << std::endl;
        return;
    }
    dump_graph(graph);

    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if (input_tensor == nullptr)
    {
        std::cout << "Cannot find input tensor, node_idx: " << node_idx << ",tensor_idx: " << tensor_idx << "\n";
        return;
    }

    /* setup input buffer */
    if (set_tensor_buffer(input_tensor, input_data, input_length) < 0)
    {
        std::cout << "Set data for input tensor failed\n";
        return;
    }

    prerun_graph(graph);
    struct timeval t0, t1;
    float avg_time = 0.f;
    for (int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        if (run_graph(graph, 1) < 0)
        {
            std::cerr << "Run graph failed\n";
            return;
        }
        gettimeofday(&t1, NULL);

        float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
    }
    std::cout << "Model file : " << model_file << "\n"
              << "image file : " << image_file << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";
    std::cout << "--------------------------------------\n";

    for (int j = 0; j < 1; j++)
    {
        tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
        void *output_data = get_tensor_buffer(output_tensor);
        int output_length = get_tensor_buffer_size(output_tensor);
        printf("j:%d output_data:%p output_length:%d\n", j, output_data, output_length);
        int dims[4];
        int dimssize = 4;
        get_tensor_shape(output_tensor, dims, dimssize);
        printf("%d:%d:%d:%d\n", dims[0], dims[1], dims[2], dims[3]);

        unsigned int i = 0;
        unsigned int *pu32Tmp = NULL;
        unsigned int u32TopN = 10;

        printf("==== The %d tensor info====\n", j);
        pu32Tmp = (unsigned int *)((HI_U64)output_data);
        for (i = 0; i < u32TopN; i++)
        {
            printf("%d:%d\n", i, pu32Tmp[i]);
        }
    }

    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        TEST_Usage(argv[0]);
        return -1;
    }
    int model = 0;
    int res;
    while ((res = getopt(argc, argv, "m:r:h")) != -1)
    {
        switch (res)
        {
        case 'm':
            model = std::strtoul(optarg, NULL, 10);
            break;
        case 'r':
            repeat_count = std::strtoul(optarg, NULL, 10);
            break;
        case 'h':
            std::cout << "[Usage]: " << argv[0] << " [-h]\n"
                      << " [-m model]  [-r repeat_count]\n";
            return 0;
        default:
            break;
        }
    }
    std::cout << "model:" << model << "\n";
    std::cout << "repeat_count:" << repeat_count << "\n";

    init_tengine();
    std::cout << "Tengine version: " << get_tengine_version() << "\n";

    if (load_tengine_plugin("nnieplugin", "libnnieplugin.so", "nnie_plugin_init") != 0)
    {
        std::cout << "load nnie plugin faield.\n";
    }
    std::cout << "load nnie plugin successful.\n";

    switch (model)
    {
    case 2:
    {
        TEST_NNIE_FasterRcnn();
    }
    break;
    case 3:
    {
        TEST_NNIE_Cnn();
    }
    break;
    case 4:
    {
        TEST_NNIE_Ssd();
    }
    break;
    case 5:
    {
        TEST_NNIE_Yolov1();
    }
    break;
    case 6:
    {
        TEST_NNIE_Yolov2();
    }
    break;
    case 7:
    {
        TEST_NNIE_Yolov3();
    }
    break;
    case 8:
    {
        TEST_NNIE_Lstm();
    }
    break;
    case 9:
    {
        TEST_NNIE_Pvanet();
    }
    break;
    default:
    {
        TEST_Usage(argv[0]);
    }
    break;
    }

    release_tengine();
    std::cout << "ALL TEST DONE\n";

    return 0;
}
