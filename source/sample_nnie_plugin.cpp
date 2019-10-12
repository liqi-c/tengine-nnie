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
#include "mpi_sys.h"
#include "mpi_vb.h"
#include "hi_nnie.h"
#include "mpi_nnie.h"
#include "tengine_c_api.h"

const char *image_file = "./test_yu.jpg.rgb";
const char *model_file = "./inst_mnist_cycle.wk";

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
    fclose(fp);
    return true;
}

typedef unsigned long long HI_U64;
int main(int argc, char *argv[])
{
    /* prepare input data */
    struct stat statbuf;
    stat(image_file, &statbuf);
    int input_length = statbuf.st_size;

    void *input_data = malloc(input_length);
    if (!get_input_data(image_file, input_data, input_length))
        return -1;
    std::cout << "input_length: " << input_length << "\n";

    init_tengine();
    //set_log_level(LOG_DEBUG);
    std::cout << "Tengine version: " << get_tengine_version() << "\n";

    if (load_tengine_plugin("nnieplugin", "libnnieplugin.so", "nnie_plugin_init") != 0)
    {
        std::cout << "load nnie plugin faield.\n";
    }
    std::cout << "load nnie plugin successful.\n";
    context_t nnie_context = nullptr;
    graph_t graph = create_graph(nnie_context, "nnie", model_file,  "noconfig");
    std::cout << "create nnie graph successful.\n";
    if (graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return -1;
    }
    dump_graph(graph);

    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if (input_tensor == nullptr)
    {
        std::cout << "Cannot find input tensor, node_idx: " << node_idx << ",tensor_idx: " << tensor_idx << "\n";
        return -1;
    }

    /* setup input buffer */
    if (set_tensor_buffer(input_tensor, input_data, input_length) < 0)
    {
        std::cout << "Set data for input tensor failed\n";
        return -1;
    }

    prerun_graph(graph);
    /* run the graph */
    int loopCount = 1;
    while (loopCount--)
    {
        run_graph(graph, 1);
        sleep(1);
        printf("loopCount:%d\n", loopCount);
    }

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
    release_tengine();

    std::cout << "ALL TEST DONE\n";

    return 0;
}
