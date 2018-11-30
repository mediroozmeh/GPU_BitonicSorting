
//Passed down by clBuildProgram

#define HALF_FILTER 8

#define      H_LOCAL_X 16
#define      H_LOCAL_Y 4
#define   V_LOCAL_X 16
#define   V_LOCAL_Y 8

#define    H_OUT_SIZE 8
#define      H_STRIDE_SIZE 1
#define V_OUT_SIZE 8
#define   V_STRIDE_SIZE 1




#define FULL_FILTER (2 * HALF_FILTER + 1)



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(H_LOCAL_X, H_LOCAL_Y, 1)))
void Horizontalconv(
    __global float *c_out,
    __global float *a_in,
    __constant float *b_in,
    int image_w,
    int image_h,
    int window_size
){
    __local float local_buffer[H_LOCAL_Y][(H_OUT_SIZE + 2 * H_STRIDE_SIZE) * H_LOCAL_X];

    //Offset to the left halo edge
    const int index_x = (get_group_id(0) * H_OUT_SIZE - H_STRIDE_SIZE) * H_LOCAL_X + get_local_id(0);
    const int index_y = get_group_id(1) * H_LOCAL_Y + get_local_id(1);

    a_in += index_y * window_size + index_x;
    c_out += index_y * window_size + index_x;

    //Load main data
    for(int i = H_STRIDE_SIZE; i < H_STRIDE_SIZE + H_OUT_SIZE; i++)
        local_buffer[get_local_id(1)][get_local_id(0) + i * H_LOCAL_X] = a_in[i * H_LOCAL_X];

    //Load left halo
    for(int i = 0; i < H_STRIDE_SIZE; i++)
        local_buffer[get_local_id(1)][get_local_id(0) + i * H_LOCAL_X]  = (index_x + i * H_LOCAL_X >= 0) ? a_in[i * H_LOCAL_X] : 0;

    //Load right halo
    for(int i = H_STRIDE_SIZE + H_OUT_SIZE; i < H_STRIDE_SIZE + H_OUT_SIZE + H_STRIDE_SIZE; i++)
        local_buffer[get_local_id(1)][get_local_id(0) + i * H_LOCAL_X]  = (index_x + i * H_LOCAL_X < image_w) ? a_in[i * H_LOCAL_X] : 0;

    //Compute and store results
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = H_STRIDE_SIZE; i < H_STRIDE_SIZE + H_OUT_SIZE; i++){
        float sum = 0;

        for(int j = -HALF_FILTER; j <= HALF_FILTER; j++)
            sum += b_in[HALF_FILTER - j] * local_buffer[get_local_id(1)][get_local_id(0) + i * H_LOCAL_X + j];

        c_out[i * H_LOCAL_X] = sum;
    }
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(V_LOCAL_X, V_LOCAL_Y, 1)))
void Verticalconv(
    __global float *c_out,
    __global float *a_in,
    __constant float *b_in,
    int image_w,
    int image_h,
    int window_size
){
    __local float local_buffer[V_LOCAL_X][(V_OUT_SIZE + 2 * V_STRIDE_SIZE) * V_LOCAL_Y + 1];

    //Offset to the upper halo edge
    const int index_x = get_group_id(0) * V_LOCAL_X + get_local_id(0);
    const int index_y = (get_group_id(1) * V_OUT_SIZE - V_STRIDE_SIZE) * V_LOCAL_Y + get_local_id(1);
    a_in += index_y * window_size + index_x;
    c_out += index_y * window_size + index_x;

    //Load main data
    for(int i = V_STRIDE_SIZE; i < V_STRIDE_SIZE + V_OUT_SIZE; i++)
        local_buffer[get_local_id(0)][get_local_id(1) + i * V_LOCAL_Y] = a_in[i * V_LOCAL_Y * window_size];

    //Load upper halo
    for(int i = 0; i < V_STRIDE_SIZE; i++)
        local_buffer[get_local_id(0)][get_local_id(1) + i * V_LOCAL_Y] = (index_y + i * V_LOCAL_Y >= 0) ? a_in[i * V_LOCAL_Y * window_size] : 0;

    //Load lower halo
    for(int i = V_STRIDE_SIZE + V_OUT_SIZE; i < V_STRIDE_SIZE + V_OUT_SIZE + V_STRIDE_SIZE; i++)
        local_buffer[get_local_id(0)][get_local_id(1) + i * V_LOCAL_Y]  = (index_y + i * V_LOCAL_Y < image_h) ? a_in[i * V_LOCAL_Y * window_size] : 0;

    //Compute and store results
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = V_STRIDE_SIZE; i < V_STRIDE_SIZE + V_OUT_SIZE; i++){
        float sum = 0;

        for(int j = -HALF_FILTER; j <= HALF_FILTER; j++)
            sum += b_in[HALF_FILTER - j] * local_buffer[get_local_id(0)][get_local_id(1) + i * V_LOCAL_Y + j];

        c_out[i * V_LOCAL_Y * window_size] = sum;
    }
}


